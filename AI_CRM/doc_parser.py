"""
doc_parser.py - 文档解析增强模块（可插拔：纯文本 / OCR / 版面分析 / 表格抽取）
================================================================================
背景（对应二面整改「文档解析增强」）：
  项目原来的入库路径只有 TextLoader + RecursiveCharacterTextSplitter：
    · 只吃纯文本，扫描件（图片型 PDF）直接 ingest 不到内容；
    · 表格被拍平成文本，行列结构丢失，检索容易串行串列；
    · 分块不区分内容类型，标题/正文/表格一视同仁。
  本模块把「解析」拆成可插拔的多后端流水线：
    1. 按扩展名选择解析后端（txt/md、pdf、docx、csv、图片…）
    2. PDF 优先走「文本层 + 版面（逐页）」；若可检测到扫描页，可选走 OCR
    3. 表格统一抽成 kind="table" 的块，保留行列结构（Markdown 表格）
    4. 分块按内容类型区分：正文按长度递归切；表格整块保留（超长按行切且重复表头）
    5. 每个块都带 page / source / kind / section / table_id 等元数据

设计原则：
  - 零强制依赖：所有第三方库（pypdf / pdfplumber / python-docx / pytesseract /
    pdf2image）全部「惰性导入」，缺哪个就降级或跳过，并在 warnings 里说明；
  - 纯标准库即可跑通 txt/md/csv 与 docx 兜底解析；
  - 解析/分块失败不影响主流程，异常向上抛由调用方兜底。

用法：
  from doc_parser import parse_and_chunk
  doc, chunks = parse_and_chunk("课程介绍.pdf")
  for c in chunks:
      print(c.metadata, c.text[:40])

自检：
  python doc_parser.py
"""

# ==================== 1. 导入依赖 ====================
from __future__ import annotations

import csv
import io
import os
import re
import time
import zipfile
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

try:  # 指标埋点（可选依赖）
    from metrics import incr as _metric_incr, observe as _metric_observe
except Exception:  # pragma: no cover
    def _metric_incr(*args, **kwargs):
        return None

    def _metric_observe(*args, **kwargs):
        return None


# ==================== 2. 数据结构 ====================
@dataclass
class Block:
    """解析出的最小内容单元（正文段落 / 标题 / 表格 / 图片）"""
    kind: str                       # text | title | table | image
    content: str
    page: int = 0                   # 所在页码（0 表示未知）
    meta: Dict[str, object] = field(default_factory=dict)


@dataclass
class ParsedDocument:
    """一份被解析后的文档（含多类型块 + 解析过程说明）"""
    source: str                     # 文件名
    path: str                       # 完整路径
    ext: str                        # 扩展名（小写，含点）
    blocks: List[Block] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    used_backends: List[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        """把所有正文/标题块拼成纯文本（不含表格）"""
        return "\n".join(b.content for b in self.blocks if b.kind in ("text", "title"))

    @property
    def tables(self) -> List[Block]:
        return [b for b in self.blocks if b.kind == "table"]

    def stats(self) -> dict:
        n_table = len(self.tables)
        return {
            "source": self.source,
            "ext": self.ext,
            "blocks": len(self.blocks),
            "text_blocks": sum(1 for b in self.blocks if b.kind == "text"),
            "tables": n_table,
            "pages": max((b.page for b in self.blocks), default=0),
            "backends": self.used_backends,
            "warnings": self.warnings,
        }


@dataclass
class Chunk:
    """入库分块（text + 元数据）"""
    text: str
    metadata: Dict[str, object] = field(default_factory=dict)


# ==================== 3. 工具函数 ====================
_TEXT_ENCODINGS = ("utf-8", "utf-8-sig", "gbk", "gb18030", "latin-1")


def _read_text_file(path: str) -> str:
    """尽力而为地按常见编码读取文本文件"""
    with open(path, "rb") as fh:
        raw = fh.read()
    for enc in _TEXT_ENCODINGS:
        try:
            return raw.decode(enc)
        except Exception:
            continue
    return raw.decode("utf-8", errors="replace")


def _clean(text: str) -> str:
    """基础清洗：统一换行、压缩 3+ 连续空行、去首尾空白"""
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _rows_to_markdown(rows: List[List[str]], header: Optional[List[str]] = None) -> str:
    """
    把二维表转成 Markdown 表格（保留行列结构，便于检索与展示）
    ==========================================================
    第一行作为表头；用 | 分隔，第二行插入分隔线。
    """
    rows = [[str(c or "").replace("\n", " ").strip() for c in row] for row in rows]
    if not rows:
        return ""
    if header is None:
        header = rows[0]
        body = rows[1:]
    else:
        body = rows
    head = "| " + " | ".join(header) + " |"
    sep = "| " + " | ".join("---" for _ in header) + " |"
    lines = [head, sep]
    for row in body:
        # 对齐列数
        row = (row + [""] * len(header))[:len(header)]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _is_scanned_pdf_page(page_text: str) -> bool:
    """版面启发式：某页几乎无文本层 → 大概率是扫描页，需要 OCR"""
    return len((page_text or "").strip()) < 20


# ==================== 4. 各类型解析后端 ====================
def parse_text(path: str, _ctx: dict) -> List[Block]:
    """纯文本 / Markdown：按空行切段（标题行识别为 title）"""
    raw = _clean(_read_text_file(path))
    blocks: List[Block] = []
    for para in re.split(r"\n\s*\n", raw):
        para = para.strip()
        if not para:
            continue
        is_title = bool(re.match(r"^#{1,6}\s", para)) or (
            len(para) <= 40 and "\n" not in para and para[-1:] in "：:"
        )
        blocks.append(Block(kind="title" if is_title else "text", content=para))
    return blocks


def parse_pdf(path: str, ctx: dict) -> List[Block]:
    """
    PDF：优先「文本层 + 逐页版面」；可选对扫描页做 OCR
    ===================================================
    依赖（惰性）：
      - pypdf 或 PyPDF2（文本层）
      - pdfplumber（表格抽取，可选）
      - pdf2image + pytesseract（OCR，可选；需系统装好 poppler / tesseract）
    """
    blocks: List[Block] = []
    reader = None
    for mod_name in ("pypdf", "PyPDF2"):
        try:
            mod = __import__(mod_name)
            reader = mod.PdfReader(path)
            ctx.setdefault("backends", []).append(f"{mod_name}:text")
            break
        except Exception:
            continue
    if reader is None:
        ctx.setdefault("warnings", []).append("缺少 pypdf/PyPDF2，无法解析 PDF 文本层")
        return blocks

    # -------- 文本层（逐页，带版面信息）--------
    page_texts: List[str] = []
    for i, page in enumerate(reader.pages, 1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        page_texts.append(text)
        text_clean = _clean(text)
        if text_clean:
            blocks.append(Block(kind="text", content=text_clean, page=i))

    # -------- 表格抽取（pdfplumber，可选）--------
    if ctx.get("extract_tables", True):
        try:
            import pdfplumber  # type: ignore

            ctx.setdefault("backends", []).append("pdfplumber:table")
            with pdfplumber.open(path) as pdf:
                for i, page in enumerate(pdf.pages, 1):
                    try:
                        for t_idx, table in enumerate(page.extract_tables() or [], 1):
                            md = _rows_to_markdown(table)
                            if md:
                                blocks.append(Block(
                                    kind="table", content=md, page=i,
                                    meta={"table_id": f"p{i}-t{t_idx}"},
                                ))
                    except Exception as exc:
                        ctx.setdefault("warnings", []).append(f"第{i}页表格抽取失败: {exc}")
        except Exception:
            ctx.setdefault("warnings", []).append("未安装 pdfplumber，跳过 PDF 表格抽取")

    # -------- OCR（扫描页，可选）--------
    ocr_enabled = ctx.get("ocr", os.getenv("DOC_OCR_ENABLE", "0") == "1")
    if ocr_enabled and any(_is_scanned_pdf_page(t) for t in page_texts):
        blocks.extend(_pdf_ocr(path, page_texts, ctx))
    elif any(_is_scanned_pdf_page(t) for t in page_texts):
        ctx.setdefault("warnings", []).append(
            "检测到疑似扫描页（文本层为空），如需解析请开启 DOC_OCR_ENABLE=1 并安装 OCR 依赖"
        )
    return blocks


def _pdf_ocr(path: str, page_texts: List[str], ctx: dict) -> List[Block]:
    """对文本层为空的页面做 OCR（pdf2image + pytesseract，缺依赖则降级）"""
    out: List[Block] = []
    try:
        from pdf2image import convert_from_path  # type: ignore
        import pytesseract  # type: ignore
    except Exception as exc:
        ctx.setdefault("warnings", []).append(f"OCR 依赖缺失（pdf2image/pytesseract）: {exc}")
        return out

    try:
        images = convert_from_path(path, dpi=200)
        ctx.setdefault("backends", []).append("ocr:pytesseract")
        for i, (img, text_layer) in enumerate(zip(images, page_texts), 1):
            if not _is_scanned_pdf_page(text_layer):
                continue                       # 有文本层的页不重复 OCR
            try:
                ocr_text = _clean(pytesseract.image_to_string(img, lang="chi_sim+eng"))
                if ocr_text:
                    out.append(Block(kind="text", content=ocr_text, page=i,
                                     meta={"ocr": True}))
            except Exception as exc:
                ctx.setdefault("warnings", []).append(f"第{i}页 OCR 失败: {exc}")
    except Exception as exc:
        ctx.setdefault("warnings", []).append(f"OCR 渲染失败: {exc}")
    return out


def parse_docx(path: str, ctx: dict) -> List[Block]:
    """
    DOCX：段落（标题识别）+ 表格结构抽取
    =====================================
    优先 python-docx；缺失时退化为「解压 XML + 正则」(纯标准库)。
    """
    blocks: List[Block] = []
    try:
        import docx  # type: ignore  # python-docx

        ctx.setdefault("backends", []).append("python-docx")
        document = docx.Document(path)
        for para in document.paragraphs:
            text = _clean(para.text)
            if not text:
                continue
            style = (getattr(para.style, "name", "") or "").lower()
            is_title = "heading" in style or "标题" in style
            blocks.append(Block(kind="title" if is_title else "text", content=text))
        for t_idx, table in enumerate(document.tables, 1):
            rows = [[cell.text for cell in row.cells] for row in table.rows]
            md = _rows_to_markdown(rows)
            if md:
                blocks.append(Block(kind="table", content=md,
                                    meta={"table_id": f"docx-t{t_idx}"}))
        return blocks
    except Exception:
        pass

    # -------- 兜底：解压 XML + 正则（纯标准库）--------
    try:
        ctx.setdefault("backends", []).append("docx:zip-xml-fallback")
        with zipfile.ZipFile(path) as zf:
            xml = zf.read("word/document.xml").decode("utf-8", errors="ignore")
        xml = xml.replace("</w:p>", "\n</w:p>")
        texts = re.findall(r"<w:t[^>]*>(.*?)</w:t>", xml, re.S)
        for t in texts:
            t = _clean(re.sub(r"<[^>]+>", "", t))
            if t:
                blocks.append(Block(kind="text", content=t))
        ctx.setdefault("warnings", []).append("未安装 python-docx，表格结构未抽取")
    except Exception as exc:
        ctx.setdefault("warnings", []).append(f"DOCX 解析失败: {exc}")
    return blocks


def parse_csv(path: str, ctx: dict) -> List[Block]:
    """CSV：整体作为一张表格块（保留行列结构）"""
    raw = _read_text_file(path)
    rows = [row for row in csv.reader(io.StringIO(raw)) if any(c.strip() for c in row)]
    md = _rows_to_markdown(rows)
    return [Block(kind="table", content=md, meta={"table_id": "csv-1"})] if md else []


def parse_image(path: str, ctx: dict) -> List[Block]:
    """图片：OCR（pytesseract，缺失则降级并给出提示）"""
    try:
        import pytesseract  # type: ignore
        from PIL import Image  # type: ignore

        ctx.setdefault("backends", []).append("ocr:pytesseract")
        text = _clean(pytesseract.image_to_string(Image.open(path), lang="chi_sim+eng"))
        return [Block(kind="text", content=text, meta={"ocr": True})] if text else []
    except Exception as exc:
        ctx.setdefault("warnings", []).append(f"图片 OCR 不可用（pytesseract/Pillow）: {exc}")
        return []


# ==================== 5. 分块（按内容类型区分） ====================
_DEFAULT_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]


def _split_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    轻量递归切分（等价于 RecursiveCharacterTextSplitter 的简化版，零依赖）
    ======================================================================
    先按段落，再按句子，最后按字符；相邻块保留 overlap 字符重叠。
    """
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    def _split(s: str, sep_idx: int) -> List[str]:
        if sep_idx >= len(_DEFAULT_SEPARATORS):
            return [s[i:i + chunk_size] for i in range(0, len(s), max(1, chunk_size - overlap))]
        sep = _DEFAULT_SEPARATORS[sep_idx]
        if sep and sep in s:
            parts = s.split(sep)
            pieces: List[str] = []
            for p in parts:
                pieces.append(p + (sep if sep else ""))
            out: List[str] = []
            buf = ""
            for piece in pieces:
                if len(buf) + len(piece) <= chunk_size:
                    buf += piece
                else:
                    if buf:
                        out.append(buf)
                    if len(piece) > chunk_size:
                        out.extend(_split(piece, sep_idx + 1))
                        buf = ""
                    else:
                        buf = piece
            if buf:
                out.append(buf)
            return out
        return _split(s, sep_idx + 1)

    chunks = _split(text, 0)
    # 加重叠
    if overlap <= 0 or len(chunks) <= 1:
        return [c for c in chunks if c.strip()]
    out = []
    for i, c in enumerate(chunks):
        if i > 0:
            c = chunks[i - 1][-overlap:] + c
        out.append(c)
    return [c for c in out if c.strip()]


def chunk_document(
    doc: ParsedDocument,
    chunk_size: int = 500,
    overlap: int = 100,
) -> List[Chunk]:
    """
    把 ParsedDocument 切成入库块
    ==============================
    规则（按内容类型区分）：
      · 正文/标题：累积后按长度递归切，块元数据带上当前 section（最近的标题）
      · 表格    ：整块保留（检索时不破坏行列）；超长则按行切且每块重复表头
    每个块元数据：source / file_path / kind / page / section / table_id(可选)
    """
    chunks: List[Chunk] = []
    base_meta = {"source": doc.source, "file_path": doc.path, "ext": doc.ext}

    section = ""
    buf: List[str] = []
    buf_page = 0

    def _flush() -> None:
        nonlocal buf, buf_page
        if not buf:
            return
        merged = "\n".join(buf)
        for piece in _split_text(merged, chunk_size, overlap):
            chunks.append(Chunk(
                text=piece,
                metadata={**base_meta, "kind": "text",
                          "page": buf_page, "section": section},
            ))
        buf = []

    for blk in doc.blocks:
        if blk.kind == "title":
            _flush()
            section = blk.content[:60]
            continue
        if blk.kind == "table":
            _flush()
            chunks.extend(_chunk_table(blk, base_meta, section, chunk_size))
            continue
        # 普通正文
        if buf and buf_page != blk.page:
            _flush()
        buf_page = blk.page or buf_page
        buf.append(blk.content)

    _flush()
    return chunks


def _chunk_table(blk: Block, base_meta: dict, section: str, chunk_size: int) -> List[Chunk]:
    """表格分块：优先整块保留；超长按行切且重复表头"""
    meta = {**base_meta, "kind": "table", "page": blk.page, "section": section}
    if blk.meta.get("table_id"):
        meta["table_id"] = blk.meta["table_id"]
    if len(blk.content) <= max(chunk_size, 400):
        return [Chunk(text=blk.content, metadata=meta)]

    lines = blk.content.split("\n")
    header = lines[:2] if len(lines) >= 2 else lines[:1]
    body = lines[2:] if len(lines) >= 2 else []
    out: List[Chunk] = []
    buf: List[str] = []
    size = len("\n".join(header))
    for row in body:
        if buf and size + len(row) > chunk_size:
            out.append(Chunk(text="\n".join(header + buf), metadata=dict(meta)))
            buf, size = [], len("\n".join(header))
        buf.append(row)
        size += len(row) + 1
    if buf:
        out.append(Chunk(text="\n".join(header + buf), metadata=dict(meta)))
    return out


# ==================== 6. 对外入口 ====================
class DocumentParser:
    """
    可插拔文档解析器
    ==================
    - 按扩展名分派后端；backend 可注册/替换（便于测试与扩展）
    - parse() 内部计时并埋点 doc_parse_ms / doc_parse_total
    """

    def __init__(self, ocr: bool = False, extract_tables: bool = True):
        self.ocr = ocr
        self.extract_tables = extract_tables
        self.backends: Dict[str, Callable[[str, dict], List[Block]]] = {
            ".txt": parse_text,
            ".md": parse_text,
            ".markdown": parse_text,
            ".pdf": parse_pdf,
            ".docx": parse_docx,
            ".csv": parse_csv,
            ".png": parse_image,
            ".jpg": parse_image,
            ".jpeg": parse_image,
            ".bmp": parse_image,
        }

    def register(self, ext: str, backend: Callable[[str, dict], List[Block]]) -> None:
        """注册 / 覆盖某个扩展名的解析后端"""
        self.backends[ext.lower()] = backend

    def supported_extensions(self) -> List[str]:
        return sorted(self.backends.keys())

    def parse(self, path: str) -> ParsedDocument:
        """解析单个文件为 ParsedDocument（不抛解析异常，降级写 warnings）"""
        ext = os.path.splitext(path)[1].lower()
        doc = ParsedDocument(
            source=os.path.basename(path), path=path, ext=ext,
        )
        ctx: dict = {"ocr": self.ocr, "extract_tables": self.extract_tables}
        started = time.perf_counter()
        backend = self.backends.get(ext)
        try:
            if backend is None:
                doc.warnings.append(f"不支持的扩展名: {ext}（可 register() 扩展）")
            else:
                doc.blocks = backend(path, ctx) or []
        except Exception as exc:
            doc.warnings.append(f"解析异常: {exc}")
        finally:
            elapsed = (time.perf_counter() - started) * 1000.0
            _metric_observe("doc_parse_ms", elapsed, ext=ext or "unknown")
            _metric_incr("doc_parse_total",
                         ext=ext or "unknown",
                         status="ok" if doc.blocks else "empty")
        doc.warnings.extend(ctx.get("warnings", []))
        doc.used_backends = ctx.get("backends", [])
        return doc

    def chunk(
        self, doc: ParsedDocument, chunk_size: int = 500, overlap: int = 100
    ) -> List[Chunk]:
        return chunk_document(doc, chunk_size=chunk_size, overlap=overlap)

    def parse_and_chunk(
        self, path: str, chunk_size: int = 500, overlap: int = 100
    ) -> Tuple[ParsedDocument, List[Chunk]]:
        doc = self.parse(path)
        return doc, self.chunk(doc, chunk_size=chunk_size, overlap=overlap)


def parse_and_chunk(
    path: str, chunk_size: int = 500, overlap: int = 100, ocr: bool = False
) -> Tuple[ParsedDocument, List[Chunk]]:
    """便捷函数：解析 + 分块一步到位"""
    return DocumentParser(ocr=ocr).parse_and_chunk(path, chunk_size, overlap)


# ==================== 7. 自检 ====================
if __name__ == "__main__":  # pragma: no cover
    import tempfile

    tmpdir = tempfile.mkdtemp()
    txt_path = os.path.join(tmpdir, "课程介绍.md")
    with open(txt_path, "w", encoding="utf-8") as fh:
        fh.write(
            "# 初三数学一对一课程\n\n"
            "课时费按小时计费，支持一对一定制方案，可按月购买。\n\n"
            "## 师资说明\n师资均为重点师范院校毕业，平均教龄5年以上。\n"
        )

    csv_path = os.path.join(tmpdir, "班型价格.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["班型", "课时", "价格"])
        writer.writerow(["一对一", "40", "300"])
        writer.writerow(["小班(6人)", "40", "150"])

    parser = DocumentParser()
    print("=" * 68)
    print("doc_parser.py 自检")
    print("=" * 68)
    for p in (txt_path, csv_path):
        doc, chunks = parser.parse_and_chunk(p)
        print(f"\n[文件] {doc.source}")
        print(f"  stats    : {doc.stats()}")
        for c in chunks:
            print(f"  - kind={c.metadata.get('kind')} page={c.metadata.get('page')} "
                  f"section={c.metadata.get('section')!r}")
            print(f"    text: {c.text[:60]!r}")
    print("=" * 68)
