"""
test_rag_ingest.py - RAG 入库流水线（doc_parser 集成）回归测试
=============================================================
验证 rag_law.RAGLaw.add_file 已接入可插拔解析器：
  · txt/md 走 doc_parser，块元数据带 kind/page/section/source
  · csv 表格被抽成 kind="table" 的结构化块
  · 解析器缺失时回退 TextLoader（此处通过注入伪解析器近似验证回退分支）

设计：用 object.__new__(RAGLaw) 绕过重型 __init__（不加载嵌入/向量库），
      注入「假向量库 + 假分块器」，只测 add_file 的编排逻辑。

运行：
  python test_rag_ingest.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag_law import RAGLaw  # noqa: E402
from doc_parser import DocumentParser  # noqa: E402


class _FakeVectorStore:
    """记录被写入的文档块，替代 Chroma（避免加载嵌入模型）"""

    def __init__(self):
        self.docs = []

    def add_documents(self, docs):
        self.docs.extend(docs)


class _FakeSplitter:
    """极简分块器：仅用于回退分支（每块直接返回整段）"""

    def split_documents(self, documents):
        return list(documents)


def _build_rag(doc_parser) -> RAGLaw:
    rag = object.__new__(RAGLaw)          # 绕过 __init__，不加载嵌入 / Chroma
    rag.vector_store = _FakeVectorStore()
    rag.text_splitter = _FakeSplitter()
    rag.doc_parser = doc_parser
    rag._rebuild_bm25 = lambda: None       # BM25 重建打桩
    return rag


def main() -> int:
    passed = failed = 0

    def check(title, cond, extra=""):
        nonlocal passed, failed
        if cond:
            passed += 1
            print(f"  [PASS] {title}")
        else:
            failed += 1
            print(f"  [FAIL] {title} {extra}")

    tmpdir = tempfile.mkdtemp()

    # ---------- 用例 1：Markdown（含标题 / 正文）----------
    md_path = os.path.join(tmpdir, "课程介绍.md")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(
            "# 初三数学一对一课程\n\n"
            "课时费按小时计费，支持一对一定制方案，可按月购买。\n\n"
            "## 师资说明\n师资均为重点师范院校毕业，平均教龄5年以上。\n"
        )

    # ---------- 用例 2：CSV（表格抽取）----------
    csv_path = os.path.join(tmpdir, "班型价格.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
        fh.write("班型,课时,价格\n一对一,40,300\n小班(6人),40,150\n")

    print("=" * 68)
    print("RAG 入库流水线回归测试（doc_parser 集成）")
    print("=" * 68)

    # ===== 解析器可用：走 doc_parser =====
    rag = _build_rag(DocumentParser())

    print("\n[1] add_file(.md) → doc_parser 路径")
    n_md = rag.add_file(md_path)
    md_docs = list(rag.vector_store.docs)
    check("返回块数 > 0", n_md > 0, f"(n={n_md})")
    check("块数与向量库写入一致", n_md == len(md_docs))
    check("每块含 source 元数据",
          all(d.metadata.get("source") == "课程介绍.md" for d in md_docs))
    check("每块含 kind 元数据",
          all(d.metadata.get("kind") for d in md_docs))
    check("正文块 kind=text",
          any(d.metadata.get("kind") == "text" for d in md_docs))
    check("块内容非空", all(d.page_content.strip() for d in md_docs))

    print("\n[2] add_file(.csv) → 表格结构化块")
    before = len(rag.vector_store.docs)
    n_csv = rag.add_file(csv_path)
    csv_docs = rag.vector_store.docs[before:]
    check("返回块数 > 0", n_csv > 0, f"(n={n_csv})")
    check("存在 kind=table 的块",
          any(d.metadata.get("kind") == "table" for d in csv_docs))
    table_doc = next((d for d in csv_docs if d.metadata.get("kind") == "table"), None)
    check("表格块保留表头", table_doc is not None and "班型" in table_doc.page_content)

    # ===== 解析器缺失：回退 TextLoader =====
    print("\n[3] doc_parser=None → 回退 TextLoader 分支")
    rag_fallback = _build_rag(None)
    n_txt = rag_fallback.add_file(md_path)
    fb_docs = list(rag_fallback.vector_store.docs)
    check("回退分支仍能入库", n_txt > 0, f"(n={n_txt})")
    check("回退块补默认 kind=text",
          all(d.metadata.get("kind") == "text" for d in fb_docs))
    check("回退块含 source",
          all(d.metadata.get("source") == "课程介绍.md" for d in fb_docs))

    print("\n" + "=" * 68)
    print(f"结果：通过 {passed} / 失败 {failed}")
    print("=" * 68)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
