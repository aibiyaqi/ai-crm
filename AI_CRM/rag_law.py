"""
rag_law.py - RAG 知识库模块
====================================
功能分区：
  1. Chroma 向量数据库初始化（持久化存储，使用BGE或DashScope嵌入）
  2. 文档上传与分块（文件 → 分块 → 嵌入 → 存储）
  3. 混合检索（向量检索 + BM25 关键词检索 + 重排 + 自适应TopK）
  4. 文件上传更新知识库（支持增量添加）

嵌入模型（二选一，通过 USE_LOCAL_EMBEDDING 切换）：
  - 本地模式：BAAI/bge-small-zh-v1.5（~33MB，中文优化，推荐）
  - 在线模式：通义千问 text-embedding-v2 API（DashScope）

改进记录：
  2026-07-02: 新增本地BGE嵌入模型支持 (方案A)
  2026-07-02: BM25分词改用jieba，支持中文 (方案C)
  2026-07-02: 新增自适应TopK + 相关性阈值过滤 (方案D)
"""

# ==================== 1. 导入依赖 ====================
import os                                # 文件路径操作
from typing import List, Optional       # 类型提示
import re                                # 正则表达式

# LangChain Chroma 向量库集成
# [WARN] 必须使用 chromadb==0.4.24，0.5.0 不兼容！
from langchain_chroma import Chroma      # Chroma 向量数据库封装

# 通义千问 text-embedding-v2 嵌入模型（在线模式）
# 在线 API，无需下载模型文件，零成本
# 兼容 LangChain Embeddings 接口
from dashscope import TextEmbedding as DashTextEmbedding

# 文本分块工具（将长文档切分成小块）
# langchain 0.2.x 已将 text_splitter 拆分为独立包
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 文档加载器（从文件读取文本）
from langchain_community.document_loaders import TextLoader

# LangChain Document 对象：新版统一在 langchain_core，旧版在 langchain.schema
# （langchain 1.x 已移除 langchain.schema，这里做兼容导入避免入库时报 ModuleNotFoundError）
try:
    from langchain_core.documents import Document
except Exception:  # pragma: no cover
    from langchain.schema import Document  # type: ignore

# BM25 关键词检索（弥补向量检索对关键词不敏感的缺陷）
from rank_bm25 import BM25Okapi

# jieba 中文分词（用于BM25中文关键词检索）
# 安装：pip install jieba
import jieba

from config import (
    CHROMA_PERSIST_DIR,      # Chroma 持久化路径：utils/chroma_kb
    CHROMA_COLLECTION_NAME,  # Chroma 集合名称：crm_kb（知识库）
    RETRIEVAL_TOP_K,         # 检索召回数：4
    CHUNK_SIZE,              # 分块大小：500 字符
    CHUNK_OVERLAP,           # 分块重叠：100 字符
    EMBEDDING_MODEL,         # 嵌入模型名称：text-embedding-v2
    USE_LOCAL_EMBEDDING,     # 是否使用本地BGE嵌入模型
    LOCAL_EMBEDDING_MODEL,   # 本地嵌入模型名称
    RERANKER_MODEL,          # 重排序模型名称
    RERANKER_ENABLED,        # 是否启用重排序
    RERANKER_TOP_K,          # 重排序后返回的结果数
    RERANKER_DEVICE,         # 运行设备 cpu/cuda
    RERANKER_BATCH_SIZE,     # 批处理大小
    RETRIEVAL_SCORE_THRESHOLD,  # 检索相关性分数阈值
    ADAPTIVE_TOP_K,          # 自适应TopK启用标志
    ADAPTIVE_TOP_K_MAX,      # 自适应最大返回数
    ADAPTIVE_TOP_K_MIN,      # 自适应最小返回数
    ADAPTIVE_TOP_K_SCORE,    # 自适应动态TopK的分数阈值
)
from logger import logger  # 日志记录器

# 文档解析增强（可插拔后端：txt/md/pdf/docx/csv/图片 + 表格抽取 + 版面/OCR）
# 缺失该模块时降级为 None，add_file 会自动回退到 TextLoader（不影响原有能力）
try:
    from doc_parser import DocumentParser as _DocumentParser
except Exception:  # pragma: no cover
    _DocumentParser = None


# ==================== 2. 通义千问嵌入模型（LangChain 兼容封装） ====================
def _resolve_local_model_path(model_name: str) -> str:
    """
    把 HuggingFace 仓库名解析为**本地缓存快照目录**（离线加载，避免联网 HEAD 超时）
    ==============================================================================
    背景：sentence-transformers / transformers 即使模型已缓存，仍会对
          huggingface.co 发起 HEAD 校验（如 adapter_config.json），
          在无外网环境下会连续超时重试（WinError 10060），阻塞服务数十秒。
    做法：优先在 HF 缓存目录里找到该仓库的 snapshots/<hash> 目录，
          直接把**本地绝对路径**交给 SentenceTransformer，彻底绕开网络。
    返回：本地快照路径（找到）/ 原 model_name（未找到，回退在线加载）。
    """
    try:
        # 仓库名 → 缓存目录名：BAAI/bge-small-zh-v1.5 → models--BAAI--bge-small-zh-v1.5
        repo_dir = "models--" + model_name.replace("/", "--")
        cache_root = os.getenv("HF_HOME") or os.path.join(
            os.path.expanduser("~"), ".cache", "huggingface", "hub"
        )
        snapshots = os.path.join(cache_root, repo_dir, "snapshots")
        if os.path.isdir(snapshots):
            # 取最新（唯一）快照目录
            candidates = [
                os.path.join(snapshots, d)
                for d in os.listdir(snapshots)
                if os.path.isdir(os.path.join(snapshots, d))
            ]
            if candidates:
                local_path = max(candidates, key=os.path.getmtime)
                logger.info(f"[LOCAL] 命中本地模型缓存: {model_name} -> {local_path}")
                return local_path
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[WARN] 解析本地模型缓存失败（回退在线加载）: {e}")
    return model_name


# ==================== 2a. BGE本地嵌入模型（推荐，中文优化） ====================
class BGEEmbeddings:
    """
    BAAI/bge-small-zh-v1.5 本地嵌入模型
    ======================================
    功能：替代 DashScope 在线 API，使用本地 BGE 模型生成向量。
    
    优势：
      - 完全本地运行，无外部依赖，零网络延迟
      - 专为中文优化，语义理解优于通用英文模型
      - 仅 33MB，加载快，适合生产部署
      - 支持 query/passage 双模式，检索精度更高
    
    使用方式：
      embeddings = BGEEmbeddings()
      vectors = embeddings.embed_documents(["文本1", "文本2"])
      vector = embeddings.embed_query("查询文本")
    
    模型来源：
      https://huggingface.co/BAAI/bge-small-zh-v1.5
    """

    def __init__(
        self,
        model_name: str = None,
        batch_size: int = 32,
        device: str = "cpu",
    ):
        """
        初始化本地 BGE 嵌入模型

        参数：
          model_name : 模型名称，默认 BAAI/bge-small-zh-v1.5
          batch_size : 批量编码的最大文本数
          device     : 运行设备 cpu / cuda
        """
        self.model_name = model_name or LOCAL_EMBEDDING_MODEL
        self.batch_size = batch_size
        self.device = device
        self._model = None  # 懒加载
        logger.info(f"[REFRESH] BGE嵌入模型已注册: {self.model_name}（首次使用时加载，约33MB）")

    def _load_model(self):
        """懒加载 sentence-transformers 模型"""
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
            # 优先使用本地缓存快照路径，避免联网 HEAD 校验超时（无外网环境）
            local_path = _resolve_local_model_path(self.model_name)
            logger.info(f"[REFRESH] 正在加载BGE嵌入模型: {local_path}...")
            self._model = SentenceTransformer(
                local_path,
                device=self.device,
            )
            # BGE 官方推荐：为query添加指令前缀以提升检索精度
            self.query_prefix = "为这个句子生成表示以用于检索相关文章："
            logger.info(f"[OK] BGE嵌入模型加载完成: {self.model_name}")
        except ImportError:
            logger.error(
                "[FAIL] sentence-transformers 未安装，无法使用BGE嵌入模型。"
                "请运行: pip install sentence-transformers"
            )
            raise
        except Exception as e:
            logger.error(f"[FAIL] BGE嵌入模型加载失败: {e}")
            raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        批量编码文档文本为向量
        文档模式：不使用指令前缀
        """
        if not texts:
            return []
        self._load_model()
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            try:
                embeddings = self._model.encode(
                    batch,
                    normalize_embeddings=True,  # 归一化，便于余弦相似度计算
                    show_progress_bar=False,
                )
                all_embeddings.extend(embeddings.tolist())
            except Exception as e:
                logger.error(f"[FAIL] BGE批量编码失败: {e}")
                all_embeddings.extend([[0.0] * 768 for _ in batch])
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """
        编码单条查询文本为向量
        查询模式：使用BGE官方推荐的指令前缀
        """
        self._load_model()
        try:
            embedding = self._model.encode(
                self.query_prefix + text,
                normalize_embeddings=True,
            )
            return embedding.tolist()
        except Exception as e:
            logger.error(f"[FAIL] BGE查询编码失败: {e}")
            return [0.0] * 768


# ==================== 2b. 通义千问嵌入模型（在线模式，备用） ====================
class DashScopeEmbeddings:
    """
    通义千问 text-embedding-v2 嵌入模型封装
    =========================================
    兼容 LangChain Embeddings 接口，作为 HuggingFaceEmbeddings 的替代方案。
    
    优势：
      - 在线 API，零下载（替代 HuggingFace 2GB 模型文件）
      - 国内用户无需 VPN，通过阿里云 DashScope 直接调用
      - 与项目已有的通义千问 LLM（dashscope SDK）共用 API Key
    
    使用方式：
      embeddings = DashScopeEmbeddings()
      vectors = embeddings.embed_documents(["文本1", "文本2"])
      vector = embeddings.embed_query("查询文本")
    
    API 文档：
      https://help.aliyun.com/zh/model-studio/getting-started/models
    """

    def __init__(
        self,
        model: str = None,
        batch_size: int = 25,
    ):
        """
        初始化嵌入模型

        参数：
          model      : 通义千问嵌入模型名称，默认 text-embedding-v2
          batch_size : 批量编码的最大文本数（API 限制单次最多 25 条）
        """
        self.model = model or EMBEDDING_MODEL
        self.batch_size = batch_size

        # 确保 dashscope API Key 已设置
        import dashscope
        _key = os.getenv("QWEN_API_KEY")
        if _key:
            dashscope.api_key = _key

        logger.info(f"[REFRESH] 嵌入模型已注册: {self.model}（DashScope 在线 API，零下载）")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        批量编码文档文本为向量

        参数：
          texts : 文本列表
        返回：
          向量列表，每个向量为 float 列表
        """
        if not texts:
            return []

        all_embeddings = []

        # DashScope API 单次最多支持 25 条文本，分批处理
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            try:
                response = DashTextEmbedding.call(
                    model=self.model,
                    input=batch,
                )
                if response.status_code == 200:
                    # 按输入顺序提取嵌入向量
                    batch_embeddings = [
                        item["embedding"]
                        for item in sorted(
                            response.output["embeddings"],
                            key=lambda x: x["text_index"],
                        )
                    ]
                    all_embeddings.extend(batch_embeddings)
                else:
                    logger.error(
                        f"[FAIL] DashScope 嵌入 API 返回错误 (HTTP {response.status_code}): "
                        f"{response.message}"
                    )
                    # 返回零向量作为兜底
                    all_embeddings.extend(
                        [[0.0] * 1536 for _ in batch]
                    )
            except Exception as e:
                logger.error(f"[FAIL] DashScope 嵌入 API 调用失败: {e}")
                all_embeddings.extend(
                    [[0.0] * 1536 for _ in batch]
                )

        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """
        编码单条查询文本为向量

        参数：
          text : 查询文本
        返回：
          向量（float 列表）
        """
        result = self.embed_documents([text])
        return result[0] if result else [0.0] * 1536


# ==================== 3. Cross-Encoder 重排序器 ====================
class CrossEncoderReranker:
    """
    Cross-Encoder 重排序器
    ========================
    功能：对初检结果进行精排，显著提升 RAG 准确率
    
    技术原理：
      - Bi-Encoder（向量检索）将 query 和 doc 分别编码为向量，计算余弦相似度
        优点是速度快、可预先计算，缺点是丢失了 query-doc 间的交叉注意力
      - Cross-Encoder 将 query 和 doc 拼接后输入同一个 Transformer，
        通过自注意力机制捕捉两者的深层交互，打分更精确
        缺点是速度较慢，无法预先计算
    
    使用方式：
      reranker = CrossEncoderReranker()
      scores = reranker.rerank("用户问题", ["文档1", "文档2", ...])
      # 返回 [(文档, 分数), ...] 按分数降序排列
    
    模型选择：
      - cross-encoder/ms-marco-MiniLM-L-6-v2  (~80MB, 轻量快速, 推荐)
      - BAAI/bge-reranker-v2-m3                (~1.2GB, 多语言强)

    注意：
      重排序模型仍从 HuggingFace 下载（约 80MB）。
      如无 VPN，请在环境变量中设置：
        export HF_ENDPOINT=https://hf-mirror.com
      或在使用前执行：
        import os; os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    """

    def __init__(
        self,
        model_name: str = None,
        device: str = None,
        batch_size: int = None,
    ):
        """
        初始化重排序器（懒加载，首次 rerank 时才下载模型）
         
        参数：
          model_name : Cross-Encoder 模型名称
          device     : 运行设备 cpu / cuda
          batch_size : 批处理大小
        """
        self.model_name = model_name or RERANKER_MODEL
        self.device = device or RERANKER_DEVICE
        self.batch_size = batch_size or RERANKER_BATCH_SIZE
        self._model = None  # 懒加载
        logger.info(f"[REFRESH] 重排序器已注册: {self.model_name}（首次使用时加载）")

    def _load_model(self):
        """
        懒加载 Cross-Encoder 模型
        ===========================
        仅在第一次调用 rerank() 时加载，避免启动时不必要的资源消耗
        """
        if self._model is not None:
            return

        try:
            from sentence_transformers import CrossEncoder

            # 优先使用本地缓存快照路径，避免联网 HEAD 校验超时（无外网环境）
            local_path = _resolve_local_model_path(self.model_name)
            logger.info(f"[REFRESH] 正在加载重排序模型: {local_path}...")
            self._model = CrossEncoder(
                local_path,
                device=self.device,
                max_length=512,  # 输入最大长度，超长则截断
            )
            logger.info(f"[OK] 重排序模型加载完成: {self.model_name}")
        except ImportError:
            logger.error(
                "[FAIL] sentence-transformers 未安装，无法使用 Cross-Encoder 重排序。"
                "请运行: pip install sentence-transformers==3.0.1"
            )
            raise
        except Exception as e:
            logger.error(f"[FAIL] 重排序模型加载失败: {e}")
            raise

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = None,
    ) -> List[str]:
        """
        对文档进行重排序
        ==================
        参数：
          query     : 用户查询
          documents : 待重排的文档列表
          top_k     : 返回的最相关文档数（默认全部返回按分数降序）
        返回：
          按相关性降序排列的文档列表（仅返回文本内容）
        """
        if not documents:
            return []

        if top_k is None:
            top_k = len(documents)

        # 懒加载模型
        self._load_model()

        # 构造 (query, doc) 对
        pairs = [[query, doc] for doc in documents]

        try:
            # Cross-Encoder 批量评分
            # 返回每个 pair 的 relevance score（分数越高越相关）
            scores = self._model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
            )

            # 将文档与分数配对，按分数降序排序
            scored_docs = list(zip(documents, scores))
            scored_docs.sort(key=lambda x: x[1], reverse=True)

            logger.info(
                f"[SEARCH] 重排序完成: {len(documents)}篇 → "
                f"top得分={scored_docs[0][1]:.4f}" if scored_docs else ""
            )

            # 返回排序后的文档列表
            return [doc for doc, _ in scored_docs[:top_k]]

        except Exception as e:
            logger.warning(f"[WARN] 重排序失败（回退到原始排序）: {e}")
            # 重排序失败时回退到原始顺序
            return documents[:top_k]


# ==================== 4. Chroma 向量数据库 ====================
class RAGLaw:
    """
    RAG 知识库管理类
    =================
    技术架构：
      向量检索（Chroma）   → 语义相似度匹配
      + BM25关键词检索     → 精确关键词匹配（如课程/班型名称，基于jieba中文分词）
      + 重排              → 双通道结果合并去重排序
      + 自适应TopK        → 基于相关性分数动态调整返回数量
    
    嵌入模型（二选一，通过 USE_LOCAL_EMBEDDING 配置切换）：
      - 本地模式（推荐）：BAAI/bge-small-zh-v1.5（~33MB，中文优化）
      - 在线模式（备用）：通义千问 text-embedding-v2（DashScope API）
    
    使用方式：
      rag = RAGLaw()
      docs = rag.hybrid_search("初三数学一对一课时费")  # 混合检索
      docs = rag.adaptive_search("智能客服有哪些功能")  # 自适应检索（推荐）
      rag.add_file("合同范本.txt")                  # 上传文档
    """
    
    def __init__(self):
        """初始化向量数据库、嵌入模型和检索配置"""
        # -------- 2.1 初始化嵌入模型（BGE本地模式 / DashScope在线模式）--------
        if USE_LOCAL_EMBEDDING:
            # 方案A: 使用本地 BGE 嵌入模型（推荐，中文优化）
            # BAAI/bge-small-zh-v1.5 仅 33MB，完全本地运行
            # 向量维度：768 维
            logger.info(
                f"[REFRESH] 正在初始化本地嵌入模型: {LOCAL_EMBEDDING_MODEL}（BGE，~33MB）..."
            )
            self.embeddings = BGEEmbeddings(
                model_name=LOCAL_EMBEDDING_MODEL,
                batch_size=32,
                device="cpu",
            )
            logger.info(f"[OK] 本地嵌入模型就绪: {LOCAL_EMBEDDING_MODEL}")
        else:
            # 备用: 使用 DashScope 在线 API
            # 向量维度：1536 维
            # 优势：零下载、零延迟启动、国内无 VPN 可用
            logger.info(
                f"[REFRESH] 正在初始化嵌入模型: {EMBEDDING_MODEL}（DashScope 在线 API）..."
            )
            self.embeddings = DashScopeEmbeddings(
                model=EMBEDDING_MODEL,
                batch_size=25,
            )
            logger.info(f"[OK] 嵌入模型就绪: {EMBEDDING_MODEL}")
        
        # -------- 2.2 初始化文本分块器 --------
        # RecursiveCharacterTextSplitter：递归分割文本
        # 先按段落分割 → 再按句子分割 → 最后按字符分割
        # 保证语义完整性
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,          # 每块最大字符数：500
            chunk_overlap=CHUNK_OVERLAP,    # 块间重叠字符数：100
            length_function=len,            # 长度计算函数
            separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""],  # 分割优先级
        )

        # -------- 2.2b 文档解析器（可插拔增强：表格/版面/OCR，缺失时降级为 None）--------
        # 相比只吃纯文本的 TextLoader：支持 PDF/DOCX/CSV/图片，表格抽成结构化块，
        # 分块保留 page / section / kind 元数据，便于检索与溯源。
        self.doc_parser = _DocumentParser() if _DocumentParser is not None else None
        if self.doc_parser is not None:
            logger.info(
                f"[OK] 文档解析器已启用，支持扩展名: {self.doc_parser.supported_extensions()}"
            )
        else:
            logger.warning("[WARN] 未找到 doc_parser，文档入库将仅支持纯文本（TextLoader）")
        
        # -------- 2.3 初始化 Chroma 向量数据库 --------
        # Chroma 是一个嵌入向量数据库，用于存储和检索文档向量
        # persist_directory：持久化目录，程序重启后数据不丢失
        # 幂等迁移：老库集合名 crm_law → crm_kb（保留向量，不重建）
        self._migrate_collection_name()
        self.vector_store = Chroma(
            persist_directory=CHROMA_PERSIST_DIR,   # 持久化路径
            embedding_function=self.embeddings,      # 嵌入函数
            collection_name=CHROMA_COLLECTION_NAME,  # 集合名称：crm_kb
        )
        logger.info(
            f"[OK] Chroma 向量数据库已加载: {CHROMA_PERSIST_DIR} "
            f"(collection={CHROMA_COLLECTION_NAME})"
        )
        # 维度自愈：若历史集合的向量维度与当前嵌入模型不一致（如换过嵌入模型），
        # 查询时会抛 "Collection expecting embedding with dimension of X, got Y"。
        # 这里检测并重建集合，避免检索链路整体失败。
        self._ensure_collection_dimension()
        
        # -------- 2.4 BM25 检索器（方案C：jieba中文分词）--------
        # BM25 用于关键词精确匹配，与向量检索互补
        # BM25 分词引擎使用 jieba（中文分词），替代原先的简单正则切分
        self.bm25_index = None       # BM25 索引（懒加载）
        self.bm25_docs = []          # BM25 文档列表
        self._rebuild_bm25()         # 从现有数据重建 BM25 索引
        
        # -------- 2.5 Cross-Encoder 重排序器（可选） --------
        # 对初检结果进行精排，提升 RAG 准确率
        # 使用懒加载，首次检索时才下载模型
        # [WARN] 重排序模型仍需从 HuggingFace 下载约 80MB
        #    如无 VPN，请设置环境变量：HF_ENDPOINT=https://hf-mirror.com
        self.reranker = None
        if RERANKER_ENABLED:
            try:
                self.reranker = CrossEncoderReranker(
                    model_name=RERANKER_MODEL,
                    device=RERANKER_DEVICE,
                    batch_size=RERANKER_BATCH_SIZE,
                )
                logger.info("[OK] Cross-Encoder 重排序器已就绪（懒加载）")
            except Exception as e:
                logger.warning(f"[WARN] 重排序器初始化失败（不影响基础检索）: {e}")
                self.reranker = None
    
    def warmup(self) -> None:
        """
        预热嵌入 / 重排模型（把冷启动成本前移到服务启动阶段）
        ====================================================
        背景：BGE 嵌入模型与 Cross-Encoder 重排器均为**懒加载**，
              首个 RAG 请求才触发加载（数十秒），在单进程服务下会阻塞
              所有并发请求，导致压测/首屏体验极差。
        做法：启动时主动 encode 一次空/占位文本，强制模型加载进内存。
        安全：任何异常都不抛出（预热是增强，不是主链路依赖）。
        """
        # 1) 预热嵌入模型（BGE / DashScope）
        try:
            self.embeddings.embed_query("预热")
            logger.info("[OK] 嵌入模型预热完成")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[WARN] 嵌入模型预热失败: {e}")

        # 2) 预热重排序模型（可选）
        try:
            if self.reranker is not None:
                self.reranker.rerank("预热", ["预热文档"])
                logger.info("[OK] 重排序模型预热完成")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[WARN] 重排序模型预热失败: {e}")

    def _ensure_collection_dimension(self) -> None:
        """
        校验 Chroma 集合向量维度与当前嵌入模型是否一致，不一致则重建集合
        =================================================================
        背景：更换嵌入模型（如 384 维 → 512 维）后，历史集合仍保存旧维度向量，
        查询/写入会抛维度不匹配异常，导致 RAG 检索整体不可用。
        策略：检测到不一致时，删除旧集合并以当前嵌入模型重建（历史向量无法跨维度复用）。
        """
        try:
            import chromadb
            client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
            try:
                col = client.get_collection(CHROMA_COLLECTION_NAME)
            except Exception:
                return  # 集合尚不存在，首次写入时自动创建
            count = col.count()
            if count == 0:
                return
            sample = col.get(limit=1, include=["embeddings"])
            embs = sample.get("embeddings")
            if embs is None or len(embs) == 0:
                return
            stored_dim = len(embs[0])
            # 探测当前嵌入模型维度
            probe = self.embeddings.embed_query("维度探测")
            current_dim = len(probe)
            if stored_dim == current_dim:
                logger.info(f"[OK] Chroma 集合维度校验通过: {stored_dim} 维")
                return
            logger.warning(
                f"[WARN] Chroma 集合维度不匹配: 存储={stored_dim} 维, "
                f"当前嵌入模型={current_dim} 维（{count} 条历史向量）→ 重建集合"
            )
            client.delete_collection(CHROMA_COLLECTION_NAME)
            # 用当前嵌入函数重建空集合
            self.vector_store = Chroma(
                persist_directory=CHROMA_PERSIST_DIR,
                embedding_function=self.embeddings,
                collection_name=CHROMA_COLLECTION_NAME,
            )
            logger.info(
                f"[OK] Chroma 集合已按 {current_dim} 维重建: {CHROMA_COLLECTION_NAME}"
            )
        except Exception as e:
            logger.warning(f"[WARN] Chroma 集合维度自愈失败（不影响启动）: {e}")

    @staticmethod
    def _migrate_collection_name():
        """
        Chroma 集合名幂等迁移（crm_law → crm_kb）
        ==========================================
        背景：知识库口径由「法规」改为「K12 课程产品资料」，集合名同步更名。
        做法：仅重命名集合元数据，不重建向量（向量与集合名无关）。
        安全：目标已存在 / 老集合不存在 → 直接返回；异常不影响主流程。
        """
        try:
            import chromadb
            client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
            names = [
                c.name if hasattr(c, "name") else c
                for c in client.list_collections()
            ]
            if CHROMA_COLLECTION_NAME not in names and "crm_law" in names:
                client.get_collection("crm_law").modify(name=CHROMA_COLLECTION_NAME)
                logger.info(f"[MIGRATE] Chroma 集合已重命名: crm_law -> {CHROMA_COLLECTION_NAME}")
        except Exception as e:
            logger.warning(f"[WARN] Chroma 集合名迁移跳过: {e}")

    # ==================== 3. 文档上传 ====================
    def add_file(self, file_path: str) -> int:
        """
        上传文件到知识库（解析增强版）
        =================================
        解析：优先走可插拔 doc_parser（txt/md/pdf/docx/csv/图片，支持表格抽取、
              版面分页、可选 OCR）；不可用或不支持时回退 TextLoader（纯文本）。
        分块：表格整块保留、正文按长度递归切，块元数据带 kind/page/section/source。
        参数：
          file_path : 文件路径
        返回：
          添加的文档块数量
        """
        # 检查文件是否存在
        if not os.path.exists(file_path):
            logger.error(f"[FAIL] 文件不存在: {file_path}")
            return 0

        chunks: List[Document] = []

        # -------- 优先：可插拔文档解析器（表格 / 版面 / OCR）--------
        if self.doc_parser is not None:
            try:
                parsed, parsed_chunks = self.doc_parser.parse_and_chunk(
                    file_path,
                    chunk_size=CHUNK_SIZE,
                    overlap=CHUNK_OVERLAP,
                )
                if parsed.warnings:
                    logger.warning(
                        f"[WARN] 解析 {os.path.basename(file_path)}: "
                        f"{'; '.join(parsed.warnings)}"
                    )
                chunks = [
                    Document(page_content=c.text, metadata=dict(c.metadata))
                    for c in parsed_chunks
                    if c.text and c.text.strip()
                ]
                logger.info(f"[DOC] 解析完成: {parsed.stats()}")
            except Exception as e:  # 解析失败 → 回退，不阻断入库
                logger.warning(f"[WARN] doc_parser 解析失败，回退 TextLoader: {e}")
                chunks = []

        # -------- 回退：纯文本读取（保持旧行为可用）--------
        if not chunks:
            loader = TextLoader(
                file_path,
                encoding="utf-8",       # 指定 UTF-8 编码，支持中文
            )
            documents = loader.load()  # 读取文件内容

            # 为每个文档添加元数据（来源 / 类型）
            for doc in documents:
                doc.metadata["source"] = os.path.basename(file_path)  # 文件名
                doc.metadata["file_path"] = file_path                 # 完整路径
                doc.metadata.setdefault("kind", "text")
                doc.metadata.setdefault("page", 0)
                doc.metadata.setdefault("section", "")

            # 文本分块：将长文档切分成小段
            chunks = self.text_splitter.split_documents(documents)

        # 将文档块添加到 Chroma 向量数据库
        # 自动计算嵌入向量并存储（Chroma 0.4.x 自动持久化）
        self.vector_store.add_documents(chunks)

        # 重建 BM25 索引（增量更新）
        self._rebuild_bm25()

        logger.info(f"[DOC] 已添加文件: {file_path} → {len(chunks)} 个文档块")
        return len(chunks)
    
    def add_text(self, text: str, metadata: dict = None) -> int:
        """
        直接添加文本到知识库
        ======================
        参数：
          text     : 文本内容
          metadata : 元数据字典（可选）
        返回：
          添加的文档块数量
        """
        if metadata is None:
            metadata = {}
        
        # 将文本封装为 LangChain Document 对象
        doc = Document(page_content=text, metadata=metadata)
        
        # 分块
        chunks = self.text_splitter.split_documents([doc])
        
        # 添加到 Chroma（自动持久化）
        self.vector_store.add_documents(chunks)
        
        # 重建 BM25
        self._rebuild_bm25()
        
        logger.info(f"[MEMO] 已添加文本 → {len(chunks)} 个文档块")
        return len(chunks)
    
    # ==================== 4. 混合检索（核心功能） ====================
    def hybrid_search(
        self,
        query: str,
        top_k: int = None
    ) -> List[str]:
        """
        混合检索：向量检索 + BM25 关键词检索 + Cross-Encoder 重排序
        =============================================================
        参数：
          query : 用户查询文本
          top_k : 返回结果数量（默认 RETRIEVAL_TOP_K=4）
        返回：
          文档内容字符串列表（按相关性降序）
         
        检索流程（四阶段）：
           用户查询
             ├─ 阶段1: 向量检索（带相关性分数）→ 语义匹配结果
             ├─ 阶段2: BM25检索（jieba中文分词）→ 关键词匹配结果
             └─ 阶段3:
                 ├─ 合并去重 → 初检候选集
                 ├─ [方案D] 相关性阈值过滤 → 剔除低分结果
                 ├─ [可选] Cross-Encoder 重排序 → 精排
                 └─ 返回 TopK
        """
        if top_k is None:
            top_k = RETRIEVAL_TOP_K
        
        # -------- 4.1 向量检索（语义相似度，带分数） --------
        # [方案D] 使用 similarity_search_with_relevance_scores 获取相关性分数
        # 用于后续的阈值过滤
        vector_results_with_scores = self.vector_store.similarity_search_with_relevance_scores(
            query,
            k=top_k * 4  # 多召回一些，给阈值过滤和重排序留出筛选空间
        )
        # 方案D: 对向量检索结果应用相关性阈值过滤
        vector_texts = []
        for doc, score in vector_results_with_scores:
            if score >= RETRIEVAL_SCORE_THRESHOLD:
                vector_texts.append(doc.page_content)
            else:
                logger.debug(f"⏬ 向量检索结果低于阈值({score:.4f} < {RETRIEVAL_SCORE_THRESHOLD})，已过滤")
        
        # 如果阈值过滤后为空，放宽阈值兜底（取最高分的结果）
        if not vector_texts and vector_results_with_scores:
            best_doc, best_score = vector_results_with_scores[0]
            vector_texts = [best_doc.page_content]
            logger.info(f"[WARN] 所有向量结果低于阈值，兜底保留最高分({best_score:.4f})")
        
        # -------- 4.2 BM25 关键词检索（方案C：jieba中文分词）--------
        bm25_texts = []
        if self.bm25_index is not None and self.bm25_docs:
            # [方案C] 使用 jieba 中文分词
            tokenized_query = self._tokenize(query)
            # BM25 打分排序
            bm25_scores = self.bm25_index.get_scores(tokenized_query)
            # 取分数最高的 top_k 个文档
            top_indices = sorted(
                range(len(bm25_scores)),
                key=lambda i: bm25_scores[i],
                reverse=True
            )[:top_k * 3]
            bm25_texts = [self.bm25_docs[i] for i in top_indices if bm25_scores[i] > 0]
        
        # -------- 4.3 合并去重（初检候选集） --------
        all_texts = []
        seen = set()
        for text in vector_texts + bm25_texts:
            if text not in seen:
                all_texts.append(text)
                seen.add(text)
        
        # -------- 4.4 Cross-Encoder 重排序（精排） --------
        if self.reranker is not None and len(all_texts) > 1:
            # 重排序会对所有初检结果进行精确打分并重新排序
            rerank_top_k = RERANKER_TOP_K if RERANKER_TOP_K > 0 else top_k
            result = self.reranker.rerank(
                query=query,
                documents=all_texts,
                top_k=rerank_top_k,
            )
            logger.info(
                f"[SEARCH] 混合检索+重排序完成: 初检{len(all_texts)}篇 → "
                f"重排后{len(result)}条"
            )
        else:
            # 重排序器未就绪时，回退到 TopK（已通过阈值过滤）
            result = all_texts[:top_k]
            logger.info(
                f"[SEARCH] 混合检索完成（无重排序）: '{query[:30]}...' → "
                f"{len(result)}条结果（阈值={RETRIEVAL_SCORE_THRESHOLD}）"
            )
        
        return result
    
    # ==================== 4b. 自适应检索（方案D） ====================
    def adaptive_search(
        self,
        query: str,
        min_results: int = None,
        max_results: int = None,
    ) -> List[str]:
        """
        自适应检索：基于相关性分数动态调整返回数量（方案D）
        ======================================================
        相比 hybrid_search 的固定 TopK，此方法会：
          1. 先检索较多候选集
          2. 根据相关性分数的分布情况，动态决定返回多少条
          3. 如果所有结果分数都很高 → 多返回几条
          4. 如果分数参差不齐 → 只返回高质量的结果
          5. 确保结果数在 [min_results, max_results] 范围内
        
        参数：
          query       : 用户查询文本
          min_results : 最少返回数（默认 ADAPTIVE_TOP_K_MIN=1）
          max_results : 最多返回数（默认 ADAPTIVE_TOP_K_MAX=6）
        返回：
          文档内容字符串列表
        """
        if min_results is None:
            min_results = ADAPTIVE_TOP_K_MIN
        if max_results is None:
            max_results = ADAPTIVE_TOP_K_MAX
        
        # -------- 1. 向量检索（带分数）--------
        results_with_scores = self.vector_store.similarity_search_with_relevance_scores(
            query,
            k=max_results * 3  # 大候选集
        )
        
        if not results_with_scores:
            logger.warning(f"[WARN] 自适应检索无结果: '{query[:30]}...'")
            return []
        
        # -------- 2. 基于分数动态截断 --------
        # 策略：计算分数下降幅度，在下降最陡的地方截断
        scored_docs = [(doc.page_content, score) for doc, score in results_with_scores]
        
        # 按分数降序排列（API已保证）
        # 找到分数下降最陡的位置
        cutoff = len(scored_docs)
        score_threshold = ADAPTIVE_TOP_K_SCORE
        
        for i in range(1, len(scored_docs)):
            prev_score = scored_docs[i - 1][1]
            curr_score = scored_docs[i][1]
            # 如果分数低于绝对阈值
            if curr_score < score_threshold:
                cutoff = i
                break
            # 如果分数下降超过 50%（相对下降剧烈）
            if prev_score > 0 and (prev_score - curr_score) / prev_score > 0.5:
                cutoff = i
                break
        
        # 确保在 [min_results, max_results] 范围内
        final_k = max(min_results, min(cutoff, max_results))
        
        result = [doc for doc, _ in scored_docs[:final_k]]
        
        logger.info(
            f"[SEARCH] 自适应检索完成: 候选{len(scored_docs)}篇 → "
            f"动态截断{len(result)}条 "
            f"(阈值={score_threshold}, 范围[{min_results},{max_results}])"
        )
        
        return result
    
    # ==================== 5. 纯向量检索 ====================
    def similarity_search(self, query: str, k: int = None) -> List[str]:
        """
        纯向量语义检索
        ================
        参数：
          query : 查询文本
          k     : 返回数量
        返回：
          文档内容字符串列表
        """
        if k is None:
            k = RETRIEVAL_TOP_K
        
        docs = self.vector_store.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]
    
    # ==================== 5b. 带分数的向量检索 ====================
    def similarity_search_with_scores(
        self,
        query: str,
        k: int = None
    ) -> List[tuple]:
        """
        带相关性分数的向量检索
        ========================
        参数：
          query : 查询文本
          k     : 返回数量
        返回：
          [(文档内容, 相关性分数), ...] 按分数降序
        """
        if k is None:
            k = RETRIEVAL_TOP_K
        
        results = self.vector_store.similarity_search_with_relevance_scores(query, k=k)
        return [(doc.page_content, score) for doc, score in results]
    
    # ==================== 6. BM25 索引管理（方案C：jieba分词） ====================
    def _rebuild_bm25(self):
        """
        从 Chroma 中重建 BM25 关键词索引
        ==================================
        每次添加文档后调用，保持 BM25 索引同步更新
        使用 jieba 中文分词（方案C）
        """
        try:
            # 从 Chroma 获取所有文档
            all_docs = self.vector_store.get()
            if all_docs and "documents" in all_docs and all_docs["documents"]:
                self.bm25_docs = all_docs["documents"]
                # [方案C] 对每个文档使用 jieba 进行中文分词
                tokenized_docs = [self._tokenize(doc) for doc in self.bm25_docs]
                self.bm25_index = BM25Okapi(tokenized_docs)
                logger.debug(f"[REFRESH] BM25 索引重建完成: {len(self.bm25_docs)} 篇文档（jieba分词）")
            else:
                self.bm25_index = None
                self.bm25_docs = []
        except Exception as e:
            logger.warning(f"[WARN] BM25 索引重建跳过（当前无文档）: {e}")
            self.bm25_index = None
            self.bm25_docs = []
    
    # ==================== 6b. 中文分词器（方案C：jieba） ====================
    def _tokenize(self, text: str) -> List[str]:
        """
        中文分词（使用 jieba 精确模式）
        ================================
        [方案C] 替代原先的简单正则切分，使用 jieba 进行中文分词。
        
        BM25 需要词级别的 token 列表，jieba 精确模式能：
          - 正确切分中文词汇（如"初三数学一对一" → ["初三", "数学", "一对一"]）
          - 识别专有名词（如公司名、产品名）
          - 过滤停用词（如"的"、"了"等无意义词）
        
        参数：
          text : 输入文本（支持中英文混合）
        返回：
          token 列表
        """
        # 使用 jieba 精确模式进行中文分词
        # jieba.lcut 返回 list，比 cut 更高效
        tokens = jieba.lcut(text)
        
        # 过滤停用词和单字符词（保留有意义的词汇）
        # 停用词列表（常用中文停用词）
        stop_words = {
            "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都",
            "一", "一个", "上", "也", "很", "到", "说", "要", "去", "你",
            "会", "着", "没有", "看", "好", "自己", "这", "他", "她", "它",
            "们", "那", "些", "之", "与", "及", "但", "而", "或", "被",
            "把", "对", "从", "以", "为", "及", "等", "向", "让", "给",
            "并", "又", "还", "可", "已", "将", "能", "做", "用", "让",
            "每", "哪", "谁", "什么", "怎么", "如何", "为何", "什么", "怎样",
            "a", "an", "the", "is", "are", "was", "were", "be", "been",
            "in", "on", "at", "to", "for", "of", "with", "by", "from",
            "this", "that", "it", "its", "we", "our", "you", "your",
            "they", "them", "their", "he", "she", "his", "her",
        }
        
        # 过滤：去除停用词、单字符词、空格
        filtered_tokens = [
            token.lower() for token in tokens
            if token.lower() not in stop_words
            and len(token.strip()) > 1  # 过滤单字符
            and not token.isspace()      # 过滤空白
        ]
        
        return filtered_tokens
    
    # ==================== 7. 获取文档统计 ====================
    def get_stats(self) -> dict:
        """
        获取知识库统计信息
        ====================
        返回：
          {
              "total_docs": 100,           # 总文档数
              "collection": "crm_kb",       # 集合名称
              "persist_dir": "utils/chroma_kb"
          }
        """
        try:
            all_docs = self.vector_store.get()
            doc_count = len(all_docs.get("documents", [])) if all_docs else 0
        except Exception:
            doc_count = 0
        
        return {
            "total_docs": doc_count,
            "collection": CHROMA_COLLECTION_NAME,
            "persist_dir": CHROMA_PERSIST_DIR,
        }


# 兼容别名：语义更准确的类名（知识库），保留 RAGLaw 以兼容既有 import
RAGKnowledgeBase = RAGLaw
