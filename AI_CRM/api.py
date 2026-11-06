"""
api.py - FastAPI 外部 API 接口服务器
======================================
功能分区：
  1. FastAPI 应用初始化 + 全局组件懒加载
  2. GET /api/status — 系统健康检查
  3. POST /api/chat — AI 对话接口
  4. POST /api/rag/upload — 上传文档到知识库
  5. POST /api/rag/text — 添加文本到知识库
  5.6 GET/POST /api/follow_ups — 跟进记录（B.4 跟进闭环）
  6. 启动入口

启动方式：
  uvicorn api:app --host 0.0.0.0 --port 8000 --reload
  python api.py
"""

# ==================== 1. 导入依赖 ====================
import os
import sys
import json
import logging
import time
from typing import Optional, List
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi.responses import PlainTextResponse

# 启动时校验关键环境变量
from config import validate_env_or_exit, DB_ENGINE, REDIS_URL
validate_env_or_exit()

# 运行时指标（可观测性增强：模型效果指标 + 监控大盘数据源）
from metrics import snapshot, render_prometheus, incr, observe

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("api")

# ==================== 1b. 速率限制器 ====================
# 压测/本地联调时可通过环境变量放宽或关闭限流：
#   RATE_LIMIT_ENABLED=0        → 完全关闭限流（仅压测用）
#   RATE_LIMIT_CHAT=100000/minute → 自定义 /api/chat 限流阈值
_RATE_LIMIT_ENABLED = os.getenv("RATE_LIMIT_ENABLED", "1").strip().lower() not in ("0", "false", "no", "off")
_RATE_LIMIT_CHAT = os.getenv("RATE_LIMIT_CHAT", "30/minute").strip()
_RATE_LIMIT_STATUS = os.getenv("RATE_LIMIT_STATUS", "30/minute").strip()


def _limit(rule: str):
    """按环境变量决定是否启用限流；关闭时返回 no-op 装饰器。"""
    if not _RATE_LIMIT_ENABLED:
        def _noop(func):
            return func
        return _noop
    return limiter.limit(rule)


limiter = None  # 具体实例在 _build_limiter() 中构造（Redis 优先，内存兜底）
if not _RATE_LIMIT_ENABLED:
    logger.warning("[WARN] 速率限制已通过 RATE_LIMIT_ENABLED=0 关闭（仅建议压测时使用）")


def _build_limiter():
    """
    限流存储选择：Redis 优先（多 worker 共享阈值），不可用则退回进程内存。
    ==========================================================================
    说明：slowapi/limits 在存储不可用时会抛异常 → 直接 500，因此这里先探活，
          只有 Redis 真的可用才挂 Redis 存储；否则保留内存存储（阈值会按进程放大，
          属于可接受的降级，日志会明确提示）。
    """
    if not _RATE_LIMIT_ENABLED:
        return Limiter(key_func=get_remote_address)
    use_redis = os.getenv("RATE_LIMIT_STORAGE", "redis").strip().lower() != "memory"
    if use_redis:
        try:
            from redis_client import get_redis
            if get_redis().available:
                logger.info(f"[OK] 限流存储 = Redis（{REDIS_URL}），多 worker 阈值共享")
                return Limiter(key_func=get_remote_address, storage_uri=REDIS_URL)
        except Exception as exc:
            logger.warning(f"[WARN] Redis 限流存储初始化失败：{exc}")
        logger.warning("[WARN] Redis 不可用，限流退回进程内存（多 worker 下实际阈值会放大）")
    return Limiter(key_func=get_remote_address)


limiter = _build_limiter()


def _redis_health() -> dict:
    """Redis 健康快照（供 /api/status；失败不影响主流程）"""
    try:
        from redis_client import get_redis
        return get_redis().health()
    except Exception as exc:  # pragma: no cover
        return {"enabled": False, "ok": False, "backend": "error", "error": str(exc)[:200]}


# ==================== 1c. JWT 认证方案 ====================
# 使用 HTTP Bearer Token 方案（兼容 Streamlit 前端的 session_token）
security = HTTPBearer(auto_error=False)


# ==================== 2. 请求/响应模型 ====================
class ChatRequest(BaseModel):
    """聊天请求"""
    query: str                                    # 用户问题
    user_id: str = "api_user"                     # 用户 ID（可选）
    customer_id: Optional[str] = None             # 客户画像 ID（可选）


class ChatResponse(BaseModel):
    """聊天响应"""
    success: bool                                 # 是否成功
    response: str = ""                            # AI 回复
    error: str = ""                               # 错误信息（失败时）


class RagTextRequest(BaseModel):
    """RAG 添加文本请求"""
    text: str                                     # 文本内容
    metadata: Optional[dict] = None               # 元数据（可选）


class RagResponse(BaseModel):
    """RAG 操作响应"""
    success: bool                                 # 是否成功
    chunk_count: int = 0                          # 分块数
    error: str = ""                               # 错误信息（失败时）


class StatusResponse(BaseModel):
    """系统状态响应"""
    status: str                                   # 整体状态
    llm: bool                                     # LLM 就绪
    database: bool                                # 数据库就绪
    rag: bool                                     # 知识库就绪
    search: bool                                  # 搜索就绪
    workflow: bool                                # 工作流就绪
    version: str = "1.0.0"                        # API 版本
    db_engine: str = ""                           # 当前数据库引擎（sqlite / mysql）
    redis: Optional[dict] = None                  # Redis 健康状态（enabled/ok/backend/延迟）


# ==================== 1d. 认证依赖函数 ====================
async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    authorization: Optional[str] = Header(None),
) -> str:
    """
    Bearer Token 认证依赖
    =======================
    从请求头中提取 Token，验证有效性，返回 account_id。
    
    验证流程：
      1. 从 Authorization: Bearer <token> 头中提取 Token
      2. 在数据库中查找该 Token
      3. 校验 Token 是否过期（7 天有效期）
      4. 返回关联的 account_id
    
    豁免路径（无需认证）：
      - /docs, /openapi.json (Swagger 文档)
      - /api/status (健康检查)
    """
    # 提取 Token
    token = None
    if credentials:
        token = credentials.credentials
    elif authorization and authorization.startswith("Bearer "):
        token = authorization[7:]

    if not token:
        raise HTTPException(
            status_code=401,
            detail="缺少认证令牌。请在请求头中添加 Authorization: Bearer <token>",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # ① 先查 Redis（命中即返回，避免每次鉴权都建库连接）
    try:
        from redis_client import get_redis
        rd = get_redis()
        cached = rd.get(rd.key("session", token))
        if cached:
            return cached
    except Exception:
        pass

    # ② 回源数据库（miss 时由 memory_sqlite 回填 Redis）
    try:
        from memory_sqlite import SQLiteMemory
        db = SQLiteMemory()
        account_id = db.get_account_id_by_token(token)
        db.close()

        if account_id is None:
            raise HTTPException(
                status_code=401,
                detail="认证令牌无效或已过期，请重新登录",
                headers={"WWW-Authenticate": "Bearer"},
            )

        return account_id

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[FAIL] Token 验证失败: {e}")
        raise HTTPException(
            status_code=500,
            detail="认证服务暂时不可用",
        )


# ==================== 3. 全局组件（懒加载）====================
class AppComponents:
    """
    全局组件容器（单例模式）
    ========================
    第一次请求时懒加载所有后端组件
    """
    def __init__(self):
        self.llm = None
        self.db = None
        self.rag = None
        self.searcher = None
        self.workflow = None
        self._initialized = False

    def ensure_initialized(self):
        """确保组件已初始化（懒加载）"""
        if self._initialized:
            return

        logger.info("[START] 开始初始化后端组件...")

        # -------- 3.1 初始化 LLM --------
        try:
            from llms import QwenLLM
            self.llm = QwenLLM()
            logger.info("[OK] LLM 初始化成功")
        except Exception as e:
            logger.error(f"[FAIL] LLM 初始化失败: {e}")
            raise RuntimeError(f"LLM 初始化失败: {e}")

        # -------- 3.2 初始化数据库 --------
        try:
            from memory_sqlite import SQLiteMemory
            self.db = SQLiteMemory()
            logger.info("[OK] 数据库初始化成功")
        except Exception as e:
            logger.error(f"[FAIL] 数据库初始化失败: {e}")
            raise RuntimeError(f"数据库初始化失败: {e}")

        # -------- 3.3 初始化 RAG（可选） --------
        try:
            from rag_law import RAGLaw
            self.rag = RAGLaw()
            logger.info("[OK] RAG 知识库初始化成功")
        except Exception as e:
            logger.warning(f"[WARN] RAG 未加载: {e}")

        # -------- 3.4 初始化搜索工具（可选） --------
        try:
            from web_search import WebSearchTool
            self.searcher = WebSearchTool()
            logger.info("[OK] 搜索工具初始化成功")
        except Exception as e:
            logger.warning(f"[WARN] 搜索工具未加载: {e}")

        # -------- 3.5 初始化工作流 --------
        try:
            from agent import CRMWorkflow
            self.workflow = CRMWorkflow(
                llm=self.llm,
                db=self.db,
                rag=self.rag,
                searcher=self.searcher,
            )
            logger.info("[OK] LangGraph 工作流初始化成功")
        except Exception as e:
            logger.error(f"[FAIL] 工作流初始化失败: {e}")
            raise RuntimeError(f"工作流初始化失败: {e}")

        self._initialized = True
        logger.info("[DONE] 所有后端组件初始化完成")


# 全局单例
components = AppComponents()


# ==================== 4. FastAPI 应用 ====================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期管理
    ==================
    startup：应用启动时预初始化组件
    shutdown：清理资源
    """
    logger.info("[GLOBE] FastAPI 服务器启动中...")
    # 启动时预初始化组件（在生命周期的开始而非第一个请求）
    try:
        components.ensure_initialized()
        logger.info("[OK] 组件预初始化完成")
    except Exception as e:
        logger.warning(f"[WARN] 组件预初始化失败（将在首次请求时重试）: {e}")

    # -------- 预热 RAG 嵌入/重排模型（避免首个请求冷启动阻塞） --------
    # 背景：BGE 嵌入模型为懒加载，首个 RAG 请求会触发模型加载（数十秒），
    #       在单进程服务下会阻塞所有并发请求，导致压测/首屏体验极差。
    #       此处启动时主动加载一次，把冷启动成本前移到服务启动阶段。
    #       配合 HF_HUB_OFFLINE=1 使用本地缓存，避免联网 HEAD 校验超时。
    try:
        if components.rag is not None:
            components.rag.warmup()
            logger.info("[OK] RAG 嵌入/重排模型预热完成")
    except Exception as e:
        logger.warning(f"[WARN] RAG 预热失败（不影响主服务，首个 RAG 请求将冷启动）: {e}")

    # -------- 启动常驻定时调度（B.1 ⑤ 兜底「主动告警推送」） --------
    # 后台守护线程按固定间隔自动巡检异常客户 / 到期跟进，结果落 alerts 表。
    # 失败不影响主服务启动（调度是增强能力，不是主链路依赖）。
    try:
        from scheduler import get_scheduler
        get_scheduler().start()
    except Exception as e:
        logger.warning(f"[WARN] 常驻定时调度启动失败（不影响主服务）: {e}")

    yield

    # -------- 关闭常驻定时调度 --------
    try:
        from scheduler import get_scheduler
        get_scheduler().stop()
    except Exception as e:
        logger.warning(f"[WARN] 常驻定时调度停止异常: {e}")
    logger.info("[WAVE] FastAPI 服务器关闭")


app = FastAPI(
    title="AI CRM API",
    description="AI 客户经营系统外部 API 接口",
    version="1.0.0",
    lifespan=lifespan,
)

# 注册速率限制异常处理器
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS 中间件（从环境变量读取白名单）
# 格式：逗号分隔的域名列表，如 "http://localhost:8501,https://myapp.com"
# 默认仅允许本地 Streamlit 前端访问
CORS_ORIGINS: str = os.getenv("CORS_ORIGINS", "http://localhost:8501")
CORS_ORIGINS_LIST = [origin.strip() for origin in CORS_ORIGINS.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS_LIST,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger.info(f"[LOCK] CORS 允许的域名: {CORS_ORIGINS_LIST}")


# ==================== 5. API 端点 ====================

# -------- 5.1 根路径 --------
@app.get("/")
async def root():
    """API 欢迎页"""
    return {
        "service": "AI CRM API",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "/api/status",
        "auth_required": "请使用 Authorization: Bearer <token> 访问受保护端点",
    }


# -------- 5.2 系统健康检查（免认证） --------
@app.get("/api/status", response_model=StatusResponse)
@_limit(_RATE_LIMIT_STATUS)  # 健康检查也做速率限制，但频率宽松
async def get_status(request: Request):
    """
    系统健康检查
    ==============
    返回各组件的就绪状态（免认证）
    """
    try:
        components.ensure_initialized()
        all_ready = all([
            components.llm is not None,
            components.db is not None,
            components.workflow is not None,
        ])
        return StatusResponse(
            status="ready" if all_ready else "degraded",
            llm=components.llm is not None,
            database=components.db is not None,
            rag=components.rag is not None,
            search=components.searcher is not None,
            workflow=components.workflow is not None,
            db_engine=DB_ENGINE,
            redis=_redis_health(),
        )
    except Exception as e:
        return StatusResponse(
            status="error",
            llm=False,
            database=False,
            rag=False,
            search=False,
            workflow=False,
            db_engine=DB_ENGINE,
            redis=_redis_health(),
        )


# -------- 5.2b 运行时指标（监控大盘数据源） --------
@app.get("/api/metrics")
async def get_metrics(recent_events: int = 20):
    """
    运行时指标快照（JSON）
    ======================
    返回「系统运行指标 + 模型效果指标」：
      · 意图路由命中率 / 链路分布 / 意图分布
      · RAG 检索耗时与条数、SQL 校验放行/拒绝率、工具调用失败率
      · 最近事件（badcase 线索）
    """
    return snapshot(include_events=max(0, recent_events))


@app.get("/metrics")
async def prometheus_metrics():
    """Prometheus 文本暴露格式（供 Prometheus / Grafana 抓取）"""
    return PlainTextResponse(
        render_prometheus(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


# -------- 5.3 AI 对话接口（需认证） --------
@app.post("/api/chat", response_model=ChatResponse)
@_limit(_RATE_LIMIT_CHAT)  # 每用户每分钟最多 30 次对话（可用 RATE_LIMIT_CHAT 覆盖）
async def chat(
    body: ChatRequest,
    request: Request,
    account_id: str = Depends(get_current_user),
):
    """
    AI 对话接口
    =============
    需要认证：
      Authorization: Bearer <token>
    
    参数：
      query      : 用户问题（必填）
      user_id    : 用户 ID（可选，默认 api_user）
      customer_id : 客户画像 ID（可选）
    
    速率限制：
      30 次/分钟/用户
    """
    if not body.query.strip():
        incr("api_request_total", endpoint="chat", status="rejected")
        return ChatResponse(success=False, error="query 不能为空")

    _t0 = time.perf_counter()
    try:
        # 懒加载组件
        components.ensure_initialized()

        # 记录请求来源（用于审计）
        logger.info(f"[LOCK] API 请求来自 account_id={account_id}")

        # 构建客户信息（如果指定了 customer_id）
        customer_info = {}
        if body.customer_id:
            try:
                customer = components.db.get_customer(body.customer_id)
                if customer:
                    customer_info = {
                        "name": customer.name,
                        "company": customer.company or "",
                        "position": customer.position or "",
                        "phone": customer.phone or "",
                        "email": customer.email or "",
                        "tags": customer.tags or [],
                        "notes": customer.notes or "",
                    }
                    logger.info(f"[USER] 已关联客户画像: {customer.name}")
            except Exception as e:
                logger.warning(f"[WARN] 查询客户画像失败: {e}")

        # 执行工作流
        logger.info(f"[CHAT] 对话请求: '{body.query[:50]}...' (用户: {body.user_id})")
        response = components.workflow.run(
            query=body.query.strip(),
            user_id=body.user_id,
            customer_info=customer_info,
        )

        # 埋点：API 层成功计数 + 端到端耗时
        incr("api_request_total", endpoint="chat", status="success")
        observe("api_chat_latency_ms", (time.perf_counter() - _t0) * 1000.0)
        return ChatResponse(success=True, response=response)

    except Exception as e:
        logger.error(f"[FAIL] 对话处理失败: {e}")
        incr("api_request_total", endpoint="chat", status="error")
        observe("api_chat_latency_ms", (time.perf_counter() - _t0) * 1000.0,
                status="error")
        return ChatResponse(success=False, error=str(e))


# -------- 5.4 上传文档到知识库（需认证 + 速率限制） --------
ALLOWED_UPLOAD_EXTENSIONS = {".txt", ".md", ".pdf", ".docx"}
MAX_UPLOAD_SIZE = int(os.getenv("MAX_UPLOAD_SIZE", str(10 * 1024 * 1024)))  # 默认 10MB


@app.post("/api/rag/upload", response_model=RagResponse)
@_limit("10/hour")  # 每用户每小时最多上传 10 个文件
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    account_id: str = Depends(get_current_user),
):
    """
    上传文档到知识库
    =================
    需要认证：
      Authorization: Bearer <token>
    
    参数：
      file : 上传的文件（仅支持 .txt / .md / .pdf / .docx，最大 10MB）
    
    速率限制：
      10 次/小时/用户
    
    安全校验：
      - 文件扩展名白名单
      - 文件大小上限
    """
    try:
        components.ensure_initialized()

        if components.rag is None:
            return RagResponse(success=False, error="知识库未初始化")

        # -------- 文件扩展名校验 --------
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in ALLOWED_UPLOAD_EXTENSIONS:
            return RagResponse(
                success=False,
                error=f"不支持的文件类型 '{ext}'。允许的类型: {', '.join(ALLOWED_UPLOAD_EXTENSIONS)}",
            )

        # -------- 读取文件内容 --------
        content = await file.read()

        # -------- 文件大小校验 --------
        if len(content) > MAX_UPLOAD_SIZE:
            max_size_mb = MAX_UPLOAD_SIZE / (1024 * 1024)
            return RagResponse(
                success=False,
                error=f"文件过大 ({len(content)/(1024*1024):.1f}MB)，最大允许 {max_size_mb:.0f}MB",
            )

        # -------- 保存上传文件（使用 UUID 重命名防路径遍历） --------
        import uuid
        upload_dir = "uploads"
        os.makedirs(upload_dir, exist_ok=True)
        safe_filename = f"{uuid.uuid4().hex}{ext}"
        file_path = os.path.join(upload_dir, safe_filename)

        with open(file_path, "wb") as f:
            f.write(content)

        logger.info(f"[DOC] 文件已保存: {file_path} ({len(content)} bytes, 上传者: {account_id})")

        # -------- 添加到知识库 --------
        chunk_count = components.rag.add_file(file_path)
        logger.info(f"[OK] 文件已导入知识库: {chunk_count} 个分块")

        return RagResponse(success=True, chunk_count=chunk_count)

    except Exception as e:
        logger.error(f"[FAIL] 上传文档失败: {e}")
        return RagResponse(success=False, error=str(e))


# -------- 5.5 添加文本到知识库（需认证 + 速率限制） --------
@app.post("/api/rag/text", response_model=RagResponse)
@_limit("30/hour")  # 每用户每小时最多添加 30 次文本
async def add_text(
    body: RagTextRequest,
    request: Request,
    account_id: str = Depends(get_current_user),
):
    """
    添加文本到知识库
    =================
    需要认证：
      Authorization: Bearer <token>
    
    参数：
      text     : 文本内容（必填）
      metadata : 元数据（可选）
    
    速率限制：
      30 次/小时/用户
    """
    if not body.text.strip():
        return RagResponse(success=False, error="text 不能为空")

    # 文本长度校验（防止滥用）
    if len(body.text) > 100000:  # 10 万字上限
        return RagResponse(success=False, error="文本内容过长（上限 100000 字符）")

    try:
        components.ensure_initialized()

        if components.rag is None:
            return RagResponse(success=False, error="知识库未初始化")

        # 添加到知识库
        chunk_count = components.rag.add_text(
            text=body.text,
            metadata=body.metadata,
        )
        logger.info(f"[OK] 文本已导入知识库: {chunk_count} 个分块 (上传者: {account_id})")

        return RagResponse(success=True, chunk_count=chunk_count)

    except Exception as e:
        logger.error(f"[FAIL] 添加文本失败: {e}")
        return RagResponse(success=False, error=str(e))


# ==================== 5.6 跟进闭环接口（B.4 · 跟进记录 / 待跟进 / 转化统计）====================
class FollowUpRequest(BaseModel):
    """新增跟进记录请求（customer_id 与 name 二选一）"""
    customer_id: Optional[str] = None             # 客户 ID（优先）
    name: Optional[str] = None                    # 客户姓名（未给 ID 时按姓名匹配）
    channel: str = "电话"                          # 跟进方式
    content: Optional[str] = None                 # 跟进内容摘要
    result: str = "已联系"                         # 跟进结果
    next_follow_up_days: Optional[int] = None      # N 天后再跟进（写入 next_follow_up_at）
    advisor_id: Optional[str] = None               # 跟进人


def _follow_up_to_dict(record) -> dict:
    """FollowUp → 可 JSON 序列化的 dict（时间统一转 ISO 字符串）"""
    def _iso(dt):
        return dt.isoformat() if dt else None
    return {
        "follow_up_id": record.follow_up_id,
        "customer_id": record.customer_id,
        "advisor_id": record.advisor_id,
        "channel": record.channel,
        "content": record.content,
        "result": record.result,
        "next_follow_up_at": _iso(record.next_follow_up_at),
        "created_at": _iso(record.created_at),
    }


# -------- 5.6.1 跟进记录列表（需认证） --------
@app.get("/api/follow_ups")
@_limit("120/minute")
async def list_follow_ups(
    request: Request,
    customer_id: Optional[str] = None,
    limit: int = 50,
    account_id: str = Depends(get_current_user),
):
    """
    查询跟进记录（按跟进时间倒序）
    ================================
    需要认证：Authorization: Bearer <token>
    参数：
      customer_id : 可选；为空则返回全部客户的跟进记录
      limit       : 最多返回条数（默认 50）
    """
    try:
        components.ensure_initialized()
        if components.db is None:
            raise HTTPException(status_code=503, detail="数据库未初始化")

        if limit <= 0 or limit > 500:
            limit = 50

        records = components.db.list_follow_ups(
            customer_id=(customer_id or None), limit=limit
        )
        return {
            "success": True,
            "count": len(records),
            "items": [_follow_up_to_dict(r) for r in records],
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[FAIL] 查询跟进记录失败: {e}")
        return {"success": False, "count": 0, "items": [], "error": str(e)}


# -------- 5.6.2 新增跟进记录（需认证，自动回写画像最近跟进时间） --------
@app.post("/api/follow_ups")
@_limit("60/hour")
async def create_follow_up(
    request: Request,
    body: FollowUpRequest,
    account_id: str = Depends(get_current_user),
):
    """
    新增一条跟进记录
    =================
    需要认证：Authorization: Bearer <token>

    副作用（写入权限纪律）：
      1. 写入 follow_ups 独立跟进记录表
      2. 回写客户画像 last_follow_up（低风险时间戳，触发下一轮异常筛查自动复位）
      3. 不改写 active_status 等高风险管理字段（仍走人工审核）
    """
    try:
        components.ensure_initialized()
        if components.db is None:
            raise HTTPException(status_code=503, detail="数据库未初始化")

        from datetime import timedelta
        from models import FollowUp

        # -------- 定位客户：ID 优先，其次按姓名精确/模糊匹配 --------
        target_id = (body.customer_id or "").strip()
        if not target_id and (body.name or "").strip():
            matches = components.db.search_customers(body.name.strip())
            exact = [c for c in matches if c.name == body.name.strip()]
            picked = exact or matches
            if picked:
                target_id = picked[0].customer_id

        if not target_id:
            raise HTTPException(
                status_code=404,
                detail="未找到对应客户，请提供 customer_id 或准确的 name",
            )

        next_at = None
        if body.next_follow_up_days is not None:
            try:
                next_at = datetime.now() + timedelta(days=int(body.next_follow_up_days))
            except (ValueError, TypeError):
                next_at = None

        record = FollowUp(
            customer_id=target_id,
            advisor_id=(body.advisor_id or "").strip() or None,
            channel=(body.channel or "电话").strip(),
            content=(body.content or "").strip() or None,
            result=(body.result or "已联系").strip(),
            next_follow_up_at=next_at,
        )
        ok = components.db.save_follow_up(record)
        if not ok:
            return {"success": False, "error": "跟进记录保存失败"}

        logger.info(f"[USER] 跟进记录已创建: {target_id} (操作人: {account_id})")
        return {"success": True, "item": _follow_up_to_dict(record)}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[FAIL] 新增跟进记录失败: {e}")
        return {"success": False, "error": str(e)}


# -------- 5.6.3 待跟进提醒清单（到期巡检，需认证） --------
@app.get("/api/follow_ups/due")
@_limit("120/minute")
async def list_due_follow_ups(
    request: Request,
    days: int = 14,
    limit: int = 100,
    account_id: str = Depends(get_current_user),
):
    """
    待跟进提醒清单（定时巡检可复用此接口）
    ========================================
    参数：
      days  : 「超期未跟进」阈值天数（默认 14 天）
      limit : 最多返回条数（默认 100）
    返回两类：
      计划到期   —— next_follow_up_at 已到点且此后无新跟进
      超期未跟进 —— last_follow_up 距今超过 days 天，或从未跟进
    """
    try:
        components.ensure_initialized()
        if components.db is None:
            raise HTTPException(status_code=503, detail="数据库未初始化")

        if days <= 0:
            days = 14
        if limit <= 0 or limit > 500:
            limit = 100

        items = components.db.list_due_follow_ups(due_days=days, limit=limit)
        planned = [i for i in items if i.get("due_type") == "计划到期"]
        overdue = [i for i in items if i.get("due_type") == "超期未跟进"]
        return {
            "success": True,
            "due_days": days,
            "total": len(items),
            "planned_due": len(planned),
            "overdue": len(overdue),
            "items": items,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[FAIL] 待跟进清单查询失败: {e}")
        return {"success": False, "total": 0, "items": [], "error": str(e)}


# -------- 5.6.4 跟进转化统计（需认证，无数据返回 null 而非 0） --------
@app.get("/api/follow_ups/stats")
@_limit("120/minute")
async def follow_up_stats(
    request: Request,
    days: int = 30,
    account_id: str = Depends(get_current_user),
):
    """
    跟进转化统计
    ==============
    口径（回答「转化率多少」前必须先声明）：
      分母 = 窗口内跟进记录条数
      分子 = 结果为「已成交 / 有意向」的跟进条数
      转化率 = 分子 ÷ 分母；无数据时 conversion_rate = null（不报 0 冒充）
    """
    try:
        components.ensure_initialized()
        if components.db is None:
            raise HTTPException(status_code=503, detail="数据库未初始化")

        if days <= 0:
            days = 30
        stats = components.db.follow_up_conversion_stats(days=days)
        stats["success"] = True
        return stats
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[FAIL] 跟进转化统计失败: {e}")
        return {
            "success": False, "window_days": days, "total": 0, "converted": 0,
            "conversion_rate": None, "covered_customers": 0, "by_result": {},
            "error": str(e),
        }


# ==================== 5.7 主动告警（B.1 ⑤ 兜底 · 定时调度产物） ====================
# 背景：异常筛查此前是「被动查询」（用户问了才跑）；scheduler.py 常驻定时巡检后，
#       命中结果落 alerts 表，顾问侧通过下面这些接口拉取「系统主动推来的」告警。
class AlertReadRequest(BaseModel):
    """告警已读请求"""
    alert_id: Optional[str] = None        # 单条已读；为空则批量已读
    advisor_id: Optional[str] = None      # 批量已读时按顾问过滤（可选）


# -------- 5.7.1 告警列表（需认证） --------
@app.get("/api/alerts")
@_limit("120/minute")
async def list_alerts(
    request: Request,
    status: Optional[str] = None,
    advisor_id: Optional[str] = None,
    limit: int = 100,
    account_id: str = Depends(get_current_user),
):
    """
    主动告警列表（前端红点 / 角标数据源）
    ======================================
    参数：
      status     : unread / read；为空返回全部
      advisor_id : 按归属顾问过滤；为空返回全部
      limit      : 最多返回条数（默认 100）
    返回：
      {success, total, unread, items:[...]}
    """
    try:
        components.ensure_initialized()
        if components.db is None:
            raise HTTPException(status_code=503, detail="数据库未初始化")

        if limit <= 0 or limit > 500:
            limit = 100
        if status not in ("unread", "read"):
            status = None

        items = components.db.list_alerts(
            status=status, advisor_id=advisor_id, limit=limit
        )
        return {
            "success": True,
            "total": len(items),
            "unread": components.db.count_unread_alerts(advisor_id=advisor_id),
            "items": items,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[FAIL] 告警列表查询失败: {e}")
        return {"success": False, "total": 0, "unread": 0, "items": [], "error": str(e)}


# -------- 5.7.2 未读告警数（轻量轮询，前端角标） --------
@app.get("/api/alerts/unread_count")
@_limit("240/minute")
async def unread_alert_count(
    request: Request,
    advisor_id: Optional[str] = None,
    account_id: str = Depends(get_current_user),
):
    """未读告警数（前端定时轮询刷新红点，比拉全量列表更省）"""
    try:
        components.ensure_initialized()
        if components.db is None:
            raise HTTPException(status_code=503, detail="数据库未初始化")
        return {
            "success": True,
            "unread": components.db.count_unread_alerts(advisor_id=advisor_id),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[FAIL] 未读告警统计失败: {e}")
        return {"success": False, "unread": 0, "error": str(e)}


# -------- 5.7.3 标记已读（单条 / 批量） --------
@app.post("/api/alerts/read")
@_limit("120/minute")
async def mark_alerts_read(
    request: Request,
    body: AlertReadRequest,
    account_id: str = Depends(get_current_user),
):
    """
    标记告警已读
    ==============
    body.alert_id 有值 → 单条已读；
    body.alert_id 为空 → 批量把未读全部已读（可按 advisor_id 过滤）。
    """
    try:
        components.ensure_initialized()
        if components.db is None:
            raise HTTPException(status_code=503, detail="数据库未初始化")

        if body.alert_id:
            ok = components.db.mark_alert_read(body.alert_id)
            return {"success": ok, "mode": "single", "affected": 1 if ok else 0}
        affected = components.db.mark_all_alerts_read(advisor_id=body.advisor_id)
        return {"success": True, "mode": "batch", "affected": affected}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[FAIL] 标记告警已读失败: {e}")
        return {"success": False, "affected": 0, "error": str(e)}


# -------- 5.7.4 调度器状态 + 手动触发一轮巡检（需认证） --------
@app.get("/api/alerts/scheduler")
@_limit("60/minute")
async def scheduler_status(
    request: Request,
    run_now: bool = False,
    account_id: str = Depends(get_current_user),
):
    """
    常驻定时调度状态（可选立即跑一轮）
    =====================================
    参数：
      run_now : true 时同步执行一轮巡检（用于演示 / 手动补跑）
    返回：
      调度器健康状态 + （run_now 时）本轮巡检统计
    """
    try:
        from scheduler import get_scheduler
        sched = get_scheduler()
        payload = {"success": True, "scheduler": sched.status()}
        if run_now:
            payload["run_result"] = sched.run_once()
            payload["scheduler"] = sched.status()
        return payload
    except Exception as e:
        logger.error(f"[FAIL] 调度器状态查询失败: {e}")
        return {"success": False, "error": str(e)}


# ==================== 6. 启动入口 ====================
if __name__ == "__main__":
    import uvicorn

    # 命令行参数解析
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    reload = os.getenv("API_RELOAD", "false").lower() == "true"
    # 多进程 worker 数：/api/chat 会阻塞在 LLM 网络调用上，
    # 单 worker 在并发场景下会排队超时；压测/生产建议按 CPU 核数设置。
    # 注意：workers>1 时不能与 reload=True 同时使用（uvicorn 限制）。
    workers = int(os.getenv("API_WORKERS", "1"))
    if reload and workers > 1:
        logger.warning("[WARN] reload=True 与 workers>1 不兼容，已强制 workers=1")
        workers = 1

    print(f"""
╔═══════════════════════════════════════════════════╗
║     AI CRM API Server                             ║
║     http://{host}:{port}                             ║
║     API Docs: http://{host}:{port}/docs            ║
║     workers={workers}  reload={reload}                        ║
║                                                   ║
║     [LOCK] 安全特性已启用:                               ║
║     [OK] JWT Token 认证（受保护端点需 Bearer Token）    ║
║     [OK] CORS 白名单: {CORS_ORIGINS}                ║
║     [OK] Rate Limiting（防暴力破解/DoS）               ║
║     [OK] 文件上传校验（类型+大小白名单）                 ║
╚═══════════════════════════════════════════════════╝
    """)

    uvicorn.run(
        "api:app",
        host=host,
        port=port,
        reload=reload,
        workers=workers,
        log_level="info",
    )
