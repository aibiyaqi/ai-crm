"""
config.py - 全局配置模块
===================================
功能分区：
  1. 环境变量加载（.env → QWEN_API_KEY）
  2. 模型参数配置（temperature / max_tokens）
  3. 检索参数配置（TopK / 分块大小）
  4. Token 管控参数（对话轮次 / 上下文长度）
  5. 数据库配置（SQLite 路径）
  6. 日志配置（文件路径 / 级别）
  7. 向量数据库路径
  8. 意图规则路由配置
  9. 嵌入模型配置（text-embedding-v2）
  10. 重排序（Reranker）配置
  11. 认证安全配置
"""

# ==================== 1. 环境变量加载 ====================
import os                      # 操作系统接口，用于读取环境变量
import sys                     # 系统接口，用于启动时退出
from dotenv import load_dotenv # python-dotenv 库，用于加载 .env 文件

# 加载 .env 文件中的环境变量（API Key 等敏感信息）
# 如果 .env 不存在则静默跳过，不会报错
load_dotenv()

# ==================== 1b. 启动安全校验 ====================
def validate_env_or_exit():
    """
    启动时校验关键环境变量
    ========================
    检查所有必须的环境变量是否已设置，缺失则直接退出。
    这是防止配置遗漏的第一道防线。
    """
    required_vars = {
        "QWEN_API_KEY": "通义千问 API Key（从 https://help.aliyun.com/zh/dashscope/ 获取）",
    }

    all_ok = True
    for var_name, hint in required_vars.items():
        if not os.getenv(var_name):
            print(f"[启动校验失败] 环境变量 '{var_name}' 未设置！")
            print(f"   提示：{hint}")
            if var_name == "QWEN_API_KEY":
                print(f"   设置方式：set {var_name}=sk-xxx 或 在 .env 文件中添加 {var_name}=sk-xxx")
            all_ok = False

    # 检查 ADMIN_PASSWORD（生产环境必须设置）
    if not os.getenv("ADMIN_PASSWORD"):
        print(f"[启动警告] 环境变量 'ADMIN_PASSWORD' 未设置，将使用随机生成的管理员密码")
        print(f"   提示：生产环境请务必设置 ADMIN_PASSWORD=your_secure_password")

    if not all_ok:
        print("\n[启动终止] 因关键环境变量缺失，系统启动终止。请配置后重试。")
        sys.exit(1)

    print("[启动校验通过] 所有关键环境变量已就绪")

# ==================== 2. 模型参数配置 ====================
# 通义千问 API Key（从 .env 读取，读取不到则返回 None）
# 启动时 validate_env_or_exit() 会确保此值存在
QWEN_API_KEY: str = os.getenv("QWEN_API_KEY")

# 通义千问模型名称（免费版 qwen-plus / qwen-turbo 够用）
QWEN_MODEL_NAME: str = "qwen-plus"

# 模型采样温度：0.0-1.0，值越低回答越保守确定
# 风险管控：设为 0.3 避免创意跑偏，适合企业场景
TEMPERATURE: float = 0.3

# 单次对话最大 Token 数（输入 + 输出）
# 通义千问 qwen-plus 最大支持 131072 Token
# 设为 4096 兼顾回答质量与响应速度
MAX_TOKENS: int = 4096

# ==================== 3. 检索参数配置（RAG） ====================
# 知识库检索时召回的最相关文档片段数
# TopK=3～5：精度与召回率的平衡点
RETRIEVAL_TOP_K: int = 4

# 文档分块大小（按字符数）
# 太大：超出模型上下文；太小：信息不完整
CHUNK_SIZE: int = 500

# 文档块重叠字符数（避免切段导致信息断裂）
CHUNK_OVERLAP: int = 100

# ==================== 4. Token 管控参数 ====================
# 对话历史保留的最大轮次（用户+助手算一轮）
# 限制历史轮次 → 控制 Token 消耗 → 省钱防超限
MAX_HISTORY_ROUNDS: int = 5

# 对话历史保留的最大 Token 数（按 Token 动态计数，而非只按轮数）
# 背景：只按"轮数"限，防不住"单轮超长"（用户贴一万字 PDF，5 轮窗口直接爆）；
#       改为"轮数 + Token 双闸门"：先按轮数取候选，再按 Token 从新到旧累加裁剪。
MAX_HISTORY_TOKENS: int = int(os.getenv("MAX_HISTORY_TOKENS", "3000"))

# 单条消息最大字符数（入口拦截超长单条消息）
# 背景：单条超长消息（如整篇 PDF 粘贴）会一次性撑爆上下文窗口；
#       入口处对超长消息做截断 + 提示，避免"一条消息打爆整个会话"。
MAX_SINGLE_MESSAGE_CHARS: int = int(os.getenv("MAX_SINGLE_MESSAGE_CHARS", "4000"))

# 裁剪兜底：当 Token 裁剪后仍超限（极端场景）时，只保留最近 N 轮
# 背景：防御底线——宁可丢历史，也要保证主链路不因上下文超限而卡死。
MIN_HISTORY_ROUNDS_ON_OVERFLOW: int = int(os.getenv("MIN_HISTORY_ROUNDS_ON_OVERFLOW", "2"))

# ==================== 5. 数据库配置（DAO 层 / 引擎切换位） ====================
# 数据库引擎：sqlite（本地开发/回滚）/ mysql（生产默认，P1 已落地）
# 切换迁移：只改 DB_ENGINE 即可，业务代码经 db.get_database() 自动适配
DB_ENGINE: str = os.getenv("DB_ENGINE", "sqlite")

# SQLite 数据库文件路径（相对于项目根目录）
# SQLite 是文件型数据库，无需额外安装数据库服务
SQLITE_PATH: str = os.getenv("DB_SQLITE_PATH", "crm.db")
DB_SQLITE_PATH: str = SQLITE_PATH

# MySQL 连接参数（P1 已落地：见 db.MySQLDatabase）
DB_MYSQL_HOST: str = os.getenv("DB_MYSQL_HOST", "127.0.0.1")
DB_MYSQL_PORT: int = int(os.getenv("DB_MYSQL_PORT", "3306"))
DB_MYSQL_USER: str = os.getenv("DB_MYSQL_USER", "crm_app")
DB_MYSQL_PASSWORD: str = os.getenv("DB_MYSQL_PASSWORD", "")
DB_MYSQL_DATABASE: str = os.getenv("DB_MYSQL_DATABASE", "crm")

# 连接池（DBUtils.PooledDB）：池满时阻塞等待，避免直接抛连接异常
DB_MYSQL_POOL_SIZE: int = int(os.getenv("DB_MYSQL_POOL_SIZE", "10"))
DB_MYSQL_MAX_OVERFLOW: int = int(os.getenv("DB_MYSQL_MAX_OVERFLOW", "20"))
# 连接回收时间（秒）：小于 MySQL wait_timeout，避免拿到被服务端踢掉的连接
DB_MYSQL_POOL_RECYCLE: int = int(os.getenv("DB_MYSQL_POOL_RECYCLE", "1800"))

# 字符集与时区：utf8mb4 支持中文/emoji；+08:00 与应用侧 datetime.now() 对齐
DB_MYSQL_CHARSET: str = os.getenv("DB_MYSQL_CHARSET", "utf8mb4")
DB_MYSQL_TIMEZONE: str = os.getenv("DB_MYSQL_TIMEZONE", "+08:00")

# 连接/读写超时（秒）：连不上时快速失败，不拖住主链路
DB_MYSQL_CONNECT_TIMEOUT: int = int(os.getenv("DB_MYSQL_CONNECT_TIMEOUT", "5"))
DB_MYSQL_READ_TIMEOUT: int = int(os.getenv("DB_MYSQL_READ_TIMEOUT", "30"))
DB_MYSQL_WRITE_TIMEOUT: int = int(os.getenv("DB_MYSQL_WRITE_TIMEOUT", "30"))

# 只读账号（NL2SQL / sql_agent 专用）：留空则用主账号 + "会话级只读"兜底
# 生产建议单独建账号：GRANT SELECT ON crm.* TO 'crm_ro'@'%';
DB_MYSQL_READONLY_USER: str = os.getenv("DB_MYSQL_READONLY_USER", "")
DB_MYSQL_READONLY_PASSWORD: str = os.getenv("DB_MYSQL_READONLY_PASSWORD", "")

# MySQL 建表脚本（相对 AI_CRM 目录）
DB_MYSQL_SCHEMA_FILE: str = os.getenv("DB_MYSQL_SCHEMA_FILE", "sql/mysql/001_init.sql")

# ==================== 6. 日志配置 ====================
# 日志文件路径
LOG_FILE_PATH: str = "logfile/app.log"

# 日志级别：DEBUG / INFO / WARNING / ERROR
# 开发阶段用 DEBUG（最详细），生产用 INFO
LOG_LEVEL: str = "DEBUG"

# ==================== 7. 向量数据库路径 ====================
# Chroma 向量数据库持久化目录（知识库 = 内部 K12 课程产品资料）
# 命名演进：chroma_law → chroma_kb（"知识库"语义，非"法规"语义）
# 迁移说明：目录/集合已重命名；rag_law 启动时对集合名做幂等迁移（保留向量，不重建）
CHROMA_PERSIST_DIR: str = "utils/chroma_kb"

# Chroma 集合名称（知识库集合）
CHROMA_COLLECTION_NAME: str = "crm_kb"

# ==================== 8. 意图规则路由配置 ====================
# 轻量级规则兜底：通过关键词快速匹配意图，减少 LLM 调用
# 优先级：关键词匹配 > LLM 分类
# 格式：{意图: [(关键词列表), 置信度阈值(匹配数)]}
# 匹配逻辑：用户问题中包含任意关键词即计数，超过阈值则命中
INTENT_RULES: dict = {
    # 实时信息类 - 匹配任何 1 个关键词即可
    "web_search": {
        "keywords": [
            "新闻", "天气", "股票", "最新", "今天", "实时",
            "热搜", "股市", "汇率", "地震", "疫情", "油价",
            "黄金", "今天天气", "最新消息", "最近", "目前",
            "当前", "有什么新闻", "热点", "快讯", "涨停",
            "跌停", "大盘", "港股", "A股", "天气预报",
            "台风", "暴雨", "高温", "降温", "路况",
        ],
        "min_match": 1,
    },
    # 业务咨询类（K12 课程产品知识库）- 匹配任何 1 个关键词即可
    # 口径：知识库 = 内部 K12 课程产品资料（课程 / 班型 / 师资 / 价格 / 报名 等）
    "rag_retrieve": {
        "keywords": [
            # —— 课程产品 ——
            "课程", "班型", "课时", "学费", "收费", "报价", "价格",
            "续费", "退费", "退课", "试听", "报名", "优惠", "活动",
            "师资", "教材", "大纲", "排课", "校区", "上课",
            "年级", "学科", "科目", "课程体系", "培养方案",
            "小升初", "初升高", "中考", "高考", "竞赛", "一对一", "小班",
            # —— 通用咨询问法 ——
            "什么是", "怎么", "如何", "方法", "步骤",
            "说明", "介绍", "解释", "定义", "含义", "概念",
            "区别", "对比", "有哪些", "包括", "包含", "相关",
        ],
        "min_match": 1,
    },
}

# ==================== 9. 嵌入模型配置（RAG 向量化） ====================
# 使用通义千问 text-embedding-v2 API（DashScope 在线 API，零下载）
# 替代原 HuggingFace all-MiniLM-L6-v2（需下载约 2GB 模型文件）
# 国内用户无需 VPN，直接通过阿里云 DashScope 服务调用
EMBEDDING_MODEL: str = "text-embedding-v2"

# ==================== 9b. 本地嵌入模型配置（方案A，主力推荐） ====================
# 使用 BAAI/bge-small-zh-v1.5（~33MB，完全本地运行，中文优化）
# 已缓存到本地，无需联网下载
# True  = 使用 BGE 本地模型（主力，已缓存）
# False = 使用 DashScope text-embedding-v2（在线 API 备用）
USE_LOCAL_EMBEDDING: bool = True

# 本地嵌入模型名称（仅 USE_LOCAL_EMBEDDING=True 时生效）
# BAAI/bge-small-zh-v1.5  ~33MB，中文优化，检索效果好
# BAAI/bge-base-zh-v1.5   ~133MB，精度更高但更慢
# BAAI/bge-large-zh-v1.5  ~400MB，精度最高但资源消耗大
LOCAL_EMBEDDING_MODEL: str = "BAAI/bge-small-zh-v1.5"

# ==================== 9c. 检索质量配置（方案D） ====================
# 向量检索相关性分数阈值
# Chroma similarity_search_with_relevance_scores 返回 0~1 的分数
# 低于此阈值的结果将被过滤掉，避免返回不相关内容
# 建议值：0.5~0.7，越高越严格
RETRIEVAL_SCORE_THRESHOLD: float = 0.5

# 自适应检索（adaptive_search）配置
# 是否启用自适应 TopK（默认 True）
ADAPTIVE_TOP_K: bool = True
# 自适应检索最少返回数
ADAPTIVE_TOP_K_MIN: int = 1
# 自适应检索最多返回数
ADAPTIVE_TOP_K_MAX: int = 6
# 自适应动态截断的分数阈值
# 低于此分数的结果将被截断
ADAPTIVE_TOP_K_SCORE: float = 0.4

# ==================== 10. 重排序（Reranker）配置 ====================
# 使用 Cross-Encoder 对初检结果进行精排，提升 RAG 准确率
# 轻量模型推荐：cross-encoder/ms-marco-MiniLM-L-6-v2 (~80MB)
# 多语言强模型：BAAI/bge-reranker-v2-m3 (~1.2GB)
# 说明：默认使用已本地缓存的 BAAI/bge-reranker-base，避免离线/无 VPN 环境启动时
#       因联网下载 cross-encoder/ms-marco-MiniLM-L-6-v2 而长时间阻塞。
#       如需英文精排可改回 "cross-encoder/ms-marco-MiniLM-L-6-v2"（需先联网缓存）。
RERANKER_MODEL: str = "BAAI/bge-reranker-base"
RERANKER_ENABLED: bool = True           # 是否启用重排序
RERANKER_TOP_K: int = 3                 # 重排序后返回的最终结果数
RERANKER_DEVICE: str = "cpu"            # 运行设备 cpu/cuda
RERANKER_BATCH_SIZE: int = 32           # 批处理大小用于批量评分

# ==================== 11. 认证安全配置 ====================
# 密码加密盐值长度（字节数）
PASSWORD_SALT_LENGTH: int = 32
# PBKDF2 迭代次数（越高越安全，但越慢）
PASSWORD_HASH_ITERATIONS: int = 100000
# 密码最小长度
PASSWORD_MIN_LENGTH: int = 8

# 手机验证码有效期（分钟）
VERIFICATION_CODE_EXPIRE_MINUTES: int = 5
# 登录失败锁定阈值
LOGIN_FAIL_LOCK_THRESHOLD: int = 5
# 登录失败锁定时间（分钟）
LOGIN_LOCK_MINUTES: int = 15

# 管理员默认账户配置（首次初始化时创建）
# 优先从 .env 环境变量读取
# 若 ADMIN_PASSWORD 未设置，则由 AuthService._init_admin_account() 自动生成随机密码
ADMIN_DEFAULT_USERNAME: str = os.getenv("ADMIN_USERNAME") or "admin"
ADMIN_DEFAULT_PASSWORD: str = os.getenv("ADMIN_PASSWORD")  # 无回退！未设置则为 None，由 auth.py 自动生成
ADMIN_DEFAULT_PHONE: str = os.getenv("ADMIN_PHONE") or "13800000000"


# ==================== 12. Redis 配置（统一基础设施） ====================
# 定位：MySQL 是事实源，Redis 是加速与协同层；不可用时降级、不阻断主链路。
# REDIS_ENABLED=0 可整体关闭（等价于"完全降级"，用于纯本地调试/排障）。
REDIS_ENABLED: bool = os.getenv("REDIS_ENABLED", "1").strip().lower() not in ("0", "false", "no", "off")
REDIS_URL: str = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")
REDIS_PASSWORD: str = os.getenv("REDIS_PASSWORD", "")
REDIS_DB: int = int(os.getenv("REDIS_DB", "0"))
REDIS_MAX_CONNECTIONS: int = int(os.getenv("REDIS_MAX_CONNECTIONS", "20"))
# 读写超时（秒）：短超时避免 Redis 抖动拖垮主链路
REDIS_TIMEOUT_S: float = float(os.getenv("REDIS_TIMEOUT_S", "1.0"))
# 熔断：连续失败 N 次后，冷却 M 秒内直接走降级（不再等超时）
REDIS_BREAKER_THRESHOLD: int = int(os.getenv("REDIS_BREAKER_THRESHOLD", "5"))
REDIS_BREAKER_COOLDOWN_S: int = int(os.getenv("REDIS_BREAKER_COOLDOWN_S", "30"))

# 键前缀与各场景 TTL（秒）
REDIS_KEY_PREFIX: str = os.getenv("REDIS_KEY_PREFIX", "crm:")
REDIS_SESSION_TTL: int = int(os.getenv("REDIS_SESSION_TTL", str(7 * 24 * 3600)))   # 会话令牌 7 天
REDIS_LOGIN_FAIL_TTL: int = int(os.getenv("REDIS_LOGIN_FAIL_TTL", "900"))           # 登录失败计数 15 分钟
REDIS_HISTORY_TTL: int = int(os.getenv("REDIS_HISTORY_TTL", "1800"))                # 对话热缓存 30 分钟
REDIS_ALERT_DEDUP_TTL: int = int(os.getenv("REDIS_ALERT_DEDUP_TTL", "86400"))       # 告警去重 24 小时
REDIS_SCAN_LOCK_TTL: int = int(os.getenv("REDIS_SCAN_LOCK_TTL", "300"))             # 跑批锁 5 分钟
