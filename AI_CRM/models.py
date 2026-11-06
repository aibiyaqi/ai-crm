"""
models.py - Pydantic 数据模型模块
====================================
功能分区：
  1. 账户认证模型（Account / PasswordResetToken / LoginLog）
  2. 用户信息模型（User）
  3. 对话消息模型（Message）
  4. 客户画像模型（CustomerProfile）
  5. RAG 文档模型（Document）
  6. Agent 状态模型（AgentState - LangGraph 用）
"""

# ==================== 1. 导入依赖 ====================
from pydantic import BaseModel, Field  # Pydantic 数据校验框架
from typing import Optional, List      # 类型提示
from datetime import datetime          # 日期时间处理
import uuid                            # 通用唯一标识符，生成 ID


# ==================== 2. 账户认证模型 ====================
class Account(BaseModel):
    """
    账户认证模型
    =============
    字段说明：
      account_id    : 账户唯一标识，自动生成 UUID
      username      : 用户名（登录用，唯一）
      password_hash : 密码哈希值（bcrypt/sha256+salt）
      role          : 角色（admin=管理员 / user=普通用户）
      phone         : 手机号（唯一，可用于登录）
      email         : 邮箱（可选）
      is_active     : 账户是否激活
      last_login    : 最后登录时间
      created_at    : 创建时间
      updated_at    : 更新时间
    
    使用场景：
      存储到 SQLite 的 accounts 表
    """
    account_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="账户唯一标识"
    )
    username: str = Field(..., description="用户名（唯一）")
    password_hash: str = Field(..., description="密码哈希值")
    role: str = Field(default="user", description="角色：admin/user")
    phone: str = Field(..., description="手机号（唯一）")
    email: Optional[str] = Field(None, description="邮箱")
    is_active: bool = Field(default=True, description="账户是否激活")
    last_login: Optional[datetime] = Field(None, description="最后登录时间")
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="创建时间"
    )
    updated_at: datetime = Field(
        default_factory=datetime.now,
        description="更新时间"
    )


class PasswordResetToken(BaseModel):
    """
    密码重置令牌模型
    ==================
    字段说明：
      token_id    : 令牌唯一标识
      account_id  : 关联账户 ID
      token       : 重置令牌（UUID 或随机字符串）
      code        : 短信验证码（6位数字）
      expires_at  : 过期时间
      used        : 是否已使用
      created_at  : 创建时间
    
    使用场景：
      存储到 SQLite 的 password_reset_tokens 表
    """
    token_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="令牌唯一标识"
    )
    account_id: str = Field(..., description="关联账户ID")
    token: str = Field(..., description="重置令牌")
    code: str = Field(..., description="短信验证码（6位数字）")
    expires_at: datetime = Field(..., description="过期时间")
    used: bool = Field(default=False, description="是否已使用")
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="创建时间"
    )


class LoginLog(BaseModel):
    """
    登录日志模型
    =============
    字段说明：
      log_id      : 日志唯一标识
      account_id  : 关联账户 ID（失败记录也可能无 account_id）
      username    : 尝试登录的用户名
      action      : 操作类型（login=登录成功 / logout=登出 / fail=登录失败）
      ip_address  : 登录 IP 地址
      detail      : 详细信息（失败原因等）
      created_at  : 操作时间
    
    使用场景：
      存储到 SQLite 的 login_logs 表
      用于安全审计和登录保护
    """
    log_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="日志唯一标识"
    )
    account_id: Optional[str] = Field(None, description="关联账户ID")
    username: str = Field(..., description="用户名")
    action: str = Field(..., description="操作类型：login/logout/fail")
    ip_address: Optional[str] = Field(None, description="登录IP")
    detail: Optional[str] = Field(None, description="详细信息")
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="操作时间"
    )


# ==================== 3. 用户信息模型 ====================
class User(BaseModel):
    """
    用户信息模型
    =============
    字段说明：
      user_id  : 用户唯一标识，自动生成 UUID
      name     : 用户姓名
      phone    : 手机号（可选）
      created_at: 注册时间，默认当前时间
      updated_at: 更新时间，默认当前时间
    
    使用场景：
      存储到 SQLite 的 users 表
    """
    user_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),  # 自动生成 UUID
        description="用户唯一标识"
    )
    name: str = Field(..., description="用户姓名")           # ... 表示必填字段
    phone: Optional[str] = Field(None, description="手机号") # None 表示可选
    created_at: datetime = Field(
        default_factory=datetime.now,                       # 默认当前时间
        description="创建时间"
    )
    updated_at: datetime = Field(
        default_factory=datetime.now,
        description="更新时间"
    )


# ==================== 4. 对话会话模型 ====================
class ChatSession(BaseModel):
    """
    对话会话模型
    =============
    字段说明：
      session_id : 会话唯一标识
      account_id : 所属账户 ID
      title      : 会话标题（自动从首条消息摘要）
      created_at : 创建时间
      updated_at : 最后活跃时间
    
    使用场景：
      支持多对话框切换，类似 ChatGPT 的会话列表
      存储到 SQLite 的 sessions 表
    """
    session_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="会话唯一标识"
    )
    account_id: str = Field(..., description="所属账户ID")
    title: str = Field(default="新对话", description="会话标题")
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="创建时间"
    )
    updated_at: datetime = Field(
        default_factory=datetime.now,
        description="最后活跃时间"
    )


# ==================== 5. 对话消息模型 ====================
class Message(BaseModel):
    """
    对话消息模型
    =============
    字段说明：
      message_id : 消息唯一标识
      session_id : 所属会话 ID（支持多对话框）
      user_id    : 发送消息的用户 ID
      role       : 角色（user=用户 / assistant=助手 / system=系统）
      content    : 消息内容
      created_at : 消息发送时间
    
    使用场景：
      存储对话历史到 SQLite 的 messages 表
      传给 LLM 的 messages 参数
    """
    message_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="消息唯一标识"
    )
    session_id: str = Field(..., description="所属会话ID")
    user_id: str = Field(..., description="用户ID")
    role: str = Field(..., description="角色：user/assistant/system")
    content: str = Field(..., description="消息内容")
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="消息时间"
    )


# ==================== 5. 客户画像模型 ====================
class CustomerProfile(BaseModel):
    """
    客户画像模型（K12 教育 CRM 核心）
    ================================
    字段分组（对应二面「客户画像字段与更新权限」考点）：
      [基础身份] customer_id / name / phone / email / age / identity / school_grade
      [业务意向] intent_course / matched_course / intent_price / hourly_fee
      [经营状态] active_status / advisor_id / source / last_follow_up
      [通用兼容] company / position（保留通用企业字段，兼容历史数据与通用场景）
      [其它]     notes / tags / created_at / updated_at

    更新权限设计（低容错字段必须人工审核，避免模型幻觉直接改库）：
      - 低风险字段：允许 Agent 抽取后自动写入（见 AUTO_UPDATE_FIELDS）
      - 高风险字段：涉价格/权益/成交结论，必须人工确认（见 MANUAL_REVIEW_FIELDS）
    """
    customer_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="客户唯一标识"
    )
    name: str = Field(..., description="客户姓名")
    phone: Optional[str] = Field(None, description="电话")
    email: Optional[str] = Field(None, description="邮箱")

    # -------- 基础身份 --------
    age: Optional[int] = Field(None, description="年龄（学生客户）")
    identity: Optional[str] = Field(None, description="身份：学生/家长/成人")
    school_grade: Optional[str] = Field(
        None, description="就读或工作信息：如 初二 / 高三 / 在职"
    )

    # -------- 业务意向 --------
    intent_course: Optional[str] = Field(None, description="意向课程")
    matched_course: Optional[str] = Field(None, description="适配课程（需顾问确认）")
    intent_price: Optional[str] = Field(None, description="意向价格/预算（高风险字段）")
    hourly_fee: Optional[str] = Field(None, description="课时费（高风险字段）")

    # -------- 经营状态 --------
    active_status: Optional[str] = Field(
        None, description="活跃状态：活跃/待跟进/沉睡/流失风险"
    )
    advisor_id: Optional[str] = Field(None, description="归属课程顾问 ID")
    source: Optional[str] = Field(None, description="客户来源渠道")
    last_follow_up: Optional[datetime] = Field(None, description="最近一次跟进时间")

    # -------- 通用/兼容字段 --------
    company: Optional[str] = Field(None, description="公司名称（通用企业场景兼容）")
    position: Optional[str] = Field(None, description="职位（通用企业场景兼容）")
    notes: Optional[str] = Field(None, description="备注信息")
    tags: List[str] = Field(
        default_factory=list,               # 默认空列表
        description="标签列表，如 ['高意向', '初三', '数学']"
    )
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="创建时间"
    )
    updated_at: datetime = Field(
        default_factory=datetime.now,
        description="更新时间"
    )


# ==================== 5b. 客户画像更新权限（字段白名单） ====================
# 设计原则（对应二面整改要点「数据写入权限」）：
#   - Agent 只能自动更新「低风险、可回溯、可纠错」字段；
#   - 涉价格 / 权益 / 成交结论的字段，必须人工审核后再落库。
# 写入入口（tools.save_customer_profile）会依据这两份白名单做字段分流。
AUTO_UPDATE_FIELDS = {
    "age", "identity", "school_grade",
    "intent_course", "source", "notes", "tags",
}
MANUAL_REVIEW_FIELDS = {
    "matched_course", "intent_price", "hourly_fee",
    "active_status", "advisor_id",
}


# ==================== 5c. 跟进记录模型（B.4 跟进闭环 · 产品化落地） ====================
# 背景（对应二面 B.4「销售到底怎么跟进」的整改）：
#   原设计里跟进进展只落在 customer_profiles 的 last_follow_up / active_status
#   两个字段上——只解决了「多久没跟、是否流失」这两个查询，属于「数据层闭环」；
#   一旦被追问「跟进记录存哪 / 谁在什么时候用什么方式跟成什么样 / 跟进转化率多少」，
#   光靠两个字段答不上来。因此把跟进动作结构化成一条独立记录（谁 · 何时 · 什么方式 ·
#   什么结果 · 下次什么时候跟），于是三件事都能算出来：
#     ① 跟进记录：list_follow_ups() 按客户 / 时间倒序列出
#     ② 跟进提醒：list_due_follow_ups() 按 next_follow_up_at 到期 + 超期未跟进
#     ③ 跟进转化：follow_up_conversion_stats() 用 FOLLOW_UP_CONVERTED_RESULTS 做分子
#   写入权限仍然遵守「低风险自动、高风险人工」：跟进记录由 / 工具写入，
#   active_status 等高风险字段不在本链路自动改写。
FOLLOW_UP_CHANNELS = ["电话", "微信", "面谈", "短信", "邮件", "其他"]
FOLLOW_UP_RESULTS = ["已联系", "未接", "有意向", "已成交", "已放弃"]
# 视为「转化」的结果集合（跟进转化率的分子口径）
FOLLOW_UP_CONVERTED_RESULTS = {"已成交", "有意向"}


class FollowUp(BaseModel):
    """
    跟进记录模型（B.4 跟进闭环的结构化载体）
    ===========================================
    字段说明：
      follow_up_id     : 跟进记录唯一标识（UUID）
      customer_id      : 关联客户 ID（对应 customer_profiles.customer_id）
      advisor_id       : 跟进人（课程顾问 ID）
      channel          : 跟进方式：电话 / 微信 / 面谈 / 短信 / 邮件 / 其他
      content          : 跟进内容摘要（聊了什么、承诺了什么）
      result           : 跟进结果：已联系 / 未接 / 有意向 / 已成交 / 已放弃
      next_follow_up_at: 计划下次跟进时间（到期提醒的判定依据）
      created_at       : 本次跟进时间

    使用场景：
      存储到 SQLite 的 follow_ups 表；
      save_follow_up() 落库同时回写 customer_profiles.last_follow_up，
      使「异常客户筛查（R1/R2）」读到的最近跟进时间始终是最新的。
    """
    follow_up_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="跟进记录唯一标识"
    )
    customer_id: str = Field(..., description="关联客户ID")
    advisor_id: Optional[str] = Field(None, description="跟进人（课程顾问ID）")
    channel: str = Field(
        default="电话",
        description="跟进方式：电话/微信/面谈/短信/邮件/其他"
    )
    content: Optional[str] = Field(None, description="跟进内容摘要")
    result: str = Field(
        default="已联系",
        description="跟进结果：已联系/未接/有意向/已成交/已放弃"
    )
    next_follow_up_at: Optional[datetime] = Field(
        None, description="计划下次跟进时间（到期提醒依据）"
    )
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="本次跟进时间"
    )


# ==================== 6. RAG 文档模型 ====================
class Document(BaseModel):
    """
    RAG 知识库文档模型
    ====================
    字段说明：
      doc_id    : 文档唯一标识
      content   : 文档内容（文本）
      metadata  : 元数据（如来源文件名、上传时间等）
    
    使用场景：
      存储到 Chroma 向量数据库
      检索时返回给 LLM 作为上下文
    """
    doc_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="文档唯一标识"
    )
    content: str = Field(..., description="文档内容")
    metadata: dict = Field(
        default_factory=dict,               # 默认空字典
        description="文档元数据"
    )


# ==================== 7. Agent 状态模型（LangGraph 用） ====================
class AgentState(BaseModel):
    """
    LangGraph Agent 状态模型
    =========================
    LangGraph 工作流中，所有节点共享此状态对象
    
    字段说明：
      query         : 用户输入的原始问题
      user_id       : 当前用户 ID
      session_id    : 当前对话会话 ID（多对话框支持）
      user_info     : 用户信息（从 SQLite 查询）
      intent        : 意图识别结果（实时信息/业务咨询/常规咨询）
      retrieved_docs: RAG 检索到的文档列表
      search_result : 网络搜索结果（Web Search）
      response      : 生成助手回复
      history       : 对话历史消息列表
      messages      : 传给 LLM 的完整消息列表
      customer_info : 客户画像信息
    
    节点流转示例：
      用户输入 → intent识别 → 条件分支
        ├─ 实时信息 → web_search → 生成回复
        ├─ 业务咨询 → rag_law → 生成回复
        └─ 常规咨询 → 直接生成回复
    """
    query: str = Field(default="", description="用户原始问题")
    user_id: str = Field(default="", description="当前用户ID")
    session_id: str = Field(default="", description="当前对话会话ID")
    user_info: dict = Field(
        default_factory=dict,
        description="用户信息（从SQLite查）"
    )
    intent: str = Field(
        default="",
        description="意图分类：实时信息/业务咨询/常规咨询"
    )
    retrieved_docs: List[str] = Field(
        default_factory=list,
        description="RAG检索到的文档内容列表"
    )
    search_result: str = Field(
        default="",
        description="网络搜索结果文本"
    )
    response: str = Field(
        default="",
        description="生成的助手回复"
    )
    history: List[dict] = Field(
        default_factory=list,
        description="对话历史消息列表"
    )
    messages: List[dict] = Field(
        default_factory=list,
        description="传给LLM的完整消息列表"
    )
    customer_info: dict = Field(
        default_factory=dict,
        description="客户画像信息"
    )
