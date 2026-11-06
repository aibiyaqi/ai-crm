# 🤖 AI CRM 智能客户经营平台

> **企业级·轻量化·AI 驱动的智能客户经营系统**
>
> 基于 LangGraph 工作流 + 通义千问大模型 + RAG 知识库（BGE本地嵌入+jieba中文分词+BM25+Cross-Encoder重排序）+ Function Calling 的智能助手

---

## 📋 目录

1. [项目概述](#1-项目概述)
2. [系统架构](#2-系统架构)
3. [技术栈](#3-技术栈)
4. [核心技术模块](#4-核心技术模块)
5. [LangGraph 工作流详解](#5-langgraph-工作流详解)
6. [Function Calling 工具](#6-function-calling-工具)
7. [RAG 知识库（混合检索 + 重排序）](#7-rag-知识库混合检索--重排序)
8. [用户认证系统](#8-用户认证系统)
9. [API 接口文档](#9-api-接口文档)
10. [Streamlit 前端页面](#10-streamlit-前端页面)
11. [项目结构](#11-项目结构)
12. [环境搭建与部署](#12-环境搭建与部署)
13. [安全措施](#13-安全措施)
14. [性能评估与优化方向](#14-性能评估与优化方向)

---

## 1. 项目概述

### 1.1 项目背景

企业销售/客服人员每天需要处理大量客户咨询，问题类型多样，传统做法是为每个场景开发独立系统，维护成本高：

| 问题类型 | 示例 | 处理方式 |
|---------|------|---------|
| **制度/政策咨询** | "退换货流程是什么？" | RAG 知识库检索 |
| **客户数据查询** | "张三的公司叫什么？" | SQL Agent 查数据库 |
| **实时信息查询** | "今天AI板块有什么新闻？" | 联网搜索 |
| **常规问答** | "你们公司做什么的？" | LLM 直接回答 |

**核心价值**：用**一个 LangGraph 工作流**统一处理所有场景，AI 自动判断用户意图并路由到不同处理路径。

### 1.2 核心亮点

1. **LangGraph 多智能体编排**：7 节点 StateGraph 工作流 + 条件分支路由，可控性强
2. **两阶段意图识别**：规则匹配（零延迟）+ LLM 语义分类兜底（LRU 缓存，精准覆盖冷门查询）
3. **混合检索 RAG**：向量检索（BGE本地嵌入/text-embedding-v2备用）+ BM25（jieba中文分词）+ Cross-Encoder 重排序 + 自适应 TopK，四阶段精度递进
4. **Function Calling**：3 个内置工具（搜索/查库/保存客户），两轮调用实现动态交互
5. **用户认证系统**：密码/验证码双登录、注册、密码重置、管理员后台
6. **Streamlit 可视化前端**：3 用户页面 + 管理员后台（知识库仅管理员可见）+ 多对话会话管理
7. **FastAPI 接口**：RESTful API，支持外部系统集成（企业微信/钉钉等）
8. **安全稳定性提升**：启动环境校验 + 随机管理员密码 + JWT Bearer 认证 + CORS 白名单 + Rate Limiting 分级限流 + SQL 注入 6 层纵深防御 + 文件上传校验 + SQLite WAL 模式 + 熔断器重试机制，共 9 项改进

---

## 2. 系统架构

```
用户输入
    │
    ▼
┌──────────────────────────────────────────────────────────┐
│  L1  用户交互层                                           │
│  ─────────────────                                        │
│  app.py  (Streamlit 前端 · 5页面 · 知识库仅管理员可见)       │
│    ├─ 💬 对话页面    (多会话管理)                          │
│    ├─ 👥 客户信息    (查看/新增客户画像)                   │
│    ├─ 📚 知识库      (上传文档/检索测试)                   │
│    ├─ 📊 统计        (Token用量/系统信息)                  │
│    └─ 🛡️ 用户管理   (管理员后台·账户启用/禁用/日志)       │
│                                                          │
│  api.py  (FastAPI RESTful接口)                            │
│    ├─ GET  /api/status      (健康检查)                    │
│    ├─ POST /api/chat        (AI对话)                      │
│    ├─ POST /api/rag/upload  (上传文档)                    │
│    └─ POST /api/rag/text    (添加文本)                    │
│                                                          │
│  auth.py (用户认证系统)                                   │
│    ├─ 密码登录    → PBKDF2+SHA256+随机盐值 100000次迭代   │
│    ├─ 验证码登录  → 手机号+6位动态码 (5分钟有效期)        │
│    ├─ 注册/重置  → 密码强度校验 (8位+大小写+数字)         │
│    ├─ 登录锁定   → 连续5次失败→锁定15分钟                 │
│    └─ 角色隔离   → admin/user 路由保护                     │
└──────────────────────────┬───────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────┐
│  L2  工作流编排层 (agent.py)                              │
│  ─────────────────────────                                │
│  LangGraph StateGraph · 7个固定节点 · 不会死循环           │
│                                                          │
│    用户输入                                               │
│       │                                                   │
│       ▼                                                   │
│  ┌─ 节点1: load_context ─────────────────────┐            │
│  │  从 SQLite 查询用户信息 + 按session_id     │            │
│  │  加载最近对话历史                           │            │
│  └────────────────┬──────────────────────────┘            │
│                   ▼                                       │
│  ┌─ 节点2: classify_intent ──────────────────┐            │
│  │  两阶段意图识别                             │            │
│  │  阶段1: 规则匹配 (config.py关键词·0ms)       │            │
│  │  阶段2: LLM语义分类 (LRU缓存 max=200)       │            │
│  └────────────────┬──────────────────────────┘            │
│                   ▼                                       │
│  ┌─ 节点3: route_intent ─────────────────────┐            │
│  │  条件路由分发                               │            │
│  │   ├─ "web_search"    → 节点4a              │            │
│  │   ├─ "rag_retrieve"  → 节点4b              │            │
│  │   └─ "direct_llm"    → 节点4c              │            │
│  └────────────────┬──────────────────────────┘            │
│     ┌─────────────┼──────────────┐                        │
│     ▼             ▼              ▼                         │
│  ┌──────┐   ┌──────────┐   ┌────────┐                     │
│  │节点4a│   │  节点4b   │   │ 节点4c │                     │
│  │网络  │   │ RAG检索   │   │ 直接   │                     │
│  │搜索  │   │ 四阶段    │   │ LLM   │                     │
│  └──┬───┘   └────┬─────┘   └───┬────┘                     │
│     └────────────┼──────────────┘                          │
│                  ▼                                         │
│  ┌─ 节点5: generate_response ────────────────┐            │
│  │  Function Calling 两轮调用                 │            │
│  │  第一轮: llm.invoke(messages, tools)       │            │
│  │  ├─ 如有 tool_calls → 执行工具 → 追加结果  │            │
│  │  └─ 第二轮: llm.invoke(messages) 生成回答  │            │
│  └────────────────┬──────────────────────────┘            │
│                   ▼                                       │
│  ┌─ 节点6: save_history ─────────────────────┐            │
│  │  保存 user + assistant 消息到 SQLite       │            │
│  └────────────────┬──────────────────────────┘            │
│                   ▼                                       │
│              返回结果                                       │
└──────────────────────────┬───────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────┐
│  L3  业务逻辑执行层                                       │
│  ─────────────────                                        │
│                                                          │
│  tools.py (ToolRegistry 注册表模式)                        │
│  ├─ web_search             │ DuckDuckGo 搜索              │
│  │   └─ web_search.py      │ 免费搜索引擎                  │
│  ├─ sql_query              │ 自然语言→SQL→执行            │
│  │   └─ sql_agent.py       │ NL2SQL + SELECT安全过滤       │
│  └─ save_customer_profile │ 保存客户画像                  │
│      └─ models.py          │ CustomerProfile(Pydantic)    │
│         + memory_sqlite.py │ → SQLite 持久化              │
│                                                          │
│  rag_law.py (四阶段混合检索)                               │
│  ├─ 阶段1: 向量检索                                       │
│  │   嵌入 → Chroma.similarity_search_with_scores()        │
│  │   带相关性分数 → 阈值过滤 (≥0.5)                        │
│  ├─ 阶段2: BM25关键词检索                                 │
│  │   jieba.lcut() 精确模式分词 + 停用词过滤                │
│  │   BM25Okapi.get_scores() 打分排序                       │
│  ├─ 合并去重: 向量结果 + BM25结果 → 初检候选集             │
│  ├─ 阶段3: Cross-Encoder 重排序                           │
│  │   query+doc 拼接 → 交叉注意力 → 精确打分 → 降序排列     │
│  └─ 阶段4: 自适应 TopK 截断                               │
│      分数下降 >50% 时截断 · 动态范围 [1,6]                 │
│      质量好时多返回 · 质量差时少返回 · 宁缺毋滥            │
│                                                          │
│  config.py (意图规则配置)                                  │
│  ├─ web_search    : 30+关键词 (新闻/天气/股票/...)         │
│  └─ rag_retrieve  : 30+关键词 (法规/政策/条款/...)         │
│     min_match=1: 命中任意1个关键词即返回该意图              │
└──────────────────────────┬───────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────┐
│  L4  大模型服务层 (llms.py)                               │
│  ─────────────────────────                                 │
│                                                          │
│  QwenLLM (通义千问封装)                                   │
│  ├─ invoke()        → 通义千问 qwen-plus API 调用         │
│  ├─ 支持 Function Calling → tools 参数传递               │
│  ├─ 流式输出 → stream=True                                │
│  ├─ Token统计 → 输入/输出 Token 计数                      │
│  └─ 编码容错 → _sanitize_messages() latin-1 字符清理      │
│                                                          │
│  BGEEmbeddings (本地嵌入·推荐)                             │
│  ├─ BAAI/bge-small-zh-v1.5 (~33MB)                       │
│  ├─ 768维向量 · query前缀自动添加                          │
│  ├─ 懒加载 + 批量编码 (batch_size=32)                     │
│  └─ 优雅降级: 加载失败→DashScope API                      │
│                                                          │
│  DashScopeEmbeddings (在线嵌入·备用)                       │
│  ├─ text-embedding-v2 (1536维)                            │
│  └─ 零下载 · 通过阿里云DashScope API调用                  │
│                                                          │
│  CrossEncoderReranker (重排序)                            │
│  ├─ cross-encoder/ms-marco-MiniLM-L-6-v2 (~80MB)         │
│  ├─ 懒加载 · query+doc 交叉注意力评分                      │
│  └─ 失败降级: 回退朴素 TopK                               │
└──────────────────────────┬───────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────┐
│  L5  数据持久层                                           │
│  ─────────────                                            │
│                                                          │
│  memory_sqlite.py → SQLite (8张表)                       │
│  ├─ accounts              │ 账户认证 (account_id,         │
│  │                        │   username, password_hash,    │
│  │                        │   role, phone, is_active)     │
│  ├─ users                 │ 用户信息                      │
│  ├─ messages              │ 对话历史 (session隔离)        │
│  ├─ customer_profiles     │ 客户画像 (name/company/       │
│  │                        │   position/phone/tags)        │
│  ├─ chat_sessions         │ 对话会话 (多会话管理)         │
│  ├─ login_logs            │ 登录审计日志                  │
│  ├─ password_reset_tokens │ 密码重置令牌                  │
│  └─ session_tokens        │ 持久化登录 Token              │
│                                                          │
│  Chroma → 向量数据库                                      │
│  ├─ 库路径: utils/chroma_kb/                              │
│  ├─ 集合名: crm_kb                                       │
│  ├─ 嵌入维度: 768 (BGE) / 1536 (DashScope)               │
│  └─ 自动持久化: 0.4.24 版本自动写入磁盘                   │
│                                                          │
│  config.py → 全局配置 (10类)                              │
│  ├─ 模型参数: QWEN_MODEL/temperature/max_tokens          │
│  ├─ 检索参数: RETRIEVAL_TOP_K=4/CHUNK_SIZE=500           │
│  ├─ 嵌入配置: USE_LOCAL_EMBEDDING/LOCAL_EMBEDDING_MODEL  │
│  ├─ 重排序: RERANKER_ENABLED/RERANKER_MODEL/TOP_K        │
│  ├─ 自适应: ADAPTIVE_TOP_K/MIN/MAX/SCORE_THRESHOLD       │
│  ├─ 意图规则: INTENT_RULES (关键词+阈值)                  │
│  ├─ Token管控: MAX_HISTORY_ROUNDS=5                      │
│  └─ 安全配置: PASSWORD_* / LOGIN_* / VERIFICATION_*       │
└──────────────────────────┬───────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────┐
│  L6  基础设施 & 外部依赖                                   │
│  ─────────────────────────                                  │
│                                                          │
│  Docker容器化                                             │
│  ├─ Dockerfile           │ Python 3.10 运行环境           │
│  └─ docker-compose.yml   │ 服务编排 + 端口映射            │
│                                                          │
│  日志系统                                                 │
│  ├─ logger.py            │ 日志记录器                     │
│  └─ logfile/app.log      │ 日志文件 (DEBUG级别)           │
│                                                          │
│  环境配置                                                 │
│  ├─ .env                 │ QWEN_API_KEY 密钥              │
│  └─ .env.example         │ 模板文件                       │
│                                                          │
│  依赖管理                                                 │
│  ├─ requirements.txt     │ 全部依赖版本已锁定              │
│  └─ 核心依赖版本:                                       │
│     ├─ langgraph==0.2.9   │ 工作流引擎                    │
│     ├─ chromadb==0.4.24   │ 向量数据库 (0.5.0 不兼容)     │
│     ├─ dashscope          │ 通义千问/嵌入 API             │
│     ├─ sentence-transformers==3.0.1                      │
│     ├─ rank-bm25          │ BM25 关键词检索               │
│     ├─ jieba              │ 中文分词                      │
│     └─ streamlit          │ 前端框架                      │
│                                                          │
│  外部网络依赖                                             │
│  ├─ 阿里云 DashScope  → 通义千问API + 嵌入API            │
│  ├─ DuckDuckGo        → 免费网络搜索                      │
│  └─ HuggingFace       → 模型下载 (首次)                   │
└──────────────────────────────────────────────────────────┘

```

---

## 3. 技术栈

| 类别 | 技术 | 版本 | 说明 |
|------|------|------|------|
| **大模型** | 通义千问 qwen-plus (dashscope SDK) | 1.20.0 | 免费版够用，零成本 |
| **工作流编排** | LangGraph (StateGraph) | 0.2.9 | 预定义节点图，可控性强 |
| **LLM 框架** | LangChain | 0.2.16 | 文本分片、文档加载 |
| **向量数据库** | Chroma | 0.4.24 | ⚠️ 必须锁定，新版不兼容 |
| **嵌入模型** | BAAI/bge-small-zh-v1.5（本地 BGE，推荐）/ text-embedding-v2（DashScope API，备用） | ~33MB / — | ✅ 本地 BGE 嵌入（~33MB），零网络依赖；DashScope API 做备用降级 |
| **中文分词** | jieba | 0.42.1 | 🆕 BM25 中文分词，替换简单正则，精准切分中文 |
| **重排序模型** | cross-encoder/ms-marco-MiniLM-L-6-v2 | ~80MB | Cross-Encoder 精排，CPU 可跑，懒加载 |
| **关键词检索** | BM25 (rank-bm25) | 0.2.2 | 弥补向量检索关键词缺陷 |
| **Web 框架** | FastAPI + Uvicorn | 0.115.0 / 0.30.6 | 外部 API 接口 |
| **前端** | Streamlit | 1.37.0 | 4 页面 + 管理员后台 |
| **数据库** | MySQL 8.0（生产）+ SQLite（本地/回滚） | PyMySQL 1.1 / SQLAlchemy 2.0 | 统一 DAO 层：`DB_ENGINE=mysql` 即切库，业务 SQL 无需改动 |
| **缓存/协同** | Redis 7.2 | 5.0.8 | 意图缓存、会话令牌、登录失败计数、限流、延迟队列、分布式锁（不可用自动降级） |
| **搜索工具** | DuckDuckGo (免费) | 7.5.1 | 无需 API Key |
| **密码加密** | PBKDF2 + SHA256 + 随机盐值 | — | 不可逆加密，防泄露 |

---

## 4. 核心技术模块

| 模块 | 文件 | 功能 |
|------|------|------|
| **大模型封装** | [`llms.py`](AI_CRM/llms.py) | 通义千问调用，兼容 OpenAI 格式，支持流式 + Function Calling |
| **数据库管理** | [`memory_sqlite.py`](AI_CRM/memory_sqlite.py) | 12 张表 CRUD；SQLite（WAL）/ MySQL（连接池）双实现，经 `db.py` 统一切换 |
| **数据访问层** | [`db.py`](AI_CRM/db.py) | Database 抽象 + 方言适配（`?`→`%s`、`ON CONFLICT`→upsert）+ 连接池 + 兼容 Row + 重试 |
| **Redis 基础设施** | [`redis_client.py`](AI_CRM/redis_client.py) | 连接池 / 熔断 / 统一降级 / 分布式锁 / 观测（`/api/status` 可见） |
| **RAG 知识库** | [`rag_law.py`](AI_CRM/rag_law.py) | Chroma + BM25 混合检索 + Cross-Encoder 重排序 |
| **SQL Agent** | [`sql_agent.py`](AI_CRM/sql_agent.py) | 自然语言→SQL→6层防注入校验→执行→结果转自然语言 |
| **搜索工具** | [`web_search.py`](AI_CRM/web_search.py) | DuckDuckGo 免费搜索，支持格式化返回 |
| **工具注册表** | [`tools.py`](AI_CRM/tools.py) | 7 个工具统一注册/执行/定义（含跟进三件套与异常筛查） |
| **工作流编排** | [`agent.py`](AI_CRM/agent.py) | LangGraph 7 节点 StateGraph + 条件路由 |
| **用户认证** | [`auth.py`](AI_CRM/auth.py) | 密码/验证码登录、注册、重置、管理员初始化 + secrets 随机密码 |
| **配置管理** | [`config.py`](AI_CRM/config.py) | 10 类全局配置 + validate_env_or_exit() 启动校验 |
| **日志工具** | [`logger.py`](AI_CRM/logger.py) | 控制台 + 文件双输出 |
| **数据模型** | [`models.py`](AI_CRM/models.py) | Pydantic 模型：Account/User/Message/CustomerProfile/AgentState |
| **FastAPI 接口** | [`api.py`](AI_CRM/api.py) | 5 个 API 端点 + JWT Bearer 认证 + CORS 白名单 + Rate Limiting |
| **Streamlit 前端** | [`app.py`](AI_CRM/app.py) | 对话 / 客户 / 跟进 / 告警 / 统计 5 页面 + 管理员后台 + 多对话管理 |
| **告警中心** | [`scheduler.py`](AI_CRM/scheduler.py) | 事件驱动 + 分层跑批产出告警；前端支持「立即巡检 / 标记已读 / 未读角标」 |
| **熔断器+重试** | [`utils/circuit_breaker.py`](AI_CRM/utils/circuit_breaker.py) | Circuit Breaker 状态机 + 指数退避重试（含随机抖动） |

---

## 5. LangGraph 工作流详解

### 5.1 节点设计（7 节点）

```
load_context (节点1)
  → classify_intent (节点2) — 纯规则关键词匹配，零 LLM 调用
    → route_intent (节点3) — 条件分支
      ├── web_search (节点4a) — DuckDuckGo 搜索
      ├── rag_retrieve (节点4b) — Chroma + BM25 + Reranker 混合检索
      └── direct_llm (节点4c) — 直接回答（默认）
        → generate_response (节点5) — 整合上下文 + Function Calling 两轮调用
          → save_history (节点6) — 保存对话到 SQLite
            → END — 返回结果
```

### 5.2 意图识别（规则匹配 + LLM 语义兜底）

对比传统方案和本项目的优化演进：

| 对比项 | 纯 LLM 分类（传统方案） | 纯规则匹配（原方案） | 规则 + LLM 语义兜底（**当前方案 ✅**） |
|--------|----------------------|-------------------|-----------------------------------|
| 耗时 | ~1-2 秒（每次调 LLM） | **0 毫秒**（关键词匹配） | 常见查询 0ms，冷门查询 ~1-2s |
| 成本 | 每次消耗 Token | **零成本** | 常见查询零成本，冷门查询少量 Token |
| 覆盖场景 | 100%（LLM 兜底） | 常见场景 90%+ | **100%**（规则未命中 → LLM 语义分类） |
| 可维护性 | 需改提示词 | 改 [`config.py`](AI_CRM/config.py:76) 关键词列表即可 | 关键词列表 + LLM 提示词双维护 |
| 代码位置 | — | [`agent.py:182-211`](AI_CRM/agent.py:182) | [`agent.py:214-331`](AI_CRM/agent.py:214) |

**两阶段策略**（[`agent.py:290-331`](AI_CRM/agent.py:290)）：

```
阶段1（规则匹配，零成本）：
  遍历 INTENT_RULES 关键词列表，命中 ≥ min_match 即返回对应意图
  → 命中 → 直接返回（0 毫秒）
  → 未命中 → 进入阶段2

阶段2（LLM 语义分类，带 LRU 缓存）：
  计算 query 的 MD5 哈希 → 查缓存
  → 缓存命中 → 直接返回（0 毫秒）
  → 缓存未命中 → 调 LLM 语义分类 → 存入缓存（max 200 条）→ 返回
```

**配置示例**（[`config.py:76-105`](AI_CRM/config.py:76)）：

```python
INTENT_RULES = {
    "web_search": {
        "keywords": ["新闻", "天气", "股票", "最新", "今天", "实时", ...],
        "min_match": 1,
    },
    "rag_retrieve": {
        "keywords": ["法规", "政策", "条款", "什么是", "如何", ...],
        "min_match": 1,
    },
}
```

**LLM 语义分类提示词**（[`agent.py:223-256`](AI_CRM/agent.py:223)）：明确限定三个合法意图（`web_search` / `rag_retrieve` / `direct_llm`），temperature=0.1 确保确定性输出，兜底返回 `direct_llm`。

> **效果**：常见查询（含关键词）零延迟通过规则匹配；冷门查询（不包含关键词但语义上需要检索/搜索）由 LLM 精准兜底，同时 LRU 缓存避免重复调用。

### 5.3 Function Calling 两轮调用

[`agent.py:300-398`](AI_CRM/agent.py:300) 的 `_node_generate_response` 中实现：

```
第一轮：LLM 收到问题 + 工具定义 → 决定是否调工具
  ├── 不调 → 直接生成回答
  └── 调工具 → 返回 tool_calls
                ↓
         执行工具（web_search / sql_query / save_customer_profile）
                ↓
         工具结果追加到 messages
                ↓
第二轮：LLM 看到工具结果 → 生成最终回答
```

---

## 6. Function Calling 工具

### 6.1 工具总览

| 工具 | 功能 | 触发场景 | 实现 |
|------|------|---------|------|
| **`web_search`** | 搜索互联网获取实时信息 | "今天AI有什么新闻？" | DuckDuckGo 免费 API |
| **`sql_query`** | 自然语言查询客户数据库 | "查询所有高价值客户" | SQLAgent 转 SQL 执行 |
| **`save_customer_profile`** | 保存客户画像到数据库 | "我的客户叫张三，在腾讯工作" | 写入 customer_profiles 表 |
| **`screen_anomaly_customers`** | 异常客户筛查（高意向未跟进/流失风险/资料不全） | "哪些客户该跟进了？" | 规则引擎 + 画像字段 |
| **`record_follow_up`** | 记录一次跟进（方式/内容/结果/下次时间） | "今天给张一诺打了电话，有意向" | 写 follow_ups + 回写画像最近跟进时间 |
| **`list_due_follow_ups`** | 待跟进提醒清单（到期 + 超期） | "今天该跟谁？" | 按 next_follow_up_at 查询 |
| **`follow_up_stats`** | 跟进转化统计 | "这个月跟进了多少、转化率多少？" | 按结果口径聚合 |

### 6.2 工具注册机制

所有工具在 [`ToolRegistry._register_builtin_tools()`](AI_CRM/tools.py:47) 中统一注册，通过 [`execute()`](AI_CRM/tools.py:96) 统一调度，try/except 兜底错误。

### 6.3 安全限制

- `sql_query`：只允许 SELECT，禁止 DROP/DELETE/UPDATE/INSERT/ALTER/TRUNCATE
- `save_customer_profile`：只能保存，无删除功能
- 错误信息返回给 LLM，LLM 向用户友好说明

---

## 7. RAG 知识库（混合检索 + 重排序）

### 7.1 入库流程

```
上传文档 (.txt / .md)
  → RecursiveCharacterTextSplitter 切分 (chunk_size=500, chunk_overlap=100)
  → 嵌入模型向量化：
      ├── BAAI/bge-small-zh-v1.5 本地嵌入（推荐 ✅，~33MB，零网络依赖）
      └── 通义千问 text-embedding-v2 API 备用（DashScope 在线 API）
  → 存入 Chroma 向量数据库（自动持久化，无需 .persist()）
```

### 7.2 检索流程（四阶段）

```
用户问题
  ├──→ 向量检索（语义相似度） → TopK 结果
  │     嵌入模型：BGE 本地（默认） / text-embedding-v2（备用）
  └──→ BM25 检索（关键词匹配，jieba 中文分词） → TopK 结果
       ↓
合并 → 去重 → 初检候选集
       ↓
相关性阈值过滤（RETRIEVAL_SCORE_THRESHOLD=0.5）
  → 低于阈值的结果被剔除（至少保留 1 条兜底）
       ↓
Cross-Encoder 重排序（精排）
  query 与每个 doc 拼接 → 交叉注意力精确打分 → 降序排列
       ↓
自适应 TopK 截断（adaptive_search）
  → 动态分析分数分布，相邻分数下降 >50% 时截断
  → 截断范围：[ADAPTIVE_TOP_K_MIN=1, ADAPTIVE_TOP_K_MAX=6]
       ↓
返回最终结果（1-6 条）
```

**为什么混合？** 纯向量检索对专业术语/缩写词（如"CRM"）效果不好，BM25 补上。

**为什么用 jieba 分词？** BM25 原使用简单正则（`re.findall(r'\w+', text.lower())`）对中文无效——"退换货流程"被识别为一个完整 token，无法与"退货""换货""流程"等子词匹配。jieba 精准切分后，`_tokenize` 返回 `["退换货", "流程"]` 等独立词元，显著提升中文检索命中率（[`rag_law.py:816-860`](AI_CRM/rag_law.py:816)）。

**为什么加重排序？** Cross-Encoder 比向量余弦相似度更精确（约 +10~15% NDCG）。

**为什么加相关性阈值和自适应 TopK？** 固定 TopK 返回固定数量的结果，质量差时仍返回低相关文档；阈值过滤（0.5）剔除低质量结果，自适应截断动态调整返回数量，保证"宁缺毋滥"（[`rag_law.py:680-750`](AI_CRM/rag_law.py:680)）。

**模型选择**：
- **嵌入模型**：`BAAI/bge-small-zh-v1.5`（~33MB，中文优化），首次加载约 2-3 秒，后续调用毫秒级。备用使用 DashScope `text-embedding-v2` API（在线，零下载）。
- **重排序模型**：[`cross-encoder/ms-marco-MiniLM-L-6-v2`](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2)（~80MB），CPU 可跑，懒加载。

> ⚠️ **国内用户注意**：BGE 嵌入模型从 HuggingFace 下载（~33MB），重排序模型也从 HuggingFace 下载（~80MB）。
> 如无 VPN，请在启动前设置镜像：
> - **Linux/Mac**：`export HF_ENDPOINT=https://hf-mirror.com`
> - **Windows CMD**：`set HF_ENDPOINT=https://hf-mirror.com`
> - **PowerShell**：`$env:HF_ENDPOINT="https://hf-mirror.com"`
> - 或在 [`docker-compose.yml`](docker-compose.yml) 中取消 `HF_ENDPOINT` 环境变量的注释
> - BGE 备选：如不想下载，设置 `USE_LOCAL_EMBEDDING=False` 使用 DashScope 在线 API

**安全机制**：
- 懒加载：BGE 嵌入和重排序模型均在首次检索时加载
- 优雅降级：BGE 加载失败 → 自动回退 DashScope API；重排序失败 → 自动回退朴素排序
- 可开关：设置 [`RERANKER_ENABLED = False`](AI_CRM/config.py:112) 可跳过重排序；设置 [`USE_LOCAL_EMBEDDING = False`](AI_CRM/config.py:113) 可切换在线 API

---

## 8. 用户认证系统

### 8.1 功能列表

| 功能 | 说明 | 代码位置 |
|------|------|---------|
| **密码登录** | 用户名/手机号 + 密码 | [`auth.py:124-168`](AI_CRM/auth.py:124) |
| **验证码登录** | 手机号 + 6 位验证码 | [`auth.py:171-208`](AI_CRM/auth.py:171) |
| **注册账号** | 用户名 + 手机号 + 密码 + 可选邮箱 | [`auth.py:211-262`](AI_CRM/auth.py:211) |
| **忘记密码** | 手机验证码验证 → 重置密码 | [`auth.py:265-318`](AI_CRM/auth.py:265) |
| **管理员后台** | 用户管理、登录审计、账户禁用 | [`app.py:1224-1375`](AI_CRM/app.py:1224) |

### 8.2 安全设计

| 措施 | 说明 |
|------|------|
| **密码不可逆存储** | PBKDF2 + SHA256 + 32 字节随机盐值，100000 次迭代 |
| **登录失败锁定** | 连续 5 次失败，锁定 15 分钟 |
| **验证码有效期** | 5 分钟过期，单次使用后失效 |
| **密码强度校验** | 至少 8 位，包含大小写字母和数字 |
| **角色权限隔离** | admin/user 路由保护，普通用户不可访问管理员页面 |
| **持久化 Token** | UUID Token，7 天有效期，支持 URL 自动恢复登录 |
| **API Key 启动校验** | 系统启动时验证 QWEN_API_KEY 是否存在，缺失则直接退出（[`config.py:28-57`](AI_CRM/config.py:28)） |
| **随机管理员密码** | 首次初始化使用 secrets 模块生成 16 位随机密码（混合大小写+数字+特殊字符），不再使用硬编码默认密码（[`auth.py:242-252`](AI_CRM/auth.py:242)） |
| **JWT Bearer 认证** | FastAPI 接口统一使用 HTTPBearer 令牌认证，`get_current_user()` 依赖注入验证 Token 有效性（[`api.py:93-149`](AI_CRM/api.py:93)） |
| **CORS 白名单** | 允许来源通过环境变量 `CORS_ORIGINS` 配置（默认 `http://localhost:8501`），禁止通配符 `*`（[`api.py:252-260`](AI_CRM/api.py:252)） |
| **Rate Limiting 分级限流** | 4 级限流策略：对话 30 次/分钟、上传 10 次/小时、文本添加 30 次/小时、状态查询 30 次/分钟（[`api.py:264-280`](AI_CRM/api.py:264)） |

### 8.3 数据库表结构

| 表名 | 用途 | 关键字段 |
|------|------|---------|
| `accounts` | 账户认证 | account_id, username, password_hash, role, phone, is_active |
| `users` | 用户信息 | user_id, name, phone |
| `messages` | 对话历史 | message_id, session_id, user_id, role, content |
| `customer_profiles` | 客户画像 | customer_id, name, company, position, tags |
| `chat_sessions` | 对话会话 | session_id, account_id, title |
| `login_logs` | 登录审计 | log_id, account_id, username, action, ip_address |
| `password_reset_tokens` | 密码重置 | token_id, account_id, code, expires_at |
| `session_tokens` | 持久化登录 | token_id, account_id, token, expires_at |

---

## 9. API 接口文档

### 9.1 接口总览

| 端点 | 方法 | 说明 | 需要 LLM | 认证 | 限流 |
|------|------|------|---------|------|------|
| `/` | GET | API 欢迎页 | ❌ | ❌ | ❌ |
| `/api/status` | GET | 系统健康检查 | ❌ | ❌ | 30次/分钟 |
| `/api/chat` | POST | AI 对话接口（核心） | ✅ | ✅ JWT Bearer | 30次/分钟 |
| `/api/rag/upload` | POST | 上传文档到知识库 | ❌ | ✅ JWT Bearer | 10次/小时 |
| `/api/rag/text` | POST | 添加文本到知识库 | ❌ | ✅ JWT Bearer | 30次/小时 |

### 9.2 核心接口

#### POST /api/chat — AI 对话

```json
// 请求
{
  "query": "我们公司有哪些高价值客户？",
  "user_id": "api_user",
  "customer_id": "c001"
}

// 响应
{
  "success": true,
  "response": "根据数据库查询，目前标签为'高价值'的客户有：张三（腾讯）、李四（阿里）...",
  "error": ""
}
```

#### POST /api/rag/upload — 上传文档

```
请求：multipart/form-data (file: 制度文档.txt)
响应：{ "success": true, "chunk_count": 12, "error": "" }
```

### 9.3 启动 API

```bash
# 方式1：直接运行
python AI_CRM/api.py

# 方式2：uvicorn 开发模式（热重载）
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

---

## 10. Streamlit 前端页面

### 10.1 页面总览

| 页面 | 功能 | 访问权限 |
|------|------|---------|
| 💬 **对话** | AI 聊天主界面，多对话会话管理 | 所有用户 |
| 👥 **客户信息** | 查看/新增客户画像，自然语言查客户 | 所有用户 |
| 📚 **知识库** | 上传文档/检索测试，查看知识库状态 | 仅管理员 |
| 📊 **统计** | Token 用量统计、系统信息 | 所有用户 |
| 🛡️ **用户管理** | 账户列表、启用/禁用、登录日志、改密码 | 仅管理员 |

### 10.2 运行方式

```bash
streamlit run AI_CRM/app.py
```

### 10.3 关键特性

- **多对话会话管理**：新建/切换/删除对话，消息按会话隔离
- **联网搜索开关**：可随时开启/关闭 AI 的联网搜索能力
- **自动恢复登录**：URL 携带 Token 可自动恢复登录状态
- **管理员后台**：账户管理 + 登录日志审计
- **首次运行随机管理员密码**：secrets 模块生成 16 位随机密码（混合大小写+数字+特殊字符），自动输出到控制台

---

## 11. 项目结构

```
AI_CRM/                          # 项目根目录
├── app.py                       # Streamlit 前端界面（4页面 + 管理员后台 + 认证）
├── agent.py                     # LangGraph 工作流编排（7节点 + 条件路由）
├── api.py                       # FastAPI 外部 API 接口（5端点）
├── auth.py                      # 用户认证模块（登录/注册/重置/管理员）
├── tools.py                     # Function Calling 工具注册表（7 工具）
├── llms.py                      # 通义千问 LLM 封装（流式 + FC + Token统计）
├── rag_law.py                   # RAG 知识库（Chroma + BM25 + Cross-Encoder）
├── sql_agent.py                 # SQL 自然语言查询 Agent
├── web_search.py                # DuckDuckGo 搜索工具
├── memory_sqlite.py             # SQLite 数据库管理（8张表 CRUD）
├── models.py                    # Pydantic 数据模型（Account/User/Message等）
├── config.py                    # 全局配置（10类参数集中管理）
├── logger.py                    # 日志配置（控制台 + 文件双输出）
├── test.py                      # 集成测试脚本
├── test_rag.py                  # RAG 模块测试
├── 排查.md                      # Bug 排查与修复记录（10个Bug）
│
├── crm.db                      # SQLite 数据库文件（自动生成）
├── .env                        # API Key 配置（已加入 .gitignore）
├── .env.example                # 环境变量模板（不含真实 Key）
│
├── logfile/                    # 日志目录
│   └── app.log                 # 运行日志文件
├── utils/
│   ├── circuit_breaker.py      # 熔断器 + 指数退避重试（3个预置实例）
│   └── chroma_kb/              # 向量数据库持久化目录（Chroma）
│
├── Dockerfile                  # 🆕 Docker 镜像构建文件
├── docker-compose.yml          # 🆕 Docker Compose 编排（一键启动）
├── .dockerignore               # 🆕 Docker 构建上下文排除
│
plans/                           # 项目规划文档
├── requirements.txt            # 依赖清单（版本已锁定）
├── 0环境学习防不兼容bug.md      # 环境搭建指南
└── 3独立环境搭建.md             # 独立环境部署文档
```

---

## 12. 环境搭建与部署

### 12.1 方式一：Docker 部署（推荐 🚀）

**前提条件**：安装 [Docker](https://docs.docker.com/engine/install/) + [Docker Compose](https://docs.docker.com/compose/install/)

```bash
# 1. 克隆项目
git clone <项目地址>
cd <仓库根目录>          # docker-compose.yml 所在目录（不是 AI_CRM/）

# 2. 配置 API Key（模板在 AI_CRM/.env.example）
cp AI_CRM/.env.example .env
# 编辑 .env，填入 QWEN_API_KEY（获取地址见下文）

# 3. 一键启动（MySQL + Redis + Streamlit + FastAPI）
docker compose up -d --build
# 国内网络可先单独构建并指定 pip 镜像源（首次构建要装 torch 等大依赖，较慢）：
# docker compose build --build-arg PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/

# 4. 验收（API/MySQL/Redis/Streamlit/鉴权 一次检查）
python scripts/verify_stack.py

# 5. 访问
# Streamlit 前端: http://localhost:8501
# FastAPI 接口:   http://localhost:8000

# 6. 查看日志 / 停止
docker compose logs -f
docker compose down
```

**Docker 架构说明**：

| 服务 | 容器名 | 端口 | 命令 |
|------|--------|------|------|
| **Streamlit 前端** | `ai-crm-streamlit` | `8501` | `streamlit run AI_CRM/app.py` |
| **FastAPI 接口** | `ai-crm-api` | `8000` | `python AI_CRM/api.py` |

**数据持久化**：Docker 容器自动将数据挂载到宿主机 `./data/` 目录，重启不丢失。

**国内用户**：如无 VPN，在 [`docker-compose.yml`](docker-compose.yml) 中取消 `HF_ENDPOINT` 环境变量的注释即可使用 HuggingFace 镜像。

### 12.2 方式二：传统虚拟环境部署

```bash
# 1. 克隆项目
git clone <项目地址>
cd AI_CRM

# 2. 创建虚拟环境（推荐 Python 3.10 - 3.12）
python -m venv venv

# 3. 激活环境
# Windows:
venv\Scripts\activate
# Linux/Mac:
# source venv/bin/activate

# 4. 安装依赖
pip install -r plans/requirements.txt

# 5. 配置 API Key
cp .env.example .env
# 编辑 .env，填入 QWEN_API_KEY
# 获取地址：https://help.aliyun.com/zh/model-studio/

# 6. 启动 Streamlit 前端
streamlit run AI_CRM/app.py

# 7. 启动 FastAPI 接口（可选）
python AI_CRM/api.py
```

### 12.3 环境要求

- Python ≥ 3.10（推荐 3.10 - 3.12）
- 通义千问 API Key（免费申请，[点此获取](https://help.aliyun.com/zh/model-studio/)）
- 网络环境（需访问 dashscope API 和 DuckDuckGo）
- ✅ 嵌入模型默认使用本地 BGE（`BAAI/bge-small-zh-v1.5`，~33MB），首次运行自动下载，后续离线使用
- ⚠️ 如不想下载模型，设置 [`USE_LOCAL_EMBEDDING = False`](AI_CRM/config.py:113) 切换为 DashScope text-embedding-v2 在线 API
- ⚠️ 重排序模型（Cross-Encoder ~80MB）可选，国内无 VPN 需设置 `HF_ENDPOINT=https://hf-mirror.com`
- 🆕 新增 jieba 中文分词依赖

### 12.4 依赖清单

[`plans/requirements.txt`](plans/requirements.txt) 已锁定全部兼容版本，关键锁定项：

| 包名 | 版本 | 说明 |
|------|------|------|
| `langgraph` | 0.2.9 | ⚠️ 必须锁定，1.x 返回格式不兼容 |
| `chromadb` | 0.4.24 | ⚠️ 必须 <0.5.0，否则 langchain-chroma 不兼容 |
| `langchain-text-splitters` | 0.2.2 | langchain 0.2.x 拆分包，必须额外安装 |
| `dashscope` | 1.20.0 | 通义千问官方 SDK（LLM 使用） |
| `sentence-transformers` | 3.0.1 | BGE 嵌入模型 + Cross-Encoder 重排序（约33MB+80MB，可选） |
| `jieba` | 0.42.1 | 🆕 BM25 中文分词，替换简单正则 |
| `slowapi` | 0.1.9 | 🆕 Rate Limiting 分级限流（4 级策略） |

---

## 12.5 数据库与缓存（MySQL / Redis）

### 12.5.1 数据库：一套 DAO，两种引擎

```
业务代码（158 处 SQL，写法保持 SQLite 风格）
        │
        ▼
db.get_database()  ← 只看 DB_ENGINE
   ├─ SQLiteDatabase  本地开发 / 回滚（WAL + busy_timeout）
   └─ MySQLDatabase   生产（DBUtils 连接池 + 方言适配 + 兼容 Row + 重试）
```

- **切换方式**：`.env` 里 `DB_ENGINE=sqlite` ↔ `DB_ENGINE=mysql`，业务代码零改动。
- **方言差异收敛在 `db.py`**：`?` → `%s`、字面量 `%` → `%%`、
  `ON CONFLICT ... excluded.x` → `AS new ON DUPLICATE KEY UPDATE ... new.x`。
- **三套库统一到 MySQL**：业务库（12 表）、评测库（4 表）、LangGraph 检查点库（1 表）。
- **建表脚本**：[`AI_CRM/sql/mysql/001_init.sql`](AI_CRM/sql/mysql/001_init.sql)、
  [`002_eval_store.sql`](AI_CRM/sql/mysql/002_eval_store.sql)、
  [`003_checkpoint.sql`](AI_CRM/sql/mysql/003_checkpoint.sql)（幂等，可重复执行）。

**从 SQLite 搬迁历史数据**（迁移前请先备份 `.db` 文件）：

```bash
# 只看源库行数，不写库
python scripts/migrate_sqlite_to_mysql.py --all --dry-run

# 正式迁移（--truncate 会先清空目标表；默认遇到非空目标库直接拒绝）
python scripts/migrate_sqlite_to_mysql.py --all --truncate
```

脚本会逐表校验 **行数 + 主键集合**，任何一张表不一致都会以非 0 退出码结束。

### 12.5.2 Redis：加速层与协同层（不是事实源）

原则：**MySQL 存真相，Redis 只做加速与协同；Redis 挂了只降级，不阻断。**

| 场景 | 键 | 降级行为 |
|---|---|---|
| 意图缓存 | `crm:intent:{md5}` | 进程内 LRU |
| 会话令牌 | `crm:session:{token}` | 回源 `session_tokens` 表 |
| 登录失败计数 | `crm:loginfail:{username}` | 回源 `login_logs` 计数 |
| 接口限流 | `slowapi` 存储 | 退回进程内限流（多 worker 阈值放大） |
| 延迟消息队列 | `crm:delay:follow_up`（ZSET） | 回源 `delayed_messages` 表轮询 |
| 跑批去重锁 | `crm:lock:scan:*` | 退化为单实例模式 |
| 告警去重 | `crm:alert:dedup:*` | 数据库唯一键兜底 |

- 统一连接池 / 超时 1s / 熔断（连续 5 次失败冷却 30s）/ 观测：`GET /api/status` 的 `redis` 字段。
- `REDIS_ENABLED=0` 可整体关闭 Redis（等价于全降级，便于排障）。

### 12.5.3 相关自测与压测

```bash
# 生成面试演示数据（8 个客户 / 10 条跟进 / 3 条告警，可重复执行不翻倍）
python scripts/seed_demo_data.py

python AI_CRM/test_db_mysql_dialect.py    # 方言适配 / 建表脚本 / SQL 护栏（不需要数据库）
python AI_CRM/test_redis_scenarios.py     # Redis 六场景（需要 fakeredis）
DB_ENGINE=mysql python AI_CRM/test_mysql_integration.py   # 真库端到端（没有 MySQL 时自动 SKIP）

# 栈起来后一键验收（API / MySQL / Redis / Streamlit / 鉴权）
python scripts/verify_stack.py

# 100 并发只读压测（不触发 LLM）：先预置 token，再跑 Locust
python scripts/provision_loadtest_tokens.py --count 100 --out .dev/loadtest_tokens.json
LOADTEST_TOKENS_FILE=.dev/loadtest_tokens.json \
  locust -f AI_CRM/locustfile.py CRMReadOnlyUser --headless -u 100 -r 20 -t 30s \
         --host http://localhost:8000
```

> token 预置必须放在 Locust 进程外：gevent 会 monkey-patch socket，
> 压测进程里直连 MySQL 容易 `WinError 10053`。

---

## 13. 安全措施

### 13.1 整体安全架构

系统共实施了 **9 项安全稳定性改进**，覆盖启动态、认证层、API 层、数据层、业务层：

| 层级 | 措施 | 状态 | 代码位置 |
|------|------|------|---------|
| **🟢 启动态** | QWEN_API_KEY 环境变量校验，缺失即退出 | ✅ 已实施 | [`config.py:28-57`](AI_CRM/config.py:28) |
| **🔐 认证层** | secrets 模块随机 16 位密码替代硬编码 admin123 | ✅ 已实施 | [`auth.py:242-252`](AI_CRM/auth.py:242) |
| **🔐 认证层** | JWT Bearer Token 认证（HTTPBearer + Depends） | ✅ 已实施 | [`api.py:93-149`](AI_CRM/api.py:93) |
| **🌐 API 层** | CORS 白名单控制，环境变量 CORS_ORIGINS 配置 | ✅ 已实施 | [`api.py:252-260`](AI_CRM/api.py:252) |
| **🌐 API 层** | Rate Limiting 分级限流（4 级策略） | ✅ 已实施 | [`api.py:264-280`](AI_CRM/api.py:264) |
| **📝 业务层** | SQL 注入 6 层纵深防御 | ✅ 已实施 | [`sql_agent.py:47-100`](AI_CRM/sql_agent.py:47) |
| **📝 业务层** | 文件上传校验（扩展名白名单 + 10MB 限制 + UUID 重命名） | ✅ 已实施 | [`api.py:406-467`](AI_CRM/api.py:406) |
| **🗄️ 数据层** | SQLite WAL 模式 + 连接优化（busy_timeout=5000） | ✅ 已实施 | [`memory_sqlite.py:65-85`](AI_CRM/memory_sqlite.py:65) |
| **🛡️ 数据层** | Circuit Breaker 熔断器 + 指数退避重试（含随机抖动） | ✅ 已实施 | [`utils/circuit_breaker.py`](AI_CRM/utils/circuit_breaker.py) |

### 13.2 SQL 注入 6 层纵深防御

[`sql_agent.py:47-100`](AI_CRM/sql_agent.py:47) 的 `_validate_sql_safe()` 函数实现多层校验：

| 层 | 校验规则 | 说明 |
|----|---------|------|
| **L1** | 仅允许 SELECT | 检查 SQL 是否以 SELECT 开头，禁止所有写操作 |
| **L2** | 关键字黑名单 | 12 个危险关键字：DROP, ALTER, DELETE, INSERT, UPDATE, TRUNCATE, CREATE, REPLACE, ATTACH, DETACH, REINDEX, EXECUTE |
| **L3** | 多语句拦截 | 检测分号 `;` 防止拼接多条 SQL（如 `SELECT 1; DROP TABLE`） |
| **L4** | 注释符号检测 | 拦截 `--` 和 `/* */` 注释，防止绕过关键字检查 |
| **L5** | 系统表保护 | 过滤 `sqlite_` 前缀的表，防止读取 `sqlite_master` 等系统表 |
| **L6** | 字符串字面量剥离 | 去除 SQL 字符串中的引号内容后再次校验，防止 `SEL'ECT'` 绕过 |

### 13.3 文件上传安全

[`api.py:406-467`](AI_CRM/api.py:406) 的 `upload_document()` 中实现：

| 措施 | 配置 |
|------|------|
| **扩展名白名单** | 仅允许 `.txt` / `.md` / `.pdf` / `.docx` |
| **文件大小限制** | 最大 10MB（`MAX_UPLOAD_SIZE = 10 * 1024 * 1024`） |
| **文件名安全** | 使用 UUID 重命名文件，防止路径遍历攻击 |
| **异常处理** | try/except 兜底所有 IO 错误，返回友好错误提示 |

### 13.4 Token 管控（省钱防溢出）

| 参数 | 值 | 说明 |
|------|-----|------|
| 检索召回 | Top3-5 | 控制输入 Token |
| temperature | 0.3 | 保守回答，适合企业场景 |
| 对话历史上限 | 最近 5 轮 | 控制累积 Token |
| 模型 | qwen-plus | 免费版够用 |

---

## 14. 性能评估与优化方向

### 14.1 请求耗时分析

| 环节 | 耗时 | 说明 |
|------|------|------|
| LLM API 调用 | 2-5 秒 | 主要瓶颈（占 80% 时间） |
| 向量检索 | 50-200 ms | Chroma + BM25 |
| SQL 查询 | 10-50 ms | SQLite 本地查询 |
| 工具执行 | 100-500 ms | 网络搜索等 |
| **总耗时** | **3-8 秒** | 取决于 LLM API |

### 14.2 并发能力

| 场景 | 单次耗时 | 单线程 QPS |
|------|---------|-----------|
| 纯 LLM 回答 | 3s | 0.33 |
| RAG 检索回答 | 4s | 0.25 |
| 含工具调用 | 8s | 0.125 |

**结论**：企业 50 人同时使用（每人每小时 10 次 ≈ 0.14 QPS）✅ 够用，瓶颈在 LLM API。

### 14.3 后续优化方向

1. **缓存机制**：相同问题缓存 LLM 回答，减少 API 调用
2. **流式输出**：首字 0.5s 出，用户体验提升
3. **父子分块**：小 chunk 检索 + 大 chunk 生成，检索命中率 +15~20%
4. **异步处理**：FastAPI async + 异步 LLM 调用，并发翻倍
5. **查询重写**：多轮对话中指代消解后检索
6. **已实施 — SQLite WAL 模式**：写入并发提升 5-10 倍，避免锁等待
7. **已实施 — 熔断器重试**：外部服务故障时自动隔离，指数退避 + 随机抖动防止雪崩

---

## 🐛 踩坑记录

详见 [`AI_CRM/排查.md`](AI_CRM/排查.md)（10 个 Bug 完整记录 + RAG 优化记录），核心踩坑：

| # | Bug | 根因 | 修复 |
|---|-----|------|------|
| 1 | API Key 未传递 | dashscope 读 `DASHSCOPE_API_KEY` 但 .env 用 `QWEN_API_KEY` | 显式 `dashscope.api_key =` 赋值 |
| 2 | `KeyError: tool_calls` | dashscope 代理属性访问异常 | try/except 兜底 |
| 3 | ✅ HuggingFace 下载失败（已根治） | 国内网络无法连接 huggingface.co | 嵌入已改为本地 BGE + DashScope API 双方案 |
| 4 | Chroma 无 `.persist()` | 0.4.24 已移除该方法 | 直接删除调用（自动持久化） |
| 5 | LangGraph 返回 `dict` 非对象 | 1.x 的 `invoke()` 返回普通 dict | 改用 `dict.get()` 兼容 |
| 6 | Function Calling 未生效 | 没传 `tools` 参数给 LLM | 两轮调用模式 |
| 7 | ✅ 默认管理员密码硬编码（已修复） | `admin / admin123` 有安全隐患 | secrets 模块生成 16 位随机密码 |
| 8 | ✅ SQL 注入可绕过（已修复） | 仅检查 SELECT 开头，可用注释绕过 | 6 层纵深防御（关键字/注释/多语句/系统表/字面量剥离） |
| 9 | ✅ API 接口无认证（已修复） | FastAPI 端点未鉴权 | JWT Bearer + Depends 依赖注入 |
| 10 | ✅ CORS 全开放（已修复） | allow_origins=["*"] 允许任意域名 | 环境变量 CORS_ORIGINS 白名单 |

---

> **📅 最后更新**：2026-07-04
>
> **项目状态**：核心安全稳定性已提升（评分 3.5 → 6.5/10），关键风险已消除，持续迭代中
> **安全稳定性改进明细**：详见 [`AI_CRM/排查.md`](AI_CRM/排查.md) 的「安全稳定性改进」章节和 [`plans/crm-security-stability-improvement-plan.md`](plans/crm-security-stability-improvement-plan.md)
