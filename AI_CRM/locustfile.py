# -*- coding: utf-8 -*-
"""
locustfile.py — CRM 教育系统 Locust 压测（100 并发用户）
=========================================================
定位
----
面试口径里说过的「40–50 人使用 ≠ 50 并发」，本质是**没有系统压测**。
本脚本用 Locust 对**真实 HTTP 链路**（FastAPI → CRMWorkflow → LLM → SQLite）
做端到端并发压测，补齐「100 线程（100 人并发）」的实测能力。

与 loadtest.py 的分工
---------------------
  · loadtest.py  —— 进程内自研线程池压测，聚焦「编排 / DB / embedding」单机边界，
                    用 MockLLM 剥离外部 API 时延，**不依赖后端服务**。
  · locustfile.py —— 走真实 HTTP，覆盖「网络 + 框架 + 鉴权 + 业务 + 落库」全链路，
                    是**端到端容量评估**，需要后端已启动。

鉴权说明（重要）
----------------
后端 [`api.py`](api.py:1) **没有 /api/login 路由**，认证走 `Authorization: Bearer <token>`，
token 由 `session_tokens` 表校验（见 [`get_current_user()`](api.py:99) 与
[`get_account_id_by_token()`](memory_sqlite.py:743)）。
因此本脚本在 `on_start` 里**直接向 session_tokens 表写入压测 token**（与后端同库），
绕开不存在的登录接口，专注压测业务链路本身。

用法
----
  # 1) Web UI 模式（浏览器打开 http://localhost:8089，手动设 100 用户）
  locust -f locustfile.py --host http://127.0.0.1:8000

  # 2) 无头模式（CI / 一键复现，100 并发用户，持续 60 秒）
  locust -f locustfile.py --host http://127.0.0.1:8000 ^
         --headless -u 100 -r 10 -t 60s ^
         --html loadtest_reports/locust_100u.html ^
         --csv  loadtest_reports/locust_100u

统计口径（必须随报告一起读，避免「观察值当压测指标」）
----------------------------------------------------
  并发用户数 = -u 100（Locust User），**不是「使用人数」**；
  请求量     = 各 task 按权重混合发起；
  QPS/RPS    = Locust 自带统计（成功请求 ÷ 墙钟）；
  延迟       = 客户端端到端观测（P50/P95/P99），单位 ms；
  成功率     = 成功 ÷ 总请求；失败率单列。
  说明       = 若后端 LLM 为真实 DashScope，瓶颈在**云端 API**而非本机；
               若后端为 MockLLM，则聚焦「框架 + 业务 + 落库」链路。

依赖：locust（见 requirements.txt）。后端需已启动，且与压测脚本共用同一 crm.db。
"""

from __future__ import annotations

import os
import sys
import uuid
import random

from locust import HttpUser, task, between, events

# 让脚本能 import 后端模块（memory_sqlite），用于直接写 session_tokens
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


# ==================== 1. 压测数据 ====================
# 与 loadtest.py 保持一致的混合查询：规则命中（走 rag/web，零 LLM 意图调用）
# 与规则未命中（走 LLM 意图分类）混合，让链路分布更接近真实使用。
_QUERIES = [
    "初三数学一对一课时费大概多少？",   # 规则命中 → rag_retrieve
    "你们都有哪些班型？",               # 规则命中 → rag_retrieve
    "高三英语小班的收费是多少？",       # 规则命中 → rag_retrieve
    "今天天气怎么样？",                 # 规则命中 → web_search
    "最近有什么教育新闻？",             # 规则命中 → web_search
    "你好呀，在吗？",                   # 规则未命中 → LLM 意图分类
    "嗯，知道了",                       # 规则未命中 → LLM 意图分类
    "随便聊聊吧",                       # 规则未命中 → LLM 意图分类
]

# 压测账号池：默认 100 个（与 -u 100 对齐）。
# 需后端已注册；可用环境变量 LOADTEST_ACCOUNTS 覆盖（逗号分隔）。
_env_accounts = os.getenv("LOADTEST_ACCOUNTS", "").strip()
if _env_accounts:
    _ACCOUNTS = [a.strip() for a in _env_accounts.split(",") if a.strip()]
else:
    _ACCOUNTS = [f"load_u{i}" for i in range(100)]

# 压测账号密码（后端注册时使用；默认与 loadtest.py 的占位一致）
_PASSWORD = os.getenv("LOADTEST_PASSWORD", "loadtest123")

# 压测 token 文件（由 scripts/provision_loadtest_tokens.py 预置）
# 设置后 Locust 进程不再连接数据库：
#   gevent 会 monkey-patch socket，PyMySQL 与它抢 socket 时容易 WinError 10053，
#   把「造数据」拆到普通进程里做，压测只读 token 文件，稳定性高一个数量级。
_TOKENS_FILE = os.getenv("LOADTEST_TOKENS_FILE", "").strip()
_TOKENS: list = []
if _TOKENS_FILE and os.path.exists(_TOKENS_FILE):
    try:
        import json
        with open(_TOKENS_FILE, "r", encoding="utf-8") as _fh:
            _TOKENS = json.load(_fh)
        print(f"[locust] 已从 {_TOKENS_FILE} 加载 {len(_TOKENS)} 个 token（不再连库）")
    except Exception as _exc:  # pragma: no cover
        print(f"[locust] token 文件读取失败，回退直连数据库: {_exc}")
        _TOKENS = []


def _pick_token(account_id: str) -> str:
    """优先用预置 token 文件；没有则回退到直连数据库预置。"""
    if _TOKENS:
        return random.choice(_TOKENS)
    return _provision_token(account_id)


# ==================== 1b. 压测 token 预置（绕开不存在的 /api/login） ====================
def _provision_token(account_id: str) -> str:
    """
    为压测账号直接写入 accounts + session_tokens 表，返回可用 token。

    后端 [`get_current_user()`](api.py:99) 只认 session_tokens 表里的 token，
    而 api.py 未暴露登录路由，故此处与后端共用同一 SQLite 库直接落一条令牌。

    关键：session_tokens.account_id 对 accounts(account_id) 有外键约束，
    因此必须**先建账户、再写令牌**，否则报 FOREIGN KEY constraint failed，
    导致所有请求 401（这正是首轮压测 100% 失败的原因）。
    失败时返回空串（后续请求将 401，仍计入统计，便于暴露鉴权问题）。
    """
    token = f"locust_{uuid.uuid4().hex}"
    try:
        from memory_sqlite import SQLiteMemory
        from models import Account

        db = SQLiteMemory()

        # 1) 先确保账户存在（满足 session_tokens 的外键约束）
        #    username / phone 均 UNIQUE，用 account_id 派生保证唯一且可重复执行。
        if db.get_account_by_id(account_id) is None:
            acct = Account(
                account_id=account_id,
                username=account_id,
                password_hash=_PASSWORD,  # 压测占位，不参与真实登录校验
                role="user",
                phone=f"1{abs(hash(account_id)) % 10**10:010d}",  # 11 位占位手机号
                email=f"{account_id}@loadtest.local",
                is_active=True,
            )
            db.save_account(acct)

        # 2) 再写会话令牌（此时外键可满足）
        ok = db.save_session_token(token, account_id, expires_days=1)
        db.close()
        return token if ok else ""
    except Exception as e:  # noqa: BLE001
        print(f"[locust] 预置 token 失败 account={account_id}: {e}")
        return ""


# ==================== 2. 虚拟用户 ====================
class CRMUser(HttpUser):
    """
    模拟一个课程顾问：预置 token → 反复发起对话咨询 / 拉取告警
    ================================================================
    权重设计（贴近真实使用节奏）：
      chat           权重 8  —— 主链路（对话咨询）
      list_alerts    权重 1  —— 告警红点拉取
      unread_count   权重 1  —— 未读告警数（前端角标轻量轮询）
    """
    # 思考时间：0.5~2.0 秒，模拟顾问阅读 / 打字间隔（避免纯压满 CPU）
    wait_time = between(0.5, 2.0)

    def on_start(self):
        """每个虚拟用户启动时取 token（优先预置文件，失败则 401 仍计入统计）"""
        self.account = random.choice(_ACCOUNTS)
        self.token = _pick_token(self.account)

    @property
    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.token}"} if self.token else {}

    # -------- 主链路：对话咨询 --------
    @task(8)
    def chat(self):
        """POST /api/chat —— 端到端对话（意图路由 / RAG / 生成 / 落库）"""
        q = random.choice(_QUERIES)
        # 注意：ChatRequest 字段为 query / user_id / customer_id（无 session_id）
        self.client.post(
            "/api/chat",
            json={"query": q, "user_id": self.account},
            headers=self._headers,
            name="/api/chat",
        )

    # -------- 告警红点拉取 --------
    @task(1)
    def list_alerts(self):
        """GET /api/alerts —— 告警列表（前端红点 / 角标）"""
        self.client.get(
            "/api/alerts?limit=20",
            headers=self._headers,
            name="/api/alerts",
        )

    # -------- 未读告警数（轻量轮询） --------
    @task(1)
    def unread_count(self):
        """GET /api/alerts/unread_count —— 未读告警数（前端角标）"""
        self.client.get(
            "/api/alerts/unread_count",
            headers=self._headers,
            name="/api/alerts/unread_count",
        )


# ==================== 2b. 只读压测用户（不触发 LLM） ====================
class CRMReadOnlyUser(HttpUser):
    """
    只读压测用户：只压「鉴权 + 连接池 + 查询」，不调用 LLM
    ==========================================================
    用途：
      · 没有可用的 DashScope Key、或不想烧 token 时，验证 API + MySQL 连接池 + Redis 降级；
      · 与 CRMUser（含 /api/chat）分开选择，避免把云端 LLM 延迟误读成本机瓶颈。

    运行：
      locust -f locustfile.py CRMReadOnlyUser --headless -u 100 -r 10 -t 30s \
             --host http://127.0.0.1:8000
    """
    wait_time = between(0.1, 0.5)

    def on_start(self):
        self.account = random.choice(_ACCOUNTS)
        self.token = _pick_token(self.account)

    @property
    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.token}"} if self.token else {}

    @task(3)
    def list_alerts(self):
        """GET /api/alerts —— 走 MySQL 查询 + 鉴权"""
        self.client.get("/api/alerts?limit=20", headers=self._headers, name="/api/alerts")

    @task(2)
    def unread_count(self):
        """GET /api/alerts/unread_count —— 轻量计数"""
        self.client.get("/api/alerts/unread_count", headers=self._headers,
                        name="/api/alerts/unread_count")

    @task(2)
    def due_follow_ups(self):
        """GET /api/follow_ups/due —— 到期提醒查询"""
        self.client.get("/api/follow_ups/due?days=14", headers=self._headers,
                        name="/api/follow_ups/due")

    @task(1)
    def follow_up_stats(self):
        """GET /api/follow_ups/stats —— 跟进转化统计（聚合查询）"""
        self.client.get("/api/follow_ups/stats?days=30", headers=self._headers,
                        name="/api/follow_ups/stats")

    @task(1)
    def status(self):
        """GET /api/status —— 健康检查（含 DB / Redis 健康快照）"""
        self.client.get("/api/status", name="/api/status")


# ==================== 3. 压测结束汇总（补口径提示） ====================
@events.test_stop.add_listener
def _on_test_stop(environment, **kwargs):
    """压测结束打印关键分位，并附统计口径说明（避免误读）"""
    stats = environment.stats.total
    print("\n" + "=" * 70)
    print("Locust 100 并发汇总")
    print("=" * 70)
    print(
        f"请求总数={stats.num_requests}  失败={stats.num_failures}  "
        f"RPS={stats.total_rps:.2f}"
    )
    print(
        f"P50={stats.get_response_time_percentile(0.50)}ms  "
        f"P95={stats.get_response_time_percentile(0.95)}ms  "
        f"P99={stats.get_response_time_percentile(0.99)}ms  "
        f"Max={stats.max_response_time}ms"
    )
    print("-" * 70)
    print("口径：并发 = Locust User 数（非真实使用人数）；")
    print("      延迟 = 客户端端到端观测；")
    print("      LLM 若为真实 DashScope，瓶颈在云端 API，不代表本机处理能力。")
    print("=" * 70)
