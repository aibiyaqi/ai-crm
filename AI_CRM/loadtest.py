# -*- coding: utf-8 -*-
"""
loadtest.py — CRM 教育系统「压测 / 容量边界」评估脚本
=====================================================================
定位
----
面试口径里说过的「40–50 人使用 ≠ 50 并发」「延迟 2–6s 瓶颈在云端 LLM API」，
本质是**没有系统压测**。本脚本把这块能力补上，且**不依赖真实云端 API**
（LLM 走 MockLLM，可注入延迟 / 错误 / 超时），从而把压测聚焦在系统自身：

  1. chat      —— 端到端并发跑真实 CRMWorkflow（含意图路由 / RAG 分支 / 生成 / 落库）
                   用 MockLLM 模拟云端大模型的「网络+推理延迟」，可控注入限流/超时。
  2. db_write  —— 多连接并发写 customer_profiles，观察 SQLite WAL 写锁行为。
  3. db_retry  —— 走 DAO 层 batch（SQLiteDatabase.execute），验证
                   db._run_with_retry「指数退避 + 抖动」重试埋点（db_lock_retry_total）。
  4. embedding —— （可选）本地 bge-small-zh-v1.5 的 CPU 推理饱和点；模型不可用则 SKIP。

产出
----
  · stdout 汇总表
  · loadtest_reports/loadtest_<mode>_<ts>.json  （机读）
  · loadtest_reports/loadtest_<mode>_<ts>.md    （人读：含统计口径与局限）

统计口径（写进报告，避免「观察值当压测指标」）
------------------------------------------------
  并发数   = 工作线程数（--workers）
  请求量   = --requests（按 worker 均分）
  QPS      = 成功请求数 ÷ 墙钟耗时
  延迟     = 客户端端到端观测（P50/P95/P99），单位 ms
  成功率   = 成功 ÷ 总请求；错误率 / 超时率单列
  说明     = LLM 为 mock（延迟由参数注入），**不代表真实云端时延**；
             未覆盖真实 DashScope 限流、真实向量检索（rag=None）、真实 embedding（若 SKIP）。

依赖：仅标准库 + 项目自身模块。无需 dashscope / 无需联网。
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# ==================== 0. 环境准备（必须在导入项目模块之前） ====================
# 用独立的临时 SQLite 库，避免污染真实 crm.db。
# config 里 load_dotenv() 默认不覆盖已存在的环境变量，故这里先设即生效。
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

_TMP_DIR = tempfile.mkdtemp(prefix="crm_loadtest_")
_TMP_DB = os.path.join(_TMP_DIR, "loadtest.db")
os.environ["DB_SQLITE_PATH"] = _TMP_DB
# 关键：不配置真实 API Key 也允许运行（本脚本不调用云端）。
os.environ.setdefault("QWEN_API_KEY", "mock-key-for-loadtest")

try:  # 让 Windows 控制台尽量以 utf-8 输出
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    pass


# ==================== 1. Mock LLM（零网络） ====================
class MockLLM:
    """
    模拟大模型：不发起任何网络请求。
    ------------------------------------------------------------
    目的：把系统自身（编排 / 分支 / 落库 / DB / CPU）从「外部 API 时延」里剥出来，
    同时可**可控地注入**云端 LLM 的两种典型故障：
      · 限流 / 5xx     → error_rate
      · 请求超时       → timeout_rate
    latency_ms 模拟「网络 + 推理」的期望延迟，jitter 做 ±比例抖动。

    兼容 llms.QwenLLM 的调用契约：invoke(messages, tools=None) -> dict
      · 传入 tools 且返回无 tool_calls → 触发「直接回答」分支
      · system 含「意图分类器」       → 返回一个意图标签
      · system 含「提取/JSON」        → 返回 '{}'（FALLBACK-B 抽取兜底）
    """

    def __init__(
        self,
        latency_ms: float = 300.0,
        jitter: float = 0.5,
        error_rate: float = 0.0,
        timeout_rate: float = 0.0,
        seed: int = 42,
        intent_weights: Optional[Dict[str, float]] = None,
    ):
        self.latency_ms = float(latency_ms)
        self.jitter = float(jitter)
        self.error_rate = float(error_rate)
        self.timeout_rate = float(timeout_rate)
        self.seed = int(seed)
        self.intent_weights = intent_weights or {
            "rag_retrieve": 0.5, "web_search": 0.2, "direct_llm": 0.3,
        }
        self._local = threading.local()
        # 计数字段（线程安全靠 GIL 下的整数自增足够近似；仅作展示）
        self.calls = 0
        self.injected_errors = 0
        self.injected_timeouts = 0

    # ---- 每线程独立随机源，保证可复现又不互锁 ----
    def _rng(self) -> random.Random:
        r = getattr(self._local, "rng", None)
        if r is None:
            r = random.Random((self.seed * 1000003) ^ threading.get_ident())
            self._local.rng = r
        return r

    @staticmethod
    def _system_text(messages: List[dict]) -> str:
        for m in messages:
            if m.get("role") == "system":
                return str(m.get("content", ""))
        return ""

    @staticmethod
    def _last_user_text(messages: List[dict]) -> str:
        for m in reversed(messages):
            if m.get("role") == "user":
                return str(m.get("content", ""))
        return ""

    def _pick_intent(self, r: random.Random) -> str:
        keys = list(self.intent_weights.keys())
        weights = [self.intent_weights[k] for k in keys]
        return r.choices(keys, weights=weights, k=1)[0]

    def invoke(self, messages: List[dict], tools: Optional[List[dict]] = None) -> dict:
        r = self._rng()
        self.calls += 1

        # 1) 模拟「网络 + 推理」延迟（sleep 释放 GIL → 真实并发）
        lat = self.latency_ms * (1.0 + self.jitter * (r.random() * 2.0 - 1.0))
        if lat > 0:
            time.sleep(lat / 1000.0)

        # 2) 注入故障：超时优先于错误
        p = r.random()
        if p < self.timeout_rate:
            self.injected_timeouts += 1
            raise TimeoutError("mock LLM timeout（模拟云端 API 超时）")
        if p < self.timeout_rate + self.error_rate:
            self.injected_errors += 1
            raise RuntimeError("mock LLM error（模拟限流 / 5xx）")

        text = self._last_user_text(messages)
        sys_text = self._system_text(messages)

        # 3) 生成阶段（带 tools）→ 不发起工具调用，直接给答复
        if tools is not None:
            return {
                "role": "assistant",
                "content": f"[mock] 已根据检索上下文为「{text[:20]}」整理答复。",
            }
        # 4) 意图分类
        if "意图分类器" in sys_text:
            return {"role": "assistant", "content": self._pick_intent(r)}
        # 5) FALLBACK-B 结构化抽取
        if "JSON" in sys_text or "提取" in sys_text:
            return {"role": "assistant", "content": "{}"}
        # 6) 其它（直答 / 兜底）
        return {"role": "assistant", "content": f"[mock] 这是对「{text[:20]}」的回答。"}


# ==================== 2. 统计工具 ====================
def _percentile(sorted_vals: List[float], p: float) -> float:
    """最近秩法求分位数（与 metrics._summary 的 p95 口径一致）"""
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    idx = int(round(p * (len(sorted_vals) - 1)))
    return float(sorted_vals[idx])


@dataclass
class LoadResult:
    mode: str
    workers: int
    requests: int
    duration_s: float = 0.0
    ok: int = 0
    errors: int = 0
    timeouts: int = 0
    skipped: bool = False
    reason: str = ""
    # 延迟统计（ms）
    lat_min: float = 0.0
    lat_avg: float = 0.0
    lat_p50: float = 0.0
    lat_p95: float = 0.0
    lat_p99: float = 0.0
    lat_max: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def qps(self) -> float:
        return round(self.ok / self.duration_s, 3) if self.duration_s > 0 else 0.0

    @property
    def success_rate(self) -> float:
        return round(self.ok / self.requests, 4) if self.requests else 0.0

    @property
    def error_rate(self) -> float:
        return round(self.errors / self.requests, 4) if self.requests else 0.0

    @property
    def timeout_rate(self) -> float:
        return round(self.timeouts / self.requests, 4) if self.requests else 0.0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.update({
            "qps": self.qps,
            "success_rate": self.success_rate,
            "error_rate": self.error_rate,
            "timeout_rate": self.timeout_rate,
        })
        return d


def _fill_latency(res: LoadResult, lats: List[float]) -> None:
    if not lats:
        return
    s = sorted(lats)
    res.lat_min = round(s[0], 2)
    res.lat_avg = round(statistics.fmean(s), 2)
    res.lat_p50 = round(_percentile(s, 0.50), 2)
    res.lat_p95 = round(_percentile(s, 0.95), 2)
    res.lat_p99 = round(_percentile(s, 0.99), 2)
    res.lat_max = round(s[-1], 2)


# ==================== 3. 场景 1：端到端 chat ====================
# 混合规则命中（走 rag / web，零 LLM 意图调用）与规则未命中（走 LLM 意图分类）的查询，
# 让链路分布更接近真实使用。
_CHAT_QUERIES = [
    "初三数学一对一课时费大概多少？",   # 规则命中 → rag_retrieve
    "你们都有哪些班型？",               # 规则命中 → rag_retrieve
    "高三英语小班的收费是多少？",       # 规则命中 → rag_retrieve
    "今天天气怎么样？",                 # 规则命中 → web_search
    "最近有什么教育新闻？",             # 规则命中 → web_search
    "你好呀，在吗？",                   # 规则未命中 → LLM 意图分类
    "嗯，知道了",                       # 规则未命中 → LLM 意图分类
    "随便聊聊吧",                       # 规则未命中 → LLM 意图分类
]


def run_chat(args: argparse.Namespace) -> LoadResult:
    from agent import CRMWorkflow          # noqa: E402  延迟导入（环境已就绪）
    from memory_sqlite import SQLiteMemory  # noqa: E402
    import metrics as M                    # noqa: E402

    # 先串行初始化 schema / WAL，避免各线程首次建表互相争用；
    # 并预置 accounts —— 因 sessions.account_id 的 FK 指向 accounts（真实使用中该账户已注册），
    # 否则 _node_save_history 建会话/存消息会因 FK 失败而只打日志、落库路径跑不通。
    from models import Account  # noqa: E402
    _seed = SQLiteMemory()
    for w in range(args.workers):
        aid = f"load_u{w}"
        if _seed.get_account_by_id(aid) is None:
            _seed.save_account(Account(
                account_id=aid,
                username=aid,
                password_hash="loadtest",
                phone=f"1390000{w:04d}",
            ))
    _seed.close()

    # 共享同一个 MockLLM（便于汇总总调用次数）；每个 worker 各持「独立数据库连接 + 工作流实例」。
    # 关键事实（并发安全边界，写进报告）：
    #   SQLiteMemory 是「一实例一连接一游标」（__init__ 里 self.conn=self._db.connect()）；
    #   单个实例跨线程共用会在 sqlite3 原生层崩溃（access violation）——
    #   线上 API 靠 **单事件循环串行化** 规避，多进程/multi-worker 部署下本就是「一 worker 一条连接」。
    #   故压测按 worker 隔离连接，既安全又能真实测出 SQLite 写并发行为。
    llm = MockLLM(
        latency_ms=args.llm_latency_ms,
        error_rate=args.llm_error_rate,
        timeout_rate=args.llm_timeout_rate,
        seed=args.seed,
    )

    per_worker = max(1, args.requests // max(1, args.workers))
    states = {"ok": 0, "err": 0, "to": 0}
    lock = threading.Lock()

    def _worker(wid: int) -> List[float]:
        lats: List[float] = []
        db = SQLiteMemory()                                   # 线程内独立连接
        # rag=None / searcher=None：本次压测关注「编排 + LLM 时延」，外部检索另行评估。
        workflow = CRMWorkflow(llm=llm, db=db, rag=None, searcher=None)
        rng = random.Random(args.seed + wid)
        for i in range(per_worker):
            q = _CHAT_QUERIES[rng.randrange(len(_CHAT_QUERIES))]
            t0 = time.perf_counter()
            try:
                workflow.run(
                    query=q,
                    user_id=f"load_u{wid}",
                    session_id=f"load_s{wid}",
                )
                dt = (time.perf_counter() - t0) * 1000.0
                lats.append(dt)
                with lock:
                    states["ok"] += 1
            except TimeoutError:
                with lock:
                    states["to"] += 1
            except Exception:
                with lock:
                    states["err"] += 1
        db.close()
        return lats

    res = LoadResult(mode="chat", workers=args.workers, requests=per_worker * args.workers)
    t_start = time.perf_counter()
    all_lats: List[float] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_worker, w) for w in range(args.workers)]
        for f in as_completed(futs):
            all_lats.extend(f.result())
    res.duration_s = round(time.perf_counter() - t_start, 3)

    res.requests = states["ok"] + states["err"] + states["to"]
    res.ok, res.errors, res.timeouts = states["ok"], states["err"], states["to"]
    _fill_latency(res, all_lats)

    # 采集系统内建埋点（agent.run 已在内部记录 chat_latency_ms / chat_requests_total）
    res.extra = {
        "llm_calls": llm.calls,
        "llm_injected_errors": llm.injected_errors,
        "llm_injected_timeouts": llm.injected_timeouts,
        "metric_chat_latency_ms": M.get_timing("chat_latency_ms"),
        "metric_llm_latency_ms": M.get_timing("llm_latency_ms"),
        "metric_derived": M.derived(),
        "note": (
            "chat 场景按 worker 隔离数据库连接（SQLiteMemory 一实例一连接一游标）；"
            "线上 API 单事件循环串行化即等效于此约束"
        ),
        "degradation_note": (
            "注入 LLM 故障时：意图节点有兜底（方案B→规则/direct_llm），失败仅降级不报错；"
            "但生成节点 LLM 异常会向上抛出→计为 err/timeout。这是当前真实健壮性边界。"
        ),
    }
    return res


# ==================== 4. 场景 2：SQLite 并发写（真实写路径） ====================
_INSERT_SQL = (
    "INSERT INTO customer_profiles "
    "(customer_id, name, phone, intent_course, source, tags, created_at, updated_at) "
    "VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP) "
    "ON CONFLICT(customer_id) DO UPDATE SET "
    "phone=excluded.phone, intent_course=excluded.intent_course, "
    "updated_at=CURRENT_TIMESTAMP"
)


def run_db_write(args: argparse.Namespace) -> LoadResult:
    """多连接并发写：每个 worker 独立 sqlite 连接（贴近多进程/多 worker 部署）。"""
    from db import SQLiteDatabase   # noqa: E402
    from memory_sqlite import SQLiteMemory  # noqa: E402
    import metrics as M             # noqa: E402

    SQLiteMemory().close()  # 先建表（幂等）

    per_worker = max(1, args.requests // max(1, args.workers))
    states = {"ok": 0, "err": 0, "to": 0}
    lock = threading.Lock()
    lock_retry_before = M.get_counter_base("db_lock_retry_total")

    def _worker(wid: int) -> List[float]:
        conn = SQLiteDatabase(path=_TMP_DB)  # 每个 worker 独立连接
        lats: List[float] = []
        for i in range(per_worker):
            cid = f"lt_{wid}_{i}"
            t0 = time.perf_counter()
            try:
                conn.execute(_INSERT_SQL, (
                    cid, f"压测客户{wid}-{i}", f"138{wid:04d}{i:04d}",
                    "初三数学一对一", "压测", '["压测"]',
                ))
                lats.append((time.perf_counter() - t0) * 1000.0)
                with lock:
                    states["ok"] += 1
            except Exception:
                with lock:
                    states["err"] += 1
        conn.close()
        return lats

    res = LoadResult(mode="db_write", workers=args.workers, requests=per_worker * args.workers)
    t_start = time.perf_counter()
    all_lats: List[float] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_worker, w) for w in range(args.workers)]
        for f in as_completed(futs):
            all_lats.extend(f.result())
    res.duration_s = round(time.perf_counter() - t_start, 3)

    res.requests = states["ok"] + states["err"] + states["to"]
    res.ok, res.errors, res.timeouts = states["ok"], states["err"], states["to"]
    _fill_latency(res, all_lats)

    res.extra = {
        "db_lock_retry_total_delta": M.get_counter_base("db_lock_retry_total") - lock_retry_before,
        "note": "走 DAO(SQLiteDatabase.execute) 层，自带指数退避重试；重试次数计入 db_lock_retry_total",
    }
    return res


# ==================== 5. 场景 3：写锁重试压测（放大争用） ====================
def run_db_retry(args: argparse.Namespace) -> LoadResult:
    """
    用「短写 + 高并发 + 大事务」放大 SQLite 写锁争用，验证应用层重试兜底。
    通过 db_lock_retry_total 观察到底触发了多少次重试。
    """
    from db import SQLiteDatabase   # noqa: E402
    from memory_sqlite import SQLiteMemory  # noqa: E402
    import metrics as M             # noqa: E402

    SQLiteMemory().close()
    per_worker = max(1, args.requests // max(1, args.workers))
    states = {"ok": 0, "err": 0}
    lock = threading.Lock()
    retry_before = M.get_counter_base("db_lock_retry_total")

    def _worker(wid: int) -> List[float]:
        conn = SQLiteDatabase(path=_TMP_DB)
        lats: List[float] = []
        for i in range(per_worker):
            cid = f"rt_{wid}_{i}"
            t0 = time.perf_counter()
            try:
                # 每条都用事务包裹（BEGIN...COMMIT），制造更多写窗口
                conn.execute(_INSERT_SQL, (
                    cid, f"重试客户{wid}-{i}", f"139{wid:04d}{i:04d}",
                    "高三英语小班", "压测-retry", '["压测","重试"]',
                ))
                lats.append((time.perf_counter() - t0) * 1000.0)
                with lock:
                    states["ok"] += 1
            except Exception:
                with lock:
                    states["err"] += 1
        conn.close()
        return lats

    res = LoadResult(mode="db_retry", workers=args.workers, requests=per_worker * args.workers)
    t_start = time.perf_counter()
    all_lats: List[float] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_worker, w) for w in range(args.workers)]
        for f in as_completed(futs):
            all_lats.extend(f.result())
    res.duration_s = round(time.perf_counter() - t_start, 3)

    res.requests = states["ok"] + states["err"]
    res.ok, res.errors = states["ok"], states["err"]
    _fill_latency(res, all_lats)

    res.extra = {
        "db_lock_retry_total_delta": M.get_counter_base("db_lock_retry_total") - retry_before,
        "note": "重试次数 > 0 说明写锁争用被应用层重试兜住了；= 0 说明并发未打到锁冲突阈值",
    }
    return res


# ==================== 6. 场景 4：本地 embedding CPU 饱和（可选） ====================
_EMBED_TEXTS = [
    "初三数学一对一课时费大概多少？",
    "你们有哪些班型和课程体系？",
    "高三英语冲刺班怎么报名？",
    "小升初衔接班的教学大纲是什么？",
    "退费和续费的规则是怎样的？",
]


def run_embedding(args: argparse.Namespace) -> LoadResult:
    res = LoadResult(mode="embedding", workers=args.workers, requests=args.requests)
    if getattr(args, "hf_offline", False):
        # 只读本地 HF 缓存：避免联网校验 config 触发分钟级重试，保证压测在无网环境可复现。
        # 注意：必须在 import huggingface_hub 之前设置，故放在 rag_law 的延迟导入之前。
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        from rag_law import BGEEmbeddings  # noqa: E402
        emb = BGEEmbeddings()
        emb.embed_query("预热")  # 懒加载 + 预热的首次成本不计入
    except Exception as e:  # 模型不可用（如离线且无本地缓存）→ 如实 SKIP，不编造数据
        res.skipped = True
        res.reason = (
            f"本地 embedding 不可用，已跳过：{type(e).__name__}: {e}"
            f"（本机若已有 HF 缓存，可加 --hf-offline 只读本地缓存重跑）"
        )
        return res

    per_worker = max(1, args.requests // max(1, args.workers))
    states = {"ok": 0, "err": 0}
    lock = threading.Lock()

    def _worker(wid: int) -> List[float]:
        lats: List[float] = []
        for i in range(per_worker):
            text = _EMBED_TEXTS[(wid + i) % len(_EMBED_TEXTS)]
            t0 = time.perf_counter()
            try:
                emb.embed_query(text)
                lats.append((time.perf_counter() - t0) * 1000.0)
                with lock:
                    states["ok"] += 1
            except Exception:
                with lock:
                    states["err"] += 1
        return lats

    t_start = time.perf_counter()
    all_lats: List[float] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_worker, w) for w in range(args.workers)]
        for f in as_completed(futs):
            all_lats.extend(f.result())
    res.duration_s = round(time.perf_counter() - t_start, 3)

    res.requests = states["ok"] + states["err"]
    res.ok, res.errors = states["ok"], states["err"]
    _fill_latency(res, all_lats)
    res.extra = {
        "texts_per_sec": round(res.ok / res.duration_s, 2) if res.duration_s else 0.0,
        "hf_offline": bool(getattr(args, "hf_offline", False)),
        "note": "本地 CPU 推理；受 GIL / 线程数影响，用于观察单机吞吐与延迟量级（非 GPU、非 DashScope 口径）",
    }
    return res


# ==================== 7. 报告输出 ====================
def _fmt_row(r: LoadResult) -> str:
    if r.skipped:
        return f"{r.mode:<10} SKIP  {r.reason}"
    return (
        f"{r.mode:<10} workers={r.workers:<4} req={r.requests:<6} "
        f"ok={r.ok:<6} err={r.errors:<5} to={r.timeouts:<5} "
        f"QPS={r.qps:<8} p50={r.lat_p50:<8} p95={r.lat_p95:<8} "
        f"p99={r.lat_p99:<8} max={r.lat_max:<8} ({r.duration_s}s)"
    )


def _write_reports(results: List[LoadResult], args: argparse.Namespace) -> Tuple[str, str]:
    out_dir = os.path.join(_HERE, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(out_dir, f"loadtest_{args.mode}_{ts}.json")
    md_path = os.path.join(out_dir, f"loadtest_{args.mode}_{ts}.md")

    import metrics as M  # noqa: E402
    payload = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "mode": args.mode,
        "params": {
            "workers": args.workers,
            "requests": args.requests,
            "llm_latency_ms": args.llm_latency_ms,
            "llm_error_rate": args.llm_error_rate,
            "llm_timeout_rate": args.llm_timeout_rate,
            "seed": args.seed,
            "hf_offline": bool(getattr(args, "hf_offline", False)),
        },
        "results": [r.to_dict() for r in results],
        "metrics_snapshot": M.snapshot(include_events=0),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    lines: List[str] = []
    lines.append(f"# CRM 教育系统 · 压测报告（mode={args.mode}）\n")
    lines.append(f"- 生成时间：{payload['ts']}")
    lines.append(f"- 工作线程（并发）：{args.workers}　请求量：{args.requests}")
    lines.append(
        f"- Mock LLM：延迟 {args.llm_latency_ms}ms / 错误率 {args.llm_error_rate} / "
        f"超时率 {args.llm_timeout_rate} / seed={args.seed}\n"
    )
    lines.append("## 结果\n")
    lines.append("| 场景 | 并发 | 请求 | 成功 | 错误 | 超时 | QPS | P50(ms) | P95(ms) | P99(ms) | Max(ms) | 耗时(s) |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in results:
        if r.skipped:
            lines.append(f"| {r.mode} | - | - | - | - | - | - | - | - | - | - | SKIP：{r.reason} |")
        else:
            lines.append(
                f"| {r.mode} | {r.workers} | {r.requests} | {r.ok} | {r.errors} | "
                f"{r.timeouts} | {r.qps} | {r.lat_p50} | {r.lat_p95} | {r.lat_p99} | "
                f"{r.lat_max} | {r.duration_s} |"
            )
    lines.append("\n## 场景附加指标\n")
    for r in results:
        if r.extra:
            lines.append(f"### {r.mode}")
            for k, v in r.extra.items():
                lines.append(f"- `{k}` = {json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v}")
            lines.append("")

    lines.append("## 统计口径与局限（必须随报告一起读）\n")
    lines.append("- 并发 = 工作线程数；QPS = 成功数 ÷ 墙钟耗时；延迟为**客户端端到端观测**。")
    lines.append("- LLM 为 **MockLLM**（延迟/错误/超时由参数注入），**不代表真实 DashScope 时延**。")
    lines.append("- chat 场景 `rag=None`，未覆盖真实向量检索与真实 LLM 调用链（仅覆盖编排/落库/指标链路）。")
    lines.append("- embedding 场景为**本地 CPU 单机口径**（`--hf-offline` 时只读本地 HF 缓存）；"
                 "模型不可用时如实 SKIP，脚本不编造数值。")
    lines.append("- db_* 场景使用**独立临时库**（不污染 crm.db）；`db_lock_retry_total` 为进程内埋点增量。")
    lines.append("- 本报告用于「容量边界 / 稳定性」评估，非吞吐 benchmark；接入线上需替换 MockLLM 为真实调用。")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return json_path, md_path


# ==================== 8. 参数与主流程 ====================
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CRM 教育系统 压测 / 容量边界评估")
    p.add_argument("--mode", choices=["chat", "db_write", "db_retry", "embedding", "all"],
                   default="all", help="压测场景")
    p.add_argument("--workers", type=int, default=10, help="并发工作线程数")
    p.add_argument("--requests", type=int, default=200, help="总请求数（按 worker 均分）")
    p.add_argument("--llm-latency-ms", type=float, default=300.0, help="Mock LLM 期望延迟(ms)")
    p.add_argument("--llm-error-rate", type=float, default=0.0, help="Mock LLM 注入错误率(0~1)")
    p.add_argument("--llm-timeout-rate", type=float, default=0.0, help="Mock LLM 注入超时率(0~1)")
    p.add_argument("--out-dir", default="loadtest_reports", help="报告输出目录")
    p.add_argument("--seed", type=int, default=42, help="随机种子（可复现）")
    p.add_argument("--no-embedding", action="store_true", help="all 模式下跳过 embedding")
    p.add_argument("--hf-offline", action="store_true",
                   help="embedding 场景只读本地 HF 缓存（无网环境可复现；首次仍须联网下载模型）")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    # 降噪：关掉 INFO 级日志（压测时海量日志会显著拖慢并掩盖真实耗时）
    try:
        import logging
        logging.disable(logging.INFO)
    except Exception:
        pass

    import metrics as M  # noqa: E402
    M.reset()

    print("=" * 78)
    print(f"CRM 教育系统 压测 · mode={args.mode} workers={args.workers} requests={args.requests}")
    print(f"临时数据库：{_TMP_DB}")
    print(f"Mock LLM：latency={args.llm_latency_ms}ms error={args.llm_error_rate} timeout={args.llm_timeout_rate}")
    print("=" * 78)

    results: List[LoadResult] = []
    mode = args.mode
    if mode in ("chat", "all"):
        results.append(run_chat(args))
    if mode in ("db_write", "all"):
        results.append(run_db_write(args))
    if mode in ("db_retry", "all"):
        results.append(run_db_retry(args))
    if mode in ("embedding", "all") and not (args.no_embedding and mode == "all"):
        results.append(run_embedding(args))

    print("\n---- 结果 ----")
    for r in results:
        print(_fmt_row(r))

    json_path, md_path = _write_reports(results, args)
    print(f"\n报告已输出：\n  JSON: {json_path}\n  MD  : {md_path}")

    # 有「非注入」的明确失败（非 SKIP）时以非零码退出，便于 CI 判定；
    # 主动注入故障（error/timeout>0）时失败是预期结果，不视为脚本失败。
    injected = args.llm_error_rate > 0 or args.llm_timeout_rate > 0
    failed = any((not r.skipped) and r.errors > 0 and r.success_rate < 0.999 for r in results)
    return 1 if (failed and not injected) else 0


if __name__ == "__main__":
    raise SystemExit(main())
