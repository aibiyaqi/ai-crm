"""
metrics.py - 运行时指标埋点模块（系统运行指标 + 模型效果指标）
================================================================
背景（对应二面整改「可观测性增强」）：
  项目原来只监控 HTTP / token / 耗时这类「系统运行指标」，
  缺少「模型效果指标」的线上埋点 —— 例如：
    · 意图路由规则命中率（此前只能在面试里口头估测 60~70%）
    · 各链路（web_search / rag_retrieve / direct_llm）分布
    · RAG 检索返回条数与耗时
    · SQL 安全校验的放行 / 拒绝比例
    · 工具调用成功率、LLM 调用耗时
  本模块把这些指标以「进程内埋点」方式采集起来，
  供 /api/metrics 端点输出快照，作为监控大盘（Grafana / Prometheus）的数据源。

设计原则：
  1. 纯标准库、零外部依赖（不引入 prometheus_client，避免依赖膨胀）；
  2. 线程安全（FastAPI 多线程 + Streamlit 会话并发写入同一进程）；
  3. 埋点失败绝不影响主流程（所有函数内部吞异常，只做尽力而为的记录）；
  4. 有界内存：timing 样本与事件都用固定上限的环形缓冲，不会无限增长。

指标类型：
  - Counter（计数）：incr(name, value=1, **labels)
  - Timing（耗时/数值分布）：observe(name, ms, **labels) / timed(...)

命名约定（标签用小写 key，值用简短英文枚举）：
  chat_requests_total{status=success|error}
  chat_latency_ms
  intent_route_total{route=rule_hit|rule_miss|llm|fallback}
  intent_total{intent=web_search|rag_retrieve|direct_llm}
  link_total{link=web_search|rag_retrieve|direct_llm}
  rag_retrieval_ms / rag_retrieval_docs
  llm_call_total{stage=...} / llm_latency_ms{stage=...}
  tool_call_total{tool=...} / tool_call_error_total{tool=...}
  sql_check_total{verdict=allow|reject,stage=ast|legacy}
  sql_exec_latency_ms

用法：
  from metrics import incr, observe, timed, snapshot, record_event

  incr("chat_requests_total", status="success")
  with timed("rag_retrieval_ms"):
      docs = rag.hybrid_search(q)
  observe("rag_retrieval_docs", len(docs))
  record_event("rag_empty", query=q)     # badcase 线索（环形缓冲）

自检：
  python metrics.py
"""

# ==================== 1. 导入依赖 ====================
from __future__ import annotations

import json
import os
import threading
import time
from collections import defaultdict, deque
from datetime import datetime
from typing import Deque, Dict, List, Optional, Tuple


# ==================== 2. 内部状态（线程安全） ====================
# timing 样本上限（每个指标独立一个环形缓冲，防止长跑内存膨胀）
_MAX_SAMPLES: int = int(os.getenv("METRICS_MAX_SAMPLES", "2048"))
# 事件环形缓冲上限（badcase / 异常线索）
_MAX_EVENTS: int = int(os.getenv("METRICS_EVENT_BUFFER", "500"))

_lock = threading.Lock()

# 计数器：指标名（含标签）→ 累计值
_counters: Dict[str, float] = defaultdict(float)
# 原始计数（不含标签的基名 → 值），便于按基名聚合查询
_counter_base: Dict[str, float] = defaultdict(float)

# timing：指标名（含标签）→ 样本环形缓冲
_samples: Dict[str, Deque[float]] = {}

# 事件：最近事件（badcase 线索 / 异常）
_events: Deque[dict] = deque(maxlen=_MAX_EVENTS)

_started_at: float = time.time()


# ==================== 3. 标签 / key 处理 ====================
def _label_str(labels: Optional[dict]) -> str:
    """把标签字典渲染成稳定顺序的字符串，保证同一逻辑指标 key 一致"""
    if not labels:
        return ""
    items = sorted((str(k), str(v)) for k, v in labels.items())
    return "{" + ",".join(f"{k}={v}" for k, v in items) + "}"


def _key(name: str, labels: Optional[dict] = None) -> str:
    """完整指标 key = 基名 + 标签串"""
    return f"{name}{_label_str(labels)}"


def _split_key(key: str) -> Tuple[str, Dict[str, str]]:
    """把完整 key 拆回 (基名, 标签字典)，用于快照聚合展示"""
    if "{" not in key or not key.endswith("}"):
        return key, {}
    base, _, rest = key.partition("{")
    rest = rest[:-1]
    labels: Dict[str, str] = {}
    for pair in rest.split(","):
        if not pair:
            continue
        k, _, v = pair.partition("=")
        labels[k] = v
    return base, labels


# ==================== 4. 写入 API ====================
def incr(name: str, value: float = 1, **labels) -> None:
    """
    计数器自增（把「我觉得规则命中率还行」变成可观测数字）
    =====================================================
    参数：
      name   : 指标基名，如 "intent_route_total"
      value  : 增量（默认 1）
      labels : 标签，如 route="rule_hit"
    """
    try:
        key = _key(name, labels)
        with _lock:
            _counters[key] += value
            _counter_base[name] += value
    except Exception:  # pragma: no cover - 埋点绝不影响主流程
        pass


def observe(name: str, value: float, **labels) -> None:
    """
    记录一个数值样本（耗时 / 条数 / token 数等），用于算 avg / p50 / p95
    =========================================================================
    """
    try:
        key = _key(name, labels)
        with _lock:
            buf = _samples.get(key)
            if buf is None:
                buf = deque(maxlen=_MAX_SAMPLES)
                _samples[key] = buf
            buf.append(float(value))
    except Exception:  # pragma: no cover
        pass


class timed:
    """
    耗时统计上下文管理器
    =====================
    用法：
      with timed("rag_retrieval_ms"):
          ...
    异常时也会记录耗时（并原样抛出），保证不吞业务异常。
    """

    __slots__ = ("name", "labels", "_t0")

    def __init__(self, name: str, **labels):
        self.name = name
        self.labels = labels
        self._t0: float = 0.0

    def __enter__(self) -> "timed":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        try:
            ms = (time.perf_counter() - self._t0) * 1000.0
            observe(self.name, ms, **self.labels)
        except Exception:  # pragma: no cover
            pass
        return False  # 不吞异常


def record_event(kind: str, **fields) -> None:
    """
    记录一条事件（badcase 线索 / 异常 / 关键决策），进环形缓冲
    =========================================================
    用途：为「badcase 库」提供线上侧线索，例如：
      record_event("intent_rule_miss", query=query)
      record_event("sql_rejected", sql=..., reason=...)
      record_event("rag_empty", query=query)
    """
    try:
        event = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "kind": str(kind),
        }
        for k, v in fields.items():
            text = v if isinstance(v, (int, float, bool)) or v is None else str(v)
            event[k] = text if text is None or isinstance(text, (int, float, bool)) else text[:300]
        with _lock:
            _events.append(event)
    except Exception:  # pragma: no cover
        pass


# ==================== 5. 读取 API ====================
def get_counter(name: str, **labels) -> float:
    """读取某个（带标签的）计数器当前值"""
    with _lock:
        return float(_counters.get(_key(name, labels), 0.0))


def get_counter_base(name: str) -> float:
    """读取某个基名计数器的聚合值（跨全部标签）"""
    with _lock:
        return float(_counter_base.get(name, 0.0))


def _summary(values: List[float]) -> dict:
    """由样本列表计算 count/sum/avg/min/max/p50/p95"""
    if not values:
        return {"count": 0, "sum": 0.0, "avg": 0.0,
                "min": 0.0, "max": 0.0, "p50": 0.0, "p95": 0.0}
    ordered = sorted(values)
    n = len(ordered)

    def _pct(p: float) -> float:
        if n == 1:
            return ordered[0]
        idx = min(n - 1, max(0, int(round(p * (n - 1)))))
        return ordered[idx]

    total = sum(ordered)
    return {
        "count": n,
        "sum": round(total, 3),
        "avg": round(total / n, 3),
        "min": round(ordered[0], 3),
        "max": round(ordered[-1], 3),
        "p50": round(_pct(0.50), 3),
        "p95": round(_pct(0.95), 3),
    }


def get_timing(name: str, **labels) -> dict:
    """读取某个（带标签的）timing 指标统计"""
    with _lock:
        buf = _samples.get(_key(name, labels))
        values = list(buf) if buf else []
    return _summary(values)


def recent_events(limit: int = 50, kind: Optional[str] = None) -> List[dict]:
    """读取最近事件（可按 kind 过滤）"""
    with _lock:
        items = list(_events)
    if kind is not None:
        items = [e for e in items if e.get("kind") == kind]
    return items[-limit:][::-1]  # 最新的在前


# ==================== 6. 派生指标（比率 / 分布） ====================
def _safe_ratio(num: float, den: float) -> Optional[float]:
    """安全比率：分母为 0 时返回 None（而不是 0，避免误读为「0%」）"""
    if den <= 0:
        return None
    return round(num / den, 4)


def derived() -> dict:
    """
    由原始计数器推导出的「模型效果 + 安全」比率指标
    ================================================
    这些才是二面里被追问的东西：
      · intent_rule_hit_rate     规则路由命中率（替代口头估测 60~70%）
      · intent_llm_fallback_rate 规则未命中、转 LLM 兜底的比例
      · sql_reject_rate          SQL 安全校验拒绝率
      · tool_error_rate          工具调用失败率
    """
    cc = {k: float(v) for k, v in _counters.items()}

    def c(name: str, **labels) -> float:
        return cc.get(_key(name, labels), 0.0)

    # -------- 意图路由 --------
    route_hit = c("intent_route_total", route="rule_hit")
    route_miss = c("intent_route_total", route="rule_miss")
    route_llm = c("intent_route_total", route="llm")
    route_fallback = c("intent_route_total", route="fallback")
    route_total = route_hit + route_miss

    # -------- 链路分布 --------
    link_dist = {
        link: c("link_total", link=link)
        for link in ("web_search", "rag_retrieve", "direct_llm")
    }

    # -------- 意图分布 --------
    intent_dist = {
        intent: c("intent_total", intent=intent)
        for intent in ("web_search", "rag_retrieve", "direct_llm")
    }

    # -------- SQL 安全 --------
    sql_allow = c("sql_check_total", verdict="allow")
    sql_reject = c("sql_check_total", verdict="reject")

    # -------- 工具调用 --------
    tool_total = get_counter_base("tool_call_total")
    tool_error = get_counter_base("tool_call_error_total")

    # -------- 对话 --------
    chat_success = c("chat_requests_total", status="success")
    chat_error = c("chat_requests_total", status="error")

    return {
        "intent_rule_hit_rate": _safe_ratio(route_hit, route_total),
        "intent_rule_miss_rate": _safe_ratio(route_miss, route_total),
        "intent_llm_rate": _safe_ratio(route_llm, route_total),
        "intent_fallback_rate": _safe_ratio(route_fallback, route_total),
        "link_distribution": link_dist,
        "intent_distribution": intent_dist,
        "sql_reject_rate": _safe_ratio(sql_reject, sql_allow + sql_reject),
        "sql_allow_total": sql_allow,
        "sql_reject_total": sql_reject,
        "tool_error_rate": _safe_ratio(tool_error, tool_total),
        "chat_success_rate": _safe_ratio(chat_success, chat_success + chat_error),
        "chat_success_total": chat_success,
        "chat_error_total": chat_error,
    }


# ==================== 7. 快照 / 导出 ====================
def snapshot(include_events: int = 0, include_raw_samples: bool = False) -> dict:
    """
    输出完整指标快照（监控大盘 / 排查用）
    ======================================
    参数：
      include_events      : >0 时附带最近 N 条事件
      include_raw_samples : 是否附带每个 timing 的原始样本（默认只给统计量）
    """
    with _lock:
        counters = {k: round(float(v), 3) for k, v in _counters.items()}
        samples = {k: list(v) for k, v in _samples.items()}

    timings = {k: _summary(v) for k, v in samples.items()}

    result = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "uptime_s": round(time.time() - _started_at, 1),
        "counters_total_keys": len(counters),
        "counters": counters,
        "timings": timings,
        "derived": derived(),
    }
    if include_raw_samples:
        result["timings_raw"] = samples
    if include_events > 0:
        result["recent_events"] = recent_events(limit=include_events)
    return result


def render_prometheus() -> str:
    """
    渲染为 Prometheus 文本暴露格式（供 /metrics 抓取）
    ==================================================
    约定：
      <name>{labels} <value>
    标签值已做转义（引号 / 反斜杠 / 换行）。
    """

    def esc(v: str) -> str:
        return str(v).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")

    lines: List[str] = []
    with _lock:
        counters = dict(_counters)
        samples = {k: list(v) for k, v in _samples.items()}

    for key, val in sorted(counters.items()):
        base, labels = _split_key(key)
        if labels:
            lbl = ",".join(f'{k}="{esc(v)}"' for k, v in sorted(labels.items()))
            lines.append(f"{base}{{{lbl}}} {val}")
        else:
            lines.append(f"{base} {val}")

    for key, values in sorted(samples.items()):
        base, labels = _split_key(key)
        stats = _summary(values)
        for stat_name in ("count", "sum", "avg", "p50", "p95", "max"):
            metric = f"{base}_{stat_name}"
            value = stats[stat_name]
            if labels:
                lbl = ",".join(f'{k}="{esc(v)}"' for k, v in sorted(labels.items()))
                lines.append(f"{metric}{{{lbl}}} {value}")
            else:
                lines.append(f"{metric} {value}")

    return "\n".join(lines) + ("\n" if lines else "")


def reset() -> None:
    """清空全部指标（仅用于测试 / 演示；线上不要调用）"""
    with _lock:
        _counters.clear()
        _counter_base.clear()
        _samples.clear()
        _events.clear()


# ==================== 8. 便捷导出 ====================
__all__ = [
    "incr", "observe", "timed", "record_event",
    "get_counter", "get_counter_base", "get_timing", "recent_events",
    "derived", "snapshot", "render_prometheus", "reset",
]


# ==================== 9. 自检 ====================
if __name__ == "__main__":  # pragma: no cover
    reset()

    # -------- 模拟一次对话的埋点 --------
    incr("chat_requests_total", status="success")
    incr("intent_route_total", route="rule_hit")
    incr("intent_total", intent="rag_retrieve")
    incr("link_total", link="rag_retrieve")

    with timed("rag_retrieval_ms"):
        time.sleep(0.01)
    observe("rag_retrieval_docs", 3)

    incr("intent_route_total", route="rule_miss")
    incr("intent_route_total", route="llm")
    incr("intent_total", intent="direct_llm")
    incr("link_total", link="direct_llm")

    incr("sql_check_total", verdict="allow")
    incr("sql_check_total", verdict="reject")
    incr("tool_call_total", tool="save_customer_profile")
    incr("tool_call_error_total", tool="save_customer_profile")
    record_event("intent_rule_miss", query="我想了解一下你们的课程体系")

    snap = snapshot(include_events=5)

    print("=" * 68)
    print("metrics.py 自检")
    print("=" * 68)
    print(f"uptime_s            : {snap['uptime_s']}")
    print(f"counters            : {snap['counters']}")
    print(f"timings             : {snap['timings']}")
    print("-" * 68)
    print("derived（模型效果 + 安全指标）:")
    for k, v in snap["derived"].items():
        print(f"  {k:26s} = {v}")
    print("-" * 68)
    print("recent_events:")
    for e in snap["recent_events"]:
        print(f"  {e}")
    print("-" * 68)
    print("Prometheus 文本格式预览:")
    print(render_prometheus().strip()[:600])
    print("=" * 68)
