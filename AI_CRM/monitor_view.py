"""
monitor_view.py - 可观测性监控大盘（Streamlit 页面）
====================================================
对应二面整改「可观测性」缺口：把「系统运行指标」与「模型效果指标」落到一个看板上，
而不是停留在口头描述。分两大板块：

  A. 系统运行指标（进程内 metrics 埋点，实时）
     · 请求量 / 成功率 / 平均对话耗时
     · 意图路由命中率（规则 vs LLM vs 兜底）—— 替代「60~70% 靠估」的口径
     · 链路分布（web_search / rag_retrieve / direct_llm）
     · LLM 调用延迟（按 stage 分段 p50/p95）、工具调用成功率
     · RAG 检索耗时 / 召回条数、SQL 安全校验放行/拦截
     · 最近事件（规则未命中、RAG 空召回、工具报错等）

  B. 模型效果指标（离线评测 eval_metrics + 评测数据管理库 eval_store）
     · 意图路由命中率、端到端准确率
     · 检索 Recall@K / Hit@K / MRR@K
     · 结构化意图安全用例通过率
     · 生成质量 groundedness / 幻觉率
     · 历史评测运行记录、badcase 与测试集统计

依赖说明：
  · 需要 streamlit；缺失时页面内给出提示而非直接崩溃。
  · metrics / eval_metrics / eval_store 全部「惰性 + 容错」导入，单项不可用不影响其它板块。

接入方式（app.py 管理员导航新增一项「[MONITOR] 监控大盘」）：
    from monitor_view import render_monitor_page
    render_monitor_page()

单独预览：
    streamlit run monitor_view.py
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Tuple

# 将项目根目录加入 sys.path，保证独立运行时可导入同目录模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import streamlit as st
except Exception:  # pragma: no cover - 非 Streamlit 环境
    st = None  # type: ignore

try:
    import metrics as _metrics
except Exception:  # pragma: no cover
    _metrics = None  # type: ignore

try:
    import eval_store as _eval_store
except Exception:  # pragma: no cover
    _eval_store = None  # type: ignore


# ==================== 1. 工具函数 ====================
def _parse_key(key: str) -> Tuple[str, Dict[str, str]]:
    """把 'name|k=v,k=v' 形式的指标 key 拆成 (name, labels)"""
    if "|" not in key:
        return key, {}
    name, _, tail = key.partition("|")
    labels: Dict[str, str] = {}
    for pair in tail.split(","):
        if "=" in pair:
            k, v = pair.split("=", 1)
            labels[k.strip()] = v.strip()
    return name, labels


def _pct(value: Optional[float]) -> str:
    """比率 → 百分比文本；None 显示 n/a（区分「无数据」与「0%」）"""
    if value is None:
        return "n/a"
    return f"{value * 100:.1f}%"


def _fmt(value, nd: int = 2, default: str = "n/a") -> str:
    if value is None:
        return default
    if isinstance(value, float):
        return f"{value:.{nd}f}"
    return str(value)


def _timing_rows(timings: Dict[str, dict], name: str) -> List[dict]:
    """从 timings 里挑出某个指标名的所有 label 维度，整理为表格行"""
    rows: List[dict] = []
    for key, stat in (timings or {}).items():
        mname, labels = _parse_key(key)
        if mname != name:
            continue
        row = {"维度": ",".join(f"{k}={v}" for k, v in labels.items()) or "-"}
        row.update({
            "次数": stat.get("count"),
            "平均(ms)": _fmt(stat.get("avg"), 1),
            "P50(ms)": _fmt(stat.get("p50"), 1),
            "P95(ms)": _fmt(stat.get("p95"), 1),
            "最大(ms)": _fmt(stat.get("max"), 1),
        })
        rows.append(row)
    return rows


def _counter_rows(counters: Dict[str, float], name: str) -> List[dict]:
    rows: List[dict] = []
    for key, val in (counters or {}).items():
        mname, labels = _parse_key(key)
        if mname != name:
            continue
        rows.append({
            "维度": ",".join(f"{k}={v}" for k, v in labels.items()) or "-",
            "值": val,
        })
    return rows


def _bar(d: Dict[str, float], title: str) -> None:
    """简易柱状图；数据为空时给出提示"""
    data = {k: v for k, v in (d or {}).items() if v}
    if not data:
        st.caption(f"{title}：暂无数据")
        return
    st.caption(title)
    try:
        st.bar_chart(data)
    except Exception:
        st.dataframe([{"维度": k, "值": v} for k, v in data.items()], use_container_width=True)


# ==================== 2. 板块 A：系统运行指标 ====================
def _render_runtime_metrics() -> None:
    st.subheader("[CHART] 系统运行指标（实时）")

    if _metrics is None:
        st.warning("metrics 模块不可用，无法展示运行指标")
        return

    c1, c2 = st.columns([3, 1])
    with c2:
        if st.button("[REFRESH] 刷新指标", use_container_width=True):
            st.rerun()
        if st.button("[WARN] 重置指标", use_container_width=True,
                     help="清空进程内全部计数（仅演示 / 压测前使用）"):
            _metrics.reset()
            st.rerun()

    snap = _metrics.snapshot(include_events=30)
    counters = snap.get("counters", {})
    timings = snap.get("timings", {})
    d = snap.get("derived", {})

    # -------- A.1 核心指标卡 --------
    chat_ok = d.get("chat_success_total", 0.0)
    chat_err = d.get("chat_error_total", 0.0)
    chat_total = chat_ok + chat_err
    chat_lat = (timings.get("chat_latency_ms") or {})
    rag_lat = (timings.get("rag_retrieval_ms") or {})

    r1 = st.columns(4)
    r1[0].metric("对话请求数", f"{int(chat_total)}")
    r1[1].metric("对话成功率", _pct(d.get("chat_success_rate")))
    r1[2].metric("平均对话耗时", f"{_fmt(chat_lat.get('avg'), 1)} ms")
    r1[3].metric("平均 RAG 检索", f"{_fmt(rag_lat.get('avg'), 1)} ms")

    r2 = st.columns(4)
    r2[0].metric("规则路由命中率", _pct(d.get("intent_rule_hit_rate")),
                 help="规则命中 / (规则命中 + 规则未命中)，替代口头估测")
    r2[1].metric("LLM 兜底占比", _pct(d.get("intent_llm_rate")))
    r2[2].metric("SQL 拦截率", _pct(d.get("sql_reject_rate")))
    r2[3].metric("工具失败率", _pct(d.get("tool_error_rate")))

    st.caption(f"快照时间：{snap.get('ts')} · 进程运行时长：{snap.get('uptime_s')}s · "
               f"计数器维度数：{snap.get('counters_total_keys')}")

    st.divider()

    # -------- A.2 链路分布 / 意图分布 --------
    col_a, col_b = st.columns(2)
    with col_a:
        _bar(d.get("link_distribution", {}), "[LINK] 链路分布（link_total）")
    with col_b:
        _bar(d.get("intent_distribution", {}), "[ROUTE] 意图分布（intent_total）")

    # -------- A.3 延迟明细 --------
    with st.expander("[CLOCK] 延迟明细（p50 / p95）", expanded=False):
        for name, title in (
            ("llm_latency_ms", "LLM 调用延迟（按 stage）"),
            ("chat_latency_ms", "对话端到端延迟"),
            ("rag_retrieval_ms", "RAG 检索延迟"),
            ("web_search_ms", "联网搜索延迟"),
            ("tool_latency_ms", "工具调用延迟"),
            ("sql_exec_latency_ms", "SQL 执行延迟"),
            ("doc_parse_ms", "文档解析延迟（按扩展名）"),
        ):
            rows = _timing_rows(timings, name)
            if rows:
                st.markdown(f"**{title}**")
                st.dataframe(rows, use_container_width=True, hide_index=True)

    # -------- A.4 计数明细 --------
    with st.expander("[NUM] 计数明细", expanded=False):
        for name, title in (
            ("intent_route_total", "意图路由（rule_hit / rule_miss / llm / fallback）"),
            ("llm_call_total", "LLM 调用次数（按 stage）"),
            ("tool_call_total", "工具调用次数"),
            ("tool_call_error_total", "工具调用失败次数"),
            ("sql_check_total", "SQL 校验（allow / reject，按 stage）"),
            ("sql_exec_error_total", "SQL 执行错误"),
            ("chat_requests_total", "对话请求（success / error）"),
        ):
            rows = _counter_rows(counters, name)
            if rows:
                st.markdown(f"**{title}**")
                st.dataframe(rows, use_container_width=True, hide_index=True)

        # RAG 召回条数分布 + 文档解析计数
        rag_docs = timings.get("rag_retrieval_docs")
        if rag_docs:
            st.markdown("**RAG 召回条数分布**")
            st.dataframe([{
                "次数": rag_docs.get("count"),
                "平均": _fmt(rag_docs.get("avg"), 1),
                "P50": _fmt(rag_docs.get("p50"), 1),
                "P95": _fmt(rag_docs.get("p95"), 1),
            }], use_container_width=True, hide_index=True)

    # -------- A.5 最近事件 --------
    events = snap.get("recent_events", [])
    with st.expander(f"[EVENT] 最近事件（{len(events)} 条）", expanded=bool(events)):
        if not events:
            st.caption("暂无事件")
        else:
            rows = []
            for ev in reversed(events):
                row = dict(ev)
                rows.append(row)
            st.dataframe(rows, use_container_width=True, hide_index=True)


# ==================== 3. 板块 B：模型效果指标 ====================
def _render_offline_eval() -> None:
    st.subheader("[TARGET] 模型效果指标（离线评测）")

    try:
        import eval_metrics
    except Exception as e:
        st.warning(f"eval_metrics 不可用：{e}")
        return

    with st.form("offline_eval_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            run_real_rag = st.checkbox(
                "使用真实 RAG 检索", value=False,
                help="关闭时用合成语料自检（零依赖、秒级返回）；开启需加载知识库",
            )
            k = st.number_input("检索截断 K", min_value=1, max_value=20, value=5)
        with c2:
            from_store = st.checkbox("从评测库读取测试集", value=False,
                                     help="读取 eval_store.db 中已激活版本的数据集")
            save_badcases = st.checkbox("badcase 落库", value=True)
        with c3:
            seed_store = st.checkbox("写入内置黄金集", value=False,
                                     help="把内置测试集 seed 进评测库（首次初始化用）")
            submitted = st.form_submit_button("[RUN] 运行评测", use_container_width=True)

    if submitted:
        args = eval_metrics.parse_args([])
        # 显式设置评测开关（合成 / 真实 二选一）
        args.self_test = not run_real_rag
        args.intent = True
        args.safety = True
        args.groundedness = True
        args.retrieval = bool(run_real_rag)
        args.k = int(k)
        args.from_store = bool(from_store)
        args.seed_store = bool(seed_store)
        args.save_badcases = bool(save_badcases)
        args.json = None
        with st.spinner("评测运行中..."):
            try:
                report = eval_metrics.build_report(args)
                st.session_state["monitor_eval_report"] = report
                st.success("评测完成")
            except Exception as e:
                st.error(f"评测失败：{e}")

    report = st.session_state.get("monitor_eval_report")
    if not report:
        st.info("点击「运行评测」生成当前模型效果指标（默认走合成语料自检，无需 API Key）")
        return

    # -------- 指标卡 --------
    intent = report.get("intent") or {}
    retrieval = report.get("retrieval") or {}
    safety = report.get("safety") or {}
    grounded = report.get("groundedness") or {}

    cols = st.columns(4)
    cols[0].metric("意图规则命中率", _pct(intent.get("rule_hit_rate")),
                   help=f"来源：{intent.get('source', '-')}")
    cols[1].metric("端到端准确率", _pct(intent.get("e2e_accuracy")))
    cols[2].metric("检索 Recall@K", _fmt(retrieval.get("recall_at_k"), 3),
                   help=f"模式：{retrieval.get('mode', '-')}")
    cols[3].metric("检索 MRR@K", _fmt(retrieval.get("mrr_at_k"), 3))

    cols2 = st.columns(4)
    cols2[0].metric("检索 Hit@K", _fmt(retrieval.get("hit_at_k"), 3))
    cols2[1].metric("安全用例通过",
                    f"{_fmt(safety.get('passed'), 0)} / {_fmt(safety.get('total'), 0)}",
                    help=f"通过率：{_pct(safety.get('accuracy'))}")
    cols2[2].metric("Groundedness", _pct(grounded.get("groundedness")))
    cols2[3].metric("幻觉率", _pct(grounded.get("hallucination_rate")))

    # -------- 明细 --------
    with st.expander("[BOOK] 评测明细", expanded=False):
        st.markdown(f"**意图评估**（规则准度：{_pct(intent.get('rule_accuracy'))}）")
        st.json({k: v for k, v in intent.items() if k != "badcases"}, expanded=False)
        st.markdown(f"**检索评估**（K={retrieval.get('k')}，模式：{retrieval.get('mode')}）")
        st.json({k: v for k, v in retrieval.items() if k != "details"}, expanded=False)
        st.markdown("**安全评估**")
        st.json({k: v for k, v in safety.items() if k not in ("results", "badcases")},
                expanded=False)
        st.markdown("**生成质量评估**")
        st.json({k: v for k, v in grounded.items() if k not in ("details", "badcases")},
                expanded=False)

    st.download_button(
        "[SAVE] 下载评测报告(JSON)",
        data=__import__("json").dumps(report, ensure_ascii=False, indent=2),
        file_name="eval_report.json",
        mime="application/json",
    )


# ==================== 4. 板块 C：评测数据管理库 ====================
def _render_eval_store() -> None:
    st.subheader("[DB] 评测数据管理库（测试集 / badcase / 运行记录）")

    if _eval_store is None:
        st.warning("eval_store 模块不可用")
        return

    try:
        store = _eval_store.EvalStore()
        stats = store.stats()
    except Exception as e:
        st.warning(f"评测库打开失败：{e}")
        return

    c = st.columns(4)
    c[0].metric("数据集", stats.get("datasets", 0))
    c[1].metric("用例数", stats.get("cases", 0))
    c[2].metric("badcase 总数", stats.get("badcases", 0))
    c[3].metric("运行记录", stats.get("runs", 0))

    tab1, tab2, tab3 = st.tabs(["[LIST] 数据集", "[WARN] badcase", "[HISTORY] 运行记录"])

    with tab1:
        try:
            datasets = store.list_datasets()
        except Exception:
            datasets = []
        if not datasets:
            st.caption("暂无数据集（可在上方勾选「写入内置黄金集」初始化）")
        else:
            st.dataframe(datasets, use_container_width=True, hide_index=True)

    with tab2:
        try:
            badcases = store.list_badcases(limit=100)
        except Exception:
            badcases = []
        if not badcases:
            st.caption("暂无 badcase")
        else:
            st.dataframe(badcases, use_container_width=True, hide_index=True)

    with tab3:
        try:
            runs = store.list_runs(limit=50)
        except Exception:
            runs = []
        if not runs:
            st.caption("暂无运行记录")
        else:
            st.dataframe(runs, use_container_width=True, hide_index=True)


# ==================== 5. 对外入口 ====================
def render_monitor_page() -> None:
    """渲染监控大盘（系统运行指标 + 模型效果指标 + 评测库）"""
    if st is None:
        print("需要 streamlit 才能渲染监控大盘：pip install streamlit")
        return

    st.markdown('<p class="main-title">[MONITOR] 可观测性监控大盘</p>',
                unsafe_allow_html=True)
    st.caption("系统运行指标（实时埋点） + 模型效果指标（离线评测） + 测试集/badcase 管理")

    tab_runtime, tab_eval, tab_store = st.tabs(
        ["[CHART] 运行指标", "[TARGET] 效果指标", "[DB] 评测数据"]
    )
    with tab_runtime:
        _render_runtime_metrics()
    with tab_eval:
        _render_offline_eval()
    with tab_store:
        _render_eval_store()


# 支持 streamlit run monitor_view.py 单独预览
if __name__ == "__main__":  # pragma: no cover
    if st is not None:
        try:
            st.set_page_config(page_title="AI CRM 监控大盘", page_icon="[MONITOR]", layout="wide")
        except Exception:
            pass
        render_monitor_page()
