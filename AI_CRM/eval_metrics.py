"""
eval_metrics.py - 离线评估脚本（对应二面整改「评估脚本」）
==========================================================
目标：把「我觉得效果不错」变成「可复现的数字」，支撑二面四段式回答的
      「实际效果」与「不足与优化」两段。

评估对象：
  1) 意图分类（指令路由）：Accuracy / 各类别混淆 / badcase 清单
  2) RAG 检索：Recall@K / Hit@K / MRR@K / badcase 清单
  3) 可选：结构化意图安全用例（越权字段 / 未授权 action 是否被拦截）

设计原则（无 API Key 也能跑）：
  - 意图评估：直接复用 config.INTENT_RULES 的规则匹配（零成本、零网络）
  - 检索评估：--self-test 使用内置合成语料（字符 n-gram 检索），验证指标实现
  - 检索评估：--retrieval 可选接入真实 RAG（Chroma + BM25），失败自动降级

用法：
  # 1) 自检（推荐先跑，验证指标与脚本正确性，无需外部依赖）
  python eval_metrics.py --self-test
  # 2) 仅评估意图路由（复用规则，无需 API Key）
  python eval_metrics.py --intent
  # 3) 检索评估（需已构建知识库）
  python eval_metrics.py --retrieval --k 5
  # 4) 全量 + 结构化意图安全用例 + 输出 JSON 报告
  python eval_metrics.py --intent --retrieval --safety --json eval_report.json
  # 5) 生成质量（groundedness / 幻觉率）
  python eval_metrics.py --groundedness
  # 6) 从评测数据管理库（eval_store.py）读取测试集 + badcase 落库
  python eval_metrics.py --seed-store --from-store --save-badcases
"""

# ==================== 1. 导入依赖 ====================
import argparse
import json
import os
import re
import sys
from typing import Callable, Dict, List, Optional, Sequence, Set


# ==================== 2. 通用指标实现 ====================
def recall_at_k(relevant: Set[str], retrieved: Sequence[str], k: int) -> float:
    """
    Recall@K = 命中相关数 / 相关总数（按标签集合计算，去重）
    ========================================================
    参数：
      relevant  : 期望命中的相关标签集合
      retrieved : 检索返回的标签序列（按相关性降序）
      k         : 截断位置
    """
    if not relevant:
        return 0.0
    top = retrieved[:k]
    hits = sum(1 for item in top if item in relevant)
    return hits / len(relevant)


def hit_at_k(relevant: Set[str], retrieved: Sequence[str], k: int) -> float:
    """Hit@K = Top-K 中是否至少命中 1 个相关项（0/1）"""
    return 1.0 if any(item in relevant for item in retrieved[:k]) else 0.0


def mrr_at_k(relevant: Set[str], retrieved: Sequence[str], k: int) -> float:
    """
    MRR@K = 第一个相关项排名的倒数（越靠前越高）
    ==============================================
    例：第 1 名命中 → 1.0；第 3 名命中 → 1/3 ≈ 0.333
    """
    for rank, item in enumerate(retrieved[:k], 1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


# ==================== 3. 意图评估（规则路由，零依赖） ====================
# 黄金集：K12 教育业务场景下「该走哪条链路」的人工标注
GOLDEN_INTENT: List[tuple] = [
    # —— rag_retrieve（课程产品资料）——
    ("初三数学一对一课时费大概多少？", "rag_retrieve"),
    ("你们都有哪些班型？", "rag_retrieve"),
    ("高一英语小班怎么报名", "rag_retrieve"),
    ("课程体系是怎么设置的", "rag_retrieve"),
    ("老学员续费有优惠吗", "rag_retrieve"),
    ("退费流程是什么", "rag_retrieve"),
    ("中考冲刺班和普通班区别", "rag_retrieve"),
    # —— web_search（实时信息）——
    ("北京今天天气怎么样", "web_search"),
    ("最近有什么新闻", "web_search"),
    ("A股大盘走势如何", "web_search"),
    # —— direct_llm（普通闲聊）——
    ("你好", "direct_llm"),
    ("谢谢你", "direct_llm"),
    ("再见", "direct_llm"),
]


def classify_by_rules(query: str, rules: Optional[dict] = None) -> Optional[str]:
    """
    规则意图分类（与 agent._rule_based_intent 同源口径）
    =====================================================
    返回：命中意图名 / None（未命中，线上将转 LLM 兜底）
    """
    if rules is None:
        from config import INTENT_RULES  # 延迟导入，避免无配置时崩溃
        rules = INTENT_RULES
    for intent, rule in rules.items():
        keywords = rule.get("keywords", [])
        min_match = rule.get("min_match", 1)
        matched = sum(1 for kw in keywords if kw in query)
        if matched >= min_match:
            return intent
    return None


def eval_intent(cases: Sequence[tuple] = None, fallback_intent: str = "direct_llm") -> dict:
    """
    评估意图路由（两套口径，对应文档「两套并列」）
    ================================================
      - rule_accuracy : 仅统计规则命中的样本（衡量规则本身是否准）
      - e2e_accuracy  : 规则未命中 → 视为交 LLM 兜底（默认 direct_llm），
                        衡量线上真实路由的预期准确率
    规则命中但类别错误 → badcase；未命中不算规则错误，但计入兜底口径。
    """
    cases = list(cases or GOLDEN_INTENT)
    confusion: Dict[str, Dict[str, int]] = {}
    badcases: List[dict] = []
    rule_hit = 0
    rule_correct = 0
    e2e_correct = 0

    for query, expected in cases:
        rule_pred = classify_by_rules(query)
        if rule_pred is not None:
            rule_hit += 1
            effective = rule_pred
            if rule_pred == expected:
                rule_correct += 1
        else:
            effective = fallback_intent  # 线上：未命中 → LLM 兜底

        confusion.setdefault(expected, {})
        confusion[expected][effective] = confusion[expected].get(effective, 0) + 1

        if effective == expected:
            e2e_correct += 1
        else:
            badcases.append({
                "query": query,
                "expected": expected,
                "rule_pred": rule_pred or "unmatched",
                "effective": effective,
            })

    total = len(cases)
    return {
        "total": total,
        "rule_hit": rule_hit,
        "rule_hit_rate": round(rule_hit / total, 4) if total else 0.0,
        "rule_correct": rule_correct,
        "rule_accuracy": round(rule_correct / rule_hit, 4) if rule_hit else 0.0,
        "e2e_correct": e2e_correct,
        "e2e_accuracy": round(e2e_correct / total, 4) if total else 0.0,
        "fallback_intent": fallback_intent,
        "confusion": confusion,
        "badcases": badcases,
    }


# ==================== 4. 检索评估 ====================
# 黄金集：query + 期望命中的关键词（用于关键词相关性判定）
GOLDEN_RETRIEVAL: List[dict] = [
    {"query": "初三数学一对一的课时费是多少", "expected_keywords": ["课时费", "一对一"]},
    {"query": "有没有中考冲刺班", "expected_keywords": ["中考", "冲刺班"]},
    {"query": "你们师资力量怎么样", "expected_keywords": ["师资", "教师"]},
    {"query": "续费有什么优惠", "expected_keywords": ["续费", "优惠"]},
    {"query": "退费怎么退", "expected_keywords": ["退费"]},
]

# 自检用合成语料（模拟 K12 课程产品资料，单篇 ≤500 字）
SYNTHETIC_CORPUS: List[tuple] = [
    ("doc1", "初三数学一对一课程说明：课时费按小时计费，一对一定制方案，可按月购买。"),
    ("doc2", "中考冲刺班介绍：面向初三学生，中考前强化冲刺，含数学、英语、物理。"),
    ("doc3", "师资介绍：授课教师均为重点师范院校毕业，平均教龄5年以上。"),
    ("doc4", "续费优惠政策：老学员续费享受9折优惠，多科联报再减。"),
    ("doc5", "班型说明：提供一对一、小班（6人）、大班等多种班型。"),
    ("doc6", "退费流程：未开课时可全额退费，已开课按剩余课时比例退费。"),
]


def _char_ngrams(text: str, n: int = 2) -> Set[str]:
    """字符 n-gram（中文无需分词，零外部依赖）"""
    text = re.sub(r"\s+", "", text or "")
    if len(text) <= n:
        return {text} if text else set()
    return {text[i:i + n] for i in range(len(text) - n + 1)}


def make_synthetic_retriever(corpus: Sequence[tuple]) -> Callable[[str, int], List[str]]:
    """
    构造合成检索器（字符 n-gram 重合度打分）
    ========================================
    返回：retriever(query, k) -> List[文档文本]
    """
    def _retrieve(query: str, k: int) -> List[str]:
        qgrams = _char_ngrams(query)
        scored = []
        for _doc_id, text in corpus:
            dgrams = _char_ngrams(text)
            score = len(qgrams & dgrams) / len(qgrams) if qgrams else 0.0
            scored.append((score, text))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [text for score, text in scored[:k] if score > 0]
    return _retrieve


def make_rag_retriever() -> Callable[[str, int], List[str]]:
    """
    构造真实 RAG 检索器（Chroma + BM25 自适应检索）
    ================================================
    初始化失败时抛出异常，由调用方降级。
    """
    from rag_law import RAGLaw

    rag = RAGLaw()  # 首次会加载嵌入模型与向量库（可能较慢）

    def _retrieve(query: str, k: int) -> List[str]:
        try:
            return rag.adaptive_search(query, max_results=k)
        except Exception:
            return rag.similarity_search(query, k=k)

    return _retrieve


def eval_retrieval(
    cases: Sequence[dict],
    retriever: Callable[[str, int], List[str]],
    k: int = 5,
) -> dict:
    """
    评估检索质量（关键词相关性判定）
    ================================
    判定口径：
      - 每篇返回文档「覆盖」其包含的期望关键词
      - Recall@K = Top-K 覆盖关键词数 / 期望关键词总数
      - Hit@K    = Top-K 是否覆盖任意关键词
      - MRR@K    = 首个「覆盖任意关键词」文档排名的倒数
    """
    cases = list(cases)
    recalls, hits, mrrs = [], [], []
    badcases: List[dict] = []
    details: List[dict] = []

    for case in cases:
        query = case["query"]
        expected: List[str] = case.get("expected_keywords", [])
        expected_set = set(expected)

        try:
            docs = retriever(query, k) or []
        except Exception as e:  # 单条失败不影响整体评估
            docs = []
            detail = {"query": query, "error": str(e)}
            details.append(detail)
            badcases.append({"query": query, "reason": f"检索异常: {e}"})
            recalls.append(0.0)
            hits.append(0.0)
            mrrs.append(0.0)
            continue

        covered: Set[str] = set()
        first_rank = None
        for rank, doc in enumerate(docs[:k], 1):
            matched = {kw for kw in expected_set if kw in (doc or "")}
            if matched:
                covered |= matched
                if first_rank is None:
                    first_rank = rank

        recall = len(covered) / len(expected_set) if expected_set else 0.0
        hit = 1.0 if covered else 0.0
        mrr = (1.0 / first_rank) if first_rank else 0.0

        recalls.append(recall)
        hits.append(hit)
        mrrs.append(mrr)

        detail = {
            "query": query,
            "expected_keywords": expected,
            "covered": sorted(covered),
            "recall": round(recall, 4),
            "hit": hit,
            "mrr": round(mrr, 4),
            "top_docs": [d[:60] for d in docs[:k]],
        }
        details.append(detail)

        if recall < 1.0:
            badcases.append({
                "query": query,
                "missing_keywords": sorted(expected_set - covered),
                "recall": round(recall, 4),
            })

    return {
        "k": k,
        "total": len(cases),
        "recall_at_k": round(_mean(recalls), 4),
        "hit_at_k": round(_mean(hits), 4),
        "mrr_at_k": round(_mean(mrrs), 4),
        "badcases": badcases,
        "details": details,
    }


# ==================== 5. 结构化意图安全用例 ====================
# 覆盖二面「Prompt 不是安全边界」：越权/高风险操作是否被服务端拦截
SAFETY_CASES: List[dict] = [
    {
        "name": "越权表访问（应被拒）",
        "intent": {"action": "query_customer", "field": "sqlite_master", "value": "1"},
        "expect": "reject",
    },
    {
        "name": "未授权 action（应被拒）",
        "intent": {"action": "drop_table", "customer_id": "c1"},
        "expect": "reject",
    },
    {
        "name": "更新未授权字段（应被拒）",
        "intent": {"action": "update_customer", "customer_id": "c1",
                   "field": "is_admin", "value": "1"},
        "expect": "reject",
    },
    {
        "name": "高风险字段转人工审核（不直接落库）",
        "intent": {"action": "update_customer", "customer_id": "c1",
                   "field": "hourly_fee", "value": "999"},
        "expect": "review",
    },
    {
        "name": "低风险字段允许更新",
        "intent": {"action": "update_customer", "customer_id": "c1",
                   "field": "school_grade", "value": "高一"},
        "expect": "allow",
    },
]


def eval_safety(cases: Sequence[dict] = None) -> dict:
    """
    评估结构化意图链路的安全行为
    ============================
    仅做「校验层」判定（不真正写库）：
      reject → validate_structured_intent 返回错误
      review → 校验通过 且 字段属于 MANUAL_REVIEW_FIELDS
      allow  → 校验通过 且 字段属于 AUTO_UPDATE_FIELDS
    """
    from sql_agent import validate_structured_intent
    from models import MANUAL_REVIEW_FIELDS, AUTO_UPDATE_FIELDS

    cases = list(cases or SAFETY_CASES)
    results, badcases = [], []
    passed = 0

    for case in cases:
        intent = case["intent"]
        expect = case["expect"]
        error = validate_structured_intent(intent)

        if error:
            actual = "reject"
        else:
            field = intent.get("field")
            if field in MANUAL_REVIEW_FIELDS:
                actual = "review"
            elif field in AUTO_UPDATE_FIELDS or field is None:
                actual = "allow"
            else:
                actual = "allow"

        ok = actual == expect
        passed += 1 if ok else 0
        results.append({
            "name": case["name"],
            "expect": expect,
            "actual": actual,
            "pass": ok,
            "error": error,
        })
        if not ok:
            badcases.append(results[-1])

    return {
        "total": len(cases),
        "passed": passed,
        "accuracy": round(passed / len(cases), 4) if cases else 0.0,
        "results": results,
        "badcases": badcases,
    }


# ==================== 5b. 生成质量：groundedness / 幻觉率 ====================
# 口径说明（面试要能讲清楚）：
#   幻觉率 / groundedness 的本质是「回答是否被检索证据支撑」，严格判定需要一个裁判。
#   本模块提供两条路径：
#     1) 默认：词面支撑度（lexical grounding）—— 零成本、可复现，适合离线回归；
#        判定口径 = 回答的每个句子能否在检索上下文中找到足够重合的字符 n-gram。
#     2) 可选：传入 judge(answer_sentence, context) -> bool 回调（接入 LLM-as-a-Judge），
#        做更贴近语义的判定；本脚本不强制依赖任何模型。
#   注意：词面法会低估「同义改写」的支撑度，属于**代理指标**，报告中会明确标注 judge 类型。

# 内置生成质量用例：query / context(检索证据) / answer(模型回答) / key_points(必答要点)
GOLDEN_GROUNDED: List[dict] = [
    {
        "query": "初三数学一对一的课时费是多少",
        "context": "初三数学一对一课程说明：课时费按小时计费，一对一定制方案，可按月购买。",
        "answer": "初三数学一对一按小时计费，支持一对一定制方案，也可以按月购买。",
        "key_points": ["按小时计费", "一对一", "按月购买"],
    },
    {
        "query": "退费怎么退",
        "context": "退费流程：未开课时可全额退费，已开课按剩余课时比例退费。",
        "answer": "未开课可以全额退费；如果已经开课，则按剩余课时比例退费。另外我们还支持7天无理由退款。",
        "key_points": ["全额退费", "按剩余课时比例退费"],
    },
]


def _split_sentences(text: str) -> List[str]:
    """中英文混合切句（按句末标点与换行切分，过滤空句）"""
    parts = re.split(r"[。！？!?\n；;]+", text or "")
    return [p.strip() for p in parts if p and p.strip()]


def _support_ratio(sentence: str, context: str, n: int = 2) -> float:
    """
    句子被上下文支撑的程度 = 句子字符 n-gram 在上下文中出现的比例
    取值范围 [0,1]；1 表示句子每个 n-gram 都能在证据里找到。
    """
    sgrams = _char_ngrams(sentence, n)
    if not sgrams:
        return 0.0
    cgrams = _char_ngrams(context, n)
    return len(sgrams & cgrams) / len(sgrams)


def eval_groundedness(
    cases: Sequence[dict] = None,
    support_threshold: float = 0.5,
    judge: Optional[Callable[[str, str], bool]] = None,
) -> dict:
    """
    评估生成质量（groundedness / 幻觉率）
    =====================================
    指标口径：
      · groundedness      = 被证据支撑的句子数 / 回答总句数
      · hallucination_rate = 1 - groundedness（未被证据支撑的句子占比）
      · key_point_coverage = 必答要点命中率（要点是否出现在回答里）
    单句判定：支撑度 ≥ support_threshold（或 judge 回调返回 True）。
    """
    cases = list(cases if cases is not None else GOLDEN_GROUNDED)
    g_scores, h_rates, kp_cov = [], [], []
    badcases: List[dict] = []
    details: List[dict] = []

    for case in cases:
        query = case.get("query", "")
        context = case.get("context", "")
        answer = case.get("answer", "")
        key_points = case.get("key_points", []) or []

        sentences = _split_sentences(answer)
        supported = 0
        unsupported_sentences: List[str] = []
        for sent in sentences:
            ok = False
            if judge is not None:
                try:
                    ok = bool(judge(sent, context))
                except Exception:
                    ok = _support_ratio(sent, context) >= support_threshold
            else:
                ok = _support_ratio(sent, context) >= support_threshold
            if ok:
                supported += 1
            else:
                unsupported_sentences.append(sent)

        total_sent = len(sentences)
        g = supported / total_sent if total_sent else 0.0
        h = 1.0 - g
        covered_kp = [kp for kp in key_points if kp in answer]
        cov = len(covered_kp) / len(key_points) if key_points else 0.0

        g_scores.append(g)
        h_rates.append(h)
        kp_cov.append(cov)

        details.append({
            "query": query,
            "sentences": total_sent,
            "groundedness": round(g, 4),
            "hallucination_rate": round(h, 4),
            "key_point_coverage": round(cov, 4),
            "unsupported_sentences": unsupported_sentences,
        })

        if h > 0:
            badcases.append({
                "query": query,
                "unsupported_sentences": unsupported_sentences,
                "hallucination_rate": round(h, 4),
            })

    return {
        "total": len(cases),
        "judge": "llm" if judge is not None else "lexical",
        "support_threshold": support_threshold,
        "groundedness": round(_mean(g_scores), 4),
        "hallucination_rate": round(_mean(h_rates), 4),
        "key_point_coverage": round(_mean(kp_cov), 4),
        "badcases": badcases,
        "details": details,
    }


# ==================== 5c. 对接评测数据管理库（eval_store） ====================
# 目标：用例不再写死在脚本里，改由 eval_store 管理版本；
#      评测产出的 badcase 也能自动落库，形成「评测 → 沉淀 → 回流」闭环。
def _open_store(path: Optional[str] = None):
    """惰性打开评测数据管理库；不可用时返回 None（保持脚本零外部依赖可跑）"""
    try:
        from eval_store import EvalStore
        return EvalStore(path)
    except Exception as e:
        print(f"[WARN] 评测数据管理库不可用，回退内置用例: {e}")
        return None


def seed_store(store=None) -> dict:
    """把内置黄金集（意图 / 检索 / 生成质量）写入评测数据管理库（已存在则跳过，避免重复）"""
    store = store or _open_store()
    if store is None:
        return {}

    ds_intent = store.ensure_dataset("intent_golden", "v1", "intent", "内置意图黄金集")
    if not store.get_cases(ds_intent):
        store.add_cases(ds_intent, [(q, t) for q, t in GOLDEN_INTENT], source="manual")

    ds_retrieval = store.ensure_dataset("retrieval_golden", "v1", "retrieval", "内置检索黄金集")
    if not store.get_cases(ds_retrieval):
        store.add_cases(
            ds_retrieval,
            [(c["query"], {"expected_keywords": c["expected_keywords"]})
             for c in GOLDEN_RETRIEVAL],
            source="manual",
        )

    ds_ground = store.ensure_dataset("grounded_golden", "v1", "e2e", "内置生成质量黄金集")
    if not store.get_cases(ds_ground):
        store.add_cases(
            ds_ground,
            [(c["query"], {"key_points": c.get("key_points", [])}) for c in GOLDEN_GROUNDED],
            source="manual",
        )

    return {
        "intent_golden": ds_intent,
        "retrieval_golden": ds_retrieval,
        "grounded_golden": ds_ground,
    }


def load_intent_cases_from_store(store=None) -> Optional[List[tuple]]:
    """从评测库读取意图用例；无数据返回 None（调用方回退内置黄金集）"""
    store = store or _open_store()
    if store is None:
        return None
    rows = store.load_dataset_cases("intent_golden")
    if not rows:
        return None
    valid = {"web_search", "rag_retrieve", "direct_llm"}
    cases: List[tuple] = []
    for r in rows:
        exp = r.get("expected")
        intent = exp.get("intent") if isinstance(exp, dict) else exp
        if intent in valid:
            cases.append((r["query"], intent))
    return cases or None


def load_retrieval_cases_from_store(store=None) -> Optional[List[dict]]:
    """从评测库读取检索用例；无数据返回 None"""
    store = store or _open_store()
    if store is None:
        return None
    rows = store.load_dataset_cases("retrieval_golden")
    if not rows:
        return None
    cases: List[dict] = []
    for r in rows:
        exp = r.get("expected") or {}
        kws = exp.get("expected_keywords") if isinstance(exp, dict) else None
        if kws:
            cases.append({"query": r["query"], "expected_keywords": kws})
    return cases or None


def save_badcases_to_store(report: dict, store=None) -> int:
    """把评测报告里的 badcase 落库（评测 → 沉淀闭环）"""
    store = store or _open_store()
    if store is None:
        return 0
    n = 0
    for bc in report.get("intent", {}).get("badcases", []):
        store.add_badcase(
            "intent_miss", bc.get("query", ""),
            expected=str(bc.get("expected", "")),
            actual=str(bc.get("effective", "")),
            source="manual", note="eval_intent",
        )
        n += 1
    for bc in report.get("retrieval", {}).get("badcases", []):
        store.add_badcase(
            "rag_empty", bc.get("query", ""),
            expected=str(bc.get("missing_keywords", [])),
            actual=str(bc.get("recall", "")),
            source="manual", note="eval_retrieval",
        )
        n += 1
    for bc in report.get("groundedness", {}).get("badcases", []):
        store.add_badcase(
            "hallucination", bc.get("query", ""),
            expected="", actual=str(bc.get("unsupported_sentences", [])),
            source="manual", note="eval_groundedness",
        )
        n += 1
    return n


# ==================== 6. 报告输出 ====================
def _print_header(title: str) -> None:
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)


def print_report(report: dict) -> None:
    """控制台打印评估报告"""
    if "intent" in report:
        _print_header("[1] 意图路由评估（两套口径）")
        m = report["intent"]
        print(f"样本数: {m['total']}  规则命中: {m['rule_hit']}({m['rule_hit_rate']:.2%})")
        print(f"  · 规则口径 rule_accuracy : {m['rule_accuracy']:.2%}（仅命中样本 {m['rule_correct']}/{m['rule_hit']}）")
        print(f"  · 端到端口径 e2e_accuracy: {m['e2e_accuracy']:.2%}（{m['e2e_correct']}/{m['total']}，未命中→LLM 兜底 {m['fallback_intent']}）")
        print("混淆矩阵（行=期望, 列=生效链路）:")
        for expected, row in m["confusion"].items():
            print(f"  {expected}: {row}")
        if m["badcases"]:
            print(f"[WARN] badcase（{len(m['badcases'])}）:")
            for bc in m["badcases"]:
                print(f"  - 「{bc['query']}」期望={bc['expected']} 规则={bc['rule_pred']} 生效={bc['effective']}")

    if "retrieval" in report:
        _print_header("[2] RAG 检索评估")
        m = report["retrieval"]
        print(f"样本数: {m['total']}  K: {m['k']}")
        print(f"Recall@{m['k']}: {m['recall_at_k']:.4f}")
        print(f"Hit@{m['k']}:    {m['hit_at_k']:.4f}")
        print(f"MRR@{m['k']}:    {m['mrr_at_k']:.4f}")
        if m["badcases"]:
            print(f"[WARN] badcase（{len(m['badcases'])}）:")
            for bc in m["badcases"]:
                print(f"  - 「{bc['query']}」缺失关键词={bc.get('missing_keywords')}")

    if "safety" in report:
        _print_header("[3] 结构化意图安全评估")
        m = report["safety"]
        print(f"用例数: {m['total']}  通过: {m['passed']}  通过率: {m['accuracy']:.2%}")
        for r in m["results"]:
            flag = "[PASS]" if r["pass"] else "[FAIL]"
            print(f"  {flag} {r['name']}: 期望={r['expect']} 实际={r['actual']}")

    if "groundedness" in report:
        _print_header("[4] 生成质量评估（groundedness / 幻觉率）")
        m = report["groundedness"]
        print(f"样本数: {m['total']}  判定方式: {m['judge']}  支撑阈值: {m['support_threshold']}")
        print(f"Groundedness（被证据支撑的比例）: {m['groundedness']:.4f}")
        print(f"Hallucination Rate（幻觉率）    : {m['hallucination_rate']:.4f}")
        print(f"Key Point Coverage（要点覆盖）  : {m['key_point_coverage']:.4f}")
        if m["badcases"]:
            print(f"[WARN] badcase（{len(m['badcases'])}）:")
            for bc in m["badcases"]:
                print(f"  - 「{bc['query']}」无支撑句子={bc.get('unsupported_sentences')}")


# ==================== 7. 入口 ====================
def build_report(args) -> dict:
    """按命令行参数构建评估报告"""
    report: dict = {}

    run_intent = args.intent or args.self_test
    run_retrieval = args.retrieval or args.self_test
    run_safety = args.safety or args.self_test
    run_grounded = getattr(args, "groundedness", False) or args.self_test

    # 可选：对接评测数据管理库（测试集版本管理 / badcase 沉淀）
    store = None
    if getattr(args, "from_store", False) or getattr(args, "save_badcases", False):
        store = _open_store()
    if getattr(args, "seed_store", False):
        store = store or _open_store()
        print(f"[OK] 已写入内置黄金集: {seed_store(store)}")

    if run_intent:
        try:
            intent_cases = load_intent_cases_from_store(store) if store else None
            report["intent"] = eval_intent(intent_cases)
            report["intent"]["source"] = "store" if intent_cases else "builtin"
        except Exception as e:
            print(f"[FAIL] 意图评估失败: {e}")

    if run_retrieval:
        retriever = None
        mode = None
        if args.self_test and not args.retrieval:
            retriever = make_synthetic_retriever(SYNTHETIC_CORPUS)
            mode = "synthetic"
        else:
            try:
                retriever = make_rag_retriever()
                mode = "rag"
            except Exception as e:
                print(f"[WARN] 真实 RAG 初始化失败，降级为合成语料自检: {e}")
                retriever = make_synthetic_retriever(SYNTHETIC_CORPUS)
                mode = "synthetic"
        try:
            retrieval_cases = load_retrieval_cases_from_store(store) if store else None
            report["retrieval"] = eval_retrieval(
                retrieval_cases or GOLDEN_RETRIEVAL, retriever, k=args.k
            )
            report["retrieval"]["mode"] = mode
            report["retrieval"]["source"] = "store" if retrieval_cases else "builtin"
        except Exception as e:
            print(f"[FAIL] 检索评估失败: {e}")

    if run_safety:
        try:
            report["safety"] = eval_safety()
        except Exception as e:
            print(f"[FAIL] 安全用例评估失败: {e}")

    if run_grounded:
        try:
            report["groundedness"] = eval_groundedness()
        except Exception as e:
            print(f"[FAIL] 生成质量评估失败: {e}")

    # 可选：把本次评测的 badcase 落库，供后续人工分诊 / 回流测试集
    if getattr(args, "save_badcases", False):
        n = save_badcases_to_store(report, store)
        print(f"[OK] badcase 已落库: {n} 条")

    return report


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CRM 离线评估脚本（意图 / 检索 / 结构化意图安全）"
    )
    parser.add_argument("--self-test", action="store_true",
                        help="自检全部指标实现（无需 API Key 与知识库）")
    parser.add_argument("--intent", action="store_true", help="评估意图路由（规则）")
    parser.add_argument("--retrieval", action="store_true", help="评估 RAG 检索（需知识库）")
    parser.add_argument("--safety", action="store_true", help="评估结构化意图安全用例")
    parser.add_argument("--groundedness", action="store_true",
                        help="评估生成质量（groundedness / 幻觉率）")
    parser.add_argument("--k", type=int, default=5, help="检索截断位置 K（默认 5）")
    parser.add_argument("--json", type=str, default=None, help="输出 JSON 报告路径")
    parser.add_argument("--from-store", action="store_true",
                        help="从评测数据管理库（eval_store.db）读取测试集")
    parser.add_argument("--seed-store", action="store_true",
                        help="把内置黄金集写入评测数据管理库")
    parser.add_argument("--save-badcases", action="store_true",
                        help="把本次评测的 badcase 落库（评测→沉淀闭环）")
    return parser.parse_args(argv)


def _ensure_utf8_stdout() -> None:
    """Windows 控制台默认 GBK，会导致中文乱码；尽量切到 UTF-8。"""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8")
            except Exception:
                pass


def main(argv: Optional[List[str]] = None) -> int:
    _ensure_utf8_stdout()
    args = parse_args(argv)

    # 默认（无任何参数）等价于 --self-test，保证开箱即跑
    if not any([args.self_test, args.intent, args.retrieval, args.safety]):
        args.self_test = True

    report = build_report(args)
    print_report(report)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n[OK] 报告已写入: {os.path.abspath(args.json)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
