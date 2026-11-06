# -*- coding: utf-8 -*-
"""
置信度（confidence）离线测算脚本
================================
文档口径（附 A+ · M.5）：置信度不是模型自报的一个数，而是由三层可观测信号组合：
  ① 向量检索的【相关性分数】——过阈比例（>= RETRIEVAL_SCORE_THRESHOLD=0.5）与分位数（P10/P50）
  ② CrossEncoder 重排的【区分度】——Top1 与 Top2 的分差（分差小 = 不确定 = 倾向兜底）
  ③ 意图的【规则命中数】是否达 min_match（避免误报）

本脚本把这三层分别量化成可复现的数字：
  - ① 对本地离线语料，用真实 BGE 嵌入算余弦相关性分数，统计过阈（>=0.5）比例与 P10/P50；
  - ② 用真实 CrossEncoder 精排，算每个 query 的 Top1-Top2 分差，再取均值；
  - ③ 复用 eval_metrics 的意图规则口径，输出规则命中率。

【为什么全程离线】
  rag_law.BGEEmbeddings._load_model() 调 SentenceTransformer 时没传 local_files_only=True，
  无网 / 弱网下会去 HuggingFace 发 HEAD 探测请求（WinError 10060，每文件重试 5 次）而长时间卡住；
  当前 transformers / huggingface_hub 版本又不吃 HF_HUB_OFFLINE 环境变量。
  因此本脚本显式 local_files_only=True（已验证本地缓存模型秒级加载），并直接把文档向量本地算余弦，
  绕开 langchain_chroma 的 relevance 打分（它对本项目 384 维旧向量会报维度不匹配）。

【口径声明（必须一起讲，否则数字会被打回）】
  - 嵌入模型：本地 BAAI/bge-small-zh-v1.5（512 维，离线加载）
  - 重排模型：优先 config.RERANKER_MODEL；未被缓存时回退到本地已缓存的备选（报告里标 reranker_model）
  - 语料：本地离线 K12 语料（内置 ~14 块）；样本量小，仅用于「口径自检 / 相对趋势」，
    不代表线上真实知识库效果（线上知识库体量更大，扩容后召回通常衰减，需持续监测）
  - 同时附「现有本地 Chroma 知识库现状」诊断（块数 / 向量维度是否与本地 BGE 匹配）

用法：
  python -u eval_confidence.py                 # 真实离线口径（推荐）
  python -u eval_confidence.py --synthetic     # 强制走合成语料自检（零模型依赖）
  python -u eval_confidence.py --k 5           # 每个 query 的候选数（默认 5）
输出：
  - 控制台摘要
  - confidence_report.json（结构化报告，便于留档 / 回归对比）
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple


# 关键：必须在导入 sentence-transformers / huggingface_hub 之前设置，
# 否则加载模型时会去 HuggingFace 探测（HEAD 请求），弱网 / 无网时长时间卡住。
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


# ---------------------------------------------------------------- 基础工具
def _force_utf8() -> None:
    """让 Windows 控制台尽量以 utf-8 输出"""
    for stream in ("stdout", "stderr"):
        try:
            getattr(sys, stream).reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except Exception:
            pass


def _nearest_rank(sorted_vals: Sequence[float], p: float) -> float:
    """最近秩法分位数（与 loadtest.py 的 _percentile 同口径）"""
    if not sorted_vals:
        return 0.0
    k = max(1, math.ceil(p * len(sorted_vals)))
    return float(sorted_vals[min(k, len(sorted_vals)) - 1])


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


# ---------------------------------------------------------------- 样本集
# 本地离线 K12 语料（产品 / 规则知识，作为置信度口径自检语料；非线上知识库）
LOCAL_K12_CORPUS: List[Tuple[str, str]] = [
    ("k12_01", "初三数学一对一课程按小时计费，课时费 300 元/课时，支持按月购买与一对一定制方案。"),
    ("k12_02", "小学数学小班（6 人）课时费 150 元/课时，每期 40 课时，适合基础巩固。"),
    ("k12_03", "高中英语一对一课时费 380 元/课时，师资为师范院校英语专业，平均教龄 5 年以上。"),
    ("k12_04", "初中物理一对一课时费 320 元/课时，可先安排试听再决定是否报名。"),
    ("k12_05", "退费政策：开课前可全额退费；开课后按已消耗课时扣除，剩余部分 7 个工作日内原路退回。"),
    ("k12_06", "试听政策：新客户可免费试听 1 次（40 分钟），试听后 3 日内报名可抵扣首期课时费 100 元。"),
    ("k12_07", "报名流程：确认班型与课时 → 签订合同 → 支付 → 分配班主任 → 排课可开始上课。"),
    ("k12_08", "师资说明：授课老师均为重点师范院校毕业，平均教龄 5 年以上，入职需通过试讲与考核。"),
    ("k12_09", "班型对照：一对一 40 课时/期；小班（6 人）40 课时/期；大班（20 人）40 课时/期。"),
    ("k12_10", "课时费会根据家长意向程度与报课周期做浮动调整，标准价以外以最终合同为准。"),
    ("k12_11", "课程产品资料与内部标准定价存放于知识库，客户实际成交价属于用户画像工单数据。"),
    ("k12_12", "跟进话术生成：结合 RAG 产品资料与客户画像（年级、意向、预算）生成个性化沟通话术。"),
    ("k12_13", "客户画像字段：姓名、手机号、年龄、身份（学生/家长）、意向课程、意向价格、课时费、备注。"),
    ("k12_14", "异常客户筛查规则：超过 14 天未跟进、明确退费诉求、长期无产品使用记录会进入待跟进清单。"),
]


def build_queries() -> List[str]:
    """
    构造置信度测算用的 query 集：
      = 黄金检索集（GOLDEN_RETRIEVAL） + 意图黄金集里「检索类（rag_retrieve）」的 query
    去重后返回，保证样本量比单用 5 条检索集更稳。
    """
    from eval_metrics import GOLDEN_RETRIEVAL, GOLDEN_INTENT

    queries: List[str] = [c["query"] for c in GOLDEN_RETRIEVAL]
    for q, expected in GOLDEN_INTENT:
        if expected == "rag_retrieve":
            queries.append(q)

    seen = set()
    uniq: List[str] = []
    for q in queries:
        if q not in seen:
            seen.add(q)
            uniq.append(q)
    return uniq


# ---------------------------------------------------------------- 模型（离线）
def _load_embedder():
    """离线加载本地 BGE 嵌入模型（失败抛异常，由上层降级）"""
    from sentence_transformers import SentenceTransformer
    from config import LOCAL_EMBEDDING_MODEL

    return LOCAL_EMBEDDING_MODEL, SentenceTransformer(LOCAL_EMBEDDING_MODEL, local_files_only=True)


def _load_reranker():
    """离线加载 CrossEncoder 重排模型；优先 config，其次本地已缓存的备选；都失败返回 (None, None)"""
    try:
        from sentence_transformers import CrossEncoder
        from config import RERANKER_MODEL
    except Exception as e:
        print(f"[WARN] 重排依赖不可用（{e}），② 信号将跳过。")
        return None, None

    candidates = [RERANKER_MODEL, "BAAI/bge-reranker-base", "cross-encoder/ms-marco-MiniLM-L-6-v2"]
    seen = set()
    for name in candidates:
        if not name or name in seen:
            continue
        seen.add(name)
        try:
            return name, CrossEncoder(name, max_length=512, local_files_only=True)
        except Exception:
            continue
    print("[WARN] 本地未缓存任何可用重排模型，② 重排区分度信号将跳过。")
    return None, None


def _cosine_scores(query: str, prefix: str, model, doc_vecs, docs: Sequence[str], k: int) -> List[Tuple[str, float]]:
    """用归一化向量点积算余弦相关性分数，取 Top-K（等价于余弦距离下的 relevance = 1 - distance）"""
    qv = model.encode(prefix + query, normalize_embeddings=True)
    scored = [
        (float(sum(float(a) * float(b) for a, b in zip(qv, dv))), docs[i])
        for i, dv in enumerate(doc_vecs)
    ]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [(text, score) for score, text in scored[:k]]


# ---------------------------------------------------------------- 现有 Chroma 知识库诊断
def diagnose_local_kb() -> Dict:
    """诊断现有本地 Chroma 知识库：块数 / 向量维度 / 与本地 BGE 维度是否匹配"""
    info: Dict = {"persist_dir": None, "exists": False, "collection": None, "count": None, "dim": None}
    try:
        from config import CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME
        info["persist_dir"] = CHROMA_PERSIST_DIR
        info["exists"] = os.path.isdir(CHROMA_PERSIST_DIR)
        info["collection"] = CHROMA_COLLECTION_NAME
        if info["exists"]:
            import chromadb

            col = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR).get_collection(CHROMA_COLLECTION_NAME)
            info["count"] = int(col.count())
            embs = col.get(include=["embeddings"]).get("embeddings")
            shape = getattr(embs, "shape", None)
            info["dim"] = int(shape[1]) if shape is not None and len(shape) == 2 else None
    except Exception as e:
        info["error"] = str(e)
    return info


# ---------------------------------------------------------------- ① / ② 真实离线口径
def run_real(k: int) -> Dict:
    """真实离线口径：本地 BGE 相关性命中 + 真实 CrossEncoder 重排区分度"""
    from config import RETRIEVAL_SCORE_THRESHOLD

    emb_name, model = _load_embedder()
    rk_name, reranker = _load_reranker()

    prefix = "为这个句子生成表示以用于检索相关文章："  # BGE 官方 query 指令前缀
    docs = [t for _, t in LOCAL_K12_CORPUS]
    doc_vecs = model.encode(docs, normalize_embeddings=True)

    queries = build_queries()
    per_query: List[Dict] = []
    all_scores: List[float] = []
    gaps: List[float] = []
    n_pass = n_total = 0

    for q in queries:
        try:
            pairs = _cosine_scores(q, prefix, model, doc_vecs, docs, k)
        except Exception as e:
            per_query.append({"query": q, "error": str(e)})
            continue

        scores = [round(s, 4) for _, s in pairs]
        all_scores.extend(scores)
        n_pass += sum(1 for s in scores if s >= RETRIEVAL_SCORE_THRESHOLD)
        n_total += len(scores)

        # ② 重排区分度：Top1 - Top2（分差越小越不确定，越倾向兜底）
        gap: Optional[float] = None
        if reranker is not None and len(pairs) >= 2:
            try:
                raw = reranker.predict(
                    [[q, t] for t, _ in pairs], batch_size=32, show_progress_bar=False
                )
                rs = sorted((float(x) for x in raw), reverse=True)
                if len(rs) >= 2:
                    gap = rs[0] - rs[1]
                    gaps.append(gap)
            except Exception:
                gap = None

        per_query.append({
            "query": q,
            "scores": scores,
            "pass": sum(1 for s in scores if s >= RETRIEVAL_SCORE_THRESHOLD),
            "returned": len(scores),
            "top1": scores[0] if scores else None,
            "rerank_gap": None if gap is None else round(gap, 4),
        })

    summary = _summarize("real", RETRIEVAL_SCORE_THRESHOLD, len(LOCAL_K12_CORPUS), per_query, all_scores, n_pass, n_total, gaps)
    summary["embedding_model"] = emb_name
    summary["reranker_model"] = rk_name
    summary["corpus"] = "local_k12_offline"
    summary["kb_diagnosis"] = diagnose_local_kb()
    return summary


def run_synthetic(k: int, threshold: float = 0.5) -> Dict:
    """
    合成语料口径：用 eval_metrics 内置合成语料 + 字符 n-gram 重合度打分。
    仅验证「过阈比例 / 分位 / 分差」口径实现是否正确，不可当真实库效果（零模型依赖）。
    """
    from eval_metrics import SYNTHETIC_CORPUS, _char_ngrams

    per_query: List[Dict] = []
    all_scores: List[float] = []
    gaps: List[float] = []
    n_pass = n_total = 0

    for q in build_queries():
        qgrams = _char_ngrams(q)
        scored: List[Tuple[float, str]] = []
        for _doc_id, text in SYNTHETIC_CORPUS:
            dgrams = _char_ngrams(text)
            score = len(qgrams & dgrams) / len(qgrams) if qgrams else 0.0
            scored.append((score, text))
        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:k]
        scores = [round(float(s), 4) for s, _ in top]
        all_scores.extend(scores)
        n_pass += sum(1 for s in scores if s >= threshold)
        n_total += len(scores)
        gap = None
        if len(scores) >= 2:
            gap = round(scores[0] - scores[1], 4)
            gaps.append(scores[0] - scores[1])
        per_query.append({
            "query": q,
            "scores": scores,
            "pass": sum(1 for s in scores if s >= threshold),
            "returned": len(scores),
            "top1": scores[0] if scores else None,
            "rerank_gap": gap,
        })

    summary = _summarize("synthetic", threshold, len(SYNTHETIC_CORPUS), per_query, all_scores, n_pass, n_total, gaps)
    summary["embedding_model"] = "char_ngram(no-model)"
    summary["reranker_model"] = "score_gap(no-model)"
    summary["corpus"] = "synthetic_ngram"
    return summary


def _summarize(
    mode: str,
    threshold: float,
    kb_count: Optional[int],
    per_query: List[Dict],
    all_scores: List[float],
    n_pass: int,
    n_total: int,
    gaps: List[float],
) -> Dict:
    sorted_scores = sorted(all_scores)
    return {
        "mode": mode,
        "threshold": threshold,
        "kb_count": kb_count,
        "query_count": len(per_query),
        "chunk_total": n_total,
        "chunk_pass": n_pass,
        "pass_rate": round(n_pass / n_total, 4) if n_total else 0.0,
        "score_p10": round(_nearest_rank(sorted_scores, 0.10), 4),
        "score_p50": round(_nearest_rank(sorted_scores, 0.50), 4),
        "score_mean": round(_mean(all_scores), 4),
        "rerank_gap_mean": round(_mean(gaps), 4),
        "rerank_gap_n": len(gaps),
        "per_query": per_query,
    }


# ---------------------------------------------------------------- ③ 意图命中信号
def intent_signal() -> Dict:
    """③ 意图规则命中信号：复用 eval_metrics 的口径（规则命中率）"""
    from eval_metrics import eval_intent, GOLDEN_INTENT

    report = eval_intent(GOLDEN_INTENT)
    return {
        "total": report["total"],
        "rule_hit": report["rule_hit"],
        "rule_hit_rate": report["rule_hit_rate"],
        "rule_accuracy": report["rule_accuracy"],
        "e2e_accuracy": report["e2e_accuracy"],
    }


# ---------------------------------------------------------------- 主流程
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="置信度三层信号离线测算")
    p.add_argument("--k", type=int, default=5, help="每个 query 的候选数（默认 5）")
    p.add_argument("--synthetic", action="store_true", help="强制走合成语料自检（零模型依赖）")
    p.add_argument("--out", type=str, default="confidence_report.json", help="JSON 报告输出路径")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    _force_utf8()
    args = parse_args(argv)

    signals: Dict = {}
    if args.synthetic:
        signals = run_synthetic(args.k)
    else:
        try:
            signals = run_real(args.k)
            if signals.get("chunk_total", 0) == 0:
                raise RuntimeError("真实离线口径未产出任何片段，降级合成语料")
        except Exception as e:
            print(f"[WARN] 真实离线口径不可用（{e}），降级为合成语料自检。")
            signals = run_synthetic(args.k)
            signals["fallback_reason"] = str(e)

    signals["intent_signal"] = intent_signal()

    report = {"generated_at": None, "signals": signals}
    try:
        import datetime

        report["generated_at"] = datetime.datetime.now().isoformat(timespec="seconds")
    except Exception:
        pass

    with io.open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)

    s = signals
    it = s["intent_signal"]
    kb = (s.get("kb_diagnosis") or {}).get("count")
    print("=" * 70)
    print(f"置信度三层信号测算（口径={s['mode']}，语料={s.get('corpus')}，"
          f"块数={s['kb_count']}，query={s['query_count']}，k={args.k}）")
    print(f"  嵌入模型={s.get('embedding_model')}  重排模型={s.get('reranker_model')}")
    print(f"① 检索相关性过阈比例（>= {s['threshold']}）：{s['pass_rate'] * 100:.1f}%"
          f"（{s['chunk_pass']}/{s['chunk_total']} 片段）")
    print(f"   相关性分数 P10={s['score_p10']}  P50={s['score_p50']}  mean={s['score_mean']}")
    print(f"② 重排 Top1-Top2 平均分差：{s['rerank_gap_mean']}（{s['rerank_gap_n']} 个 query）")
    print(f"③ 意图规则命中率：{it['rule_hit_rate'] * 100:.1f}%（{it['rule_hit']}/{it['total']}），"
          f"规则口径准确率 {it['rule_accuracy'] * 100:.1f}%，端到端 {it['e2e_accuracy'] * 100:.1f}%")
    if kb is not None:
        print(f"   注：本地 Chroma 现有知识库块数={kb}（同机诊断，仅作背景）")
    print(f"报告已写入：{args.out}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
