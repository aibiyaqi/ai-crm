# -*- coding: utf-8 -*-
"""
test_follow_up.py - B.4「跟进闭环」自检脚本（L1 证据）
======================================================
覆盖链路：跟进记录 → 回写画像最近跟进时间 → 到期巡检 → 转化统计 → 埋点

设计要点（面试可当场复现）：
  1. 使用独立临时库（导入 config 前改 DB_SQLITE_PATH），不污染 crm.db
  2. 全部断言均为「代码行为」而非「人工声明」：
       - follow_ups 表 + 两个索引真实存在
       - save_follow_up 写入成功后，画像 last_follow_up 被回写
       - active_status（MANUAL_REVIEW_FIELDS）保持不被自动改写
       - list_due_follow_ups 两类口径（计划到期 / 超期未跟进）分别命中
       - 「跟过了就不再提醒」的复位逻辑成立
       - follow_up_conversion_stats 的分子/分母与「无数据返回 None」口径成立
       - metrics 埋点 follow_up_total / follow_up_converted_total 真实累加

运行：
  python test_follow_up.py
退出码：
  0 = 全部通过；1 = 存在失败用例
"""

# ==================== 0. 测试库隔离（必须在导入 config / memory_sqlite 之前）====================
import os
import sys
import shutil
import tempfile
from datetime import datetime, timedelta

_TMP_DIR = tempfile.mkdtemp(prefix="fu_selftest_")
os.environ["DB_SQLITE_PATH"] = os.path.join(_TMP_DIR, "test_follow_up.db")
os.environ.setdefault("DB_ENGINE", "sqlite")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ==================== 1. 导入被测模块 ====================
import metrics  # noqa: E402  可观测性埋点
from memory_sqlite import SQLiteMemory  # noqa: E402
from models import (  # noqa: E402
    CustomerProfile,
    FollowUp,
    FOLLOW_UP_CHANNELS,
    FOLLOW_UP_RESULTS,
    FOLLOW_UP_CONVERTED_RESULTS,
)

# ==================== 2. 极简断言框架 ====================
_PASSED = []
_FAILED = []


def check(name: str, cond: bool, extra: str = "") -> bool:
    """记录并打印一条断言结果"""
    if cond:
        _PASSED.append(name)
        print(f"  [PASS] {name}" + (f"  | {extra}" if extra else ""))
    else:
        _FAILED.append(name)
        print(f"  [FAIL] {name}" + (f"  | {extra}" if extra else ""))
    return bool(cond)


def section(title: str) -> None:
    print("\n" + "=" * 68)
    print(title)
    print("=" * 68)


# ==================== 3. 用例主流程 ====================
def main() -> int:
    now = datetime.now()
    db = SQLiteMemory()
    metrics.reset()

    # -------- 3.1 建表与索引 --------
    section("3.1 数据层：follow_ups 表 + 索引真实落库")
    db.cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='follow_ups'"
    )
    check("follow_ups 表已创建", db.cursor.fetchone() is not None)

    db.cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_follow_ups%'"
    )
    idx = {r[0] for r in db.cursor.fetchall()}
    check(
        "跟进索引已创建（客户维度 + 下次跟进时间维度）",
        {"idx_follow_ups_customer", "idx_follow_ups_next"} <= idx,
        f"实际={sorted(idx)}",
    )

    db.cursor.execute("PRAGMA table_info(follow_ups)")
    cols = {r[1] for r in db.cursor.fetchall()}
    expect_cols = {
        "follow_up_id", "customer_id", "advisor_id", "channel",
        "content", "result", "next_follow_up_at", "created_at",
    }
    check("跟进记录字段齐全（谁/何时/什么方式/什么结果/下次何时跟）",
          expect_cols <= cols, f"缺={sorted(expect_cols - cols)}")

    # -------- 3.2 造数据 --------
    section("3.2 造数据：3 位客户 + 3 条跟进记录")
    c1 = CustomerProfile(name="张三", phone="13800000001", active_status="活跃")
    c2 = CustomerProfile(name="李四", phone="13800000002")
    c3 = CustomerProfile(name="王五", phone="13800000003")
    for c in (c1, c2, c3):
        db.save_customer(c)

    # c1：填了「下次跟进时间」且已到点 → 应命中「计划到期」
    r1 = FollowUp(
        customer_id=c1.customer_id, channel="电话", content="沟通秋季班课包",
        result="有意向", next_follow_up_at=now - timedelta(days=2),
    )
    # c2：20 天前跟进过（阈值 14 天）→ 应命中「超期未跟进」
    r2 = FollowUp(
        customer_id=c2.customer_id, channel="微信", content="推送资料后未回",
        result="已联系", created_at=now - timedelta(days=20),
    )
    # c3：刚成交 → 既不该出现在计划到期，也不该出现在超期未跟进
    r3 = FollowUp(
        customer_id=c3.customer_id, channel="面谈", content="现场签约",
        result="已成交",
    )

    ok1, ok2, ok3 = db.save_follow_up(r1), db.save_follow_up(r2), db.save_follow_up(r3)
    check("三条跟进记录均写入成功", ok1 and ok2 and ok3, f"{ok1}/{ok2}/{ok3}")

    # -------- 3.3 跟进记录查询 --------
    section("3.3 跟进记录：按时间倒序列出")
    all_records = db.list_follow_ups()
    check("跟进记录总数 = 3", len(all_records) == 3, f"实际={len(all_records)}")
    check("按 created_at 倒序", all(
        all_records[i].created_at >= all_records[i + 1].created_at
        for i in range(len(all_records) - 1)
    ))
    only_c1 = db.list_follow_ups(customer_id=c1.customer_id)
    check("按客户过滤生效", len(only_c1) == 1 and only_c1[0].customer_id == c1.customer_id)

    # -------- 3.4 回写画像「最近跟进时间」+ 写入权限纪律 --------
    section("3.4 闭环复位：回写 last_follow_up，但不动 active_status")
    fresh_c1 = db.get_customer(c1.customer_id)
    lf = fresh_c1.last_follow_up
    if isinstance(lf, str):
        lf = SQLiteMemory._parse_dt(lf)
    delta = abs((lf - r1.created_at).total_seconds()) if lf else 9e9
    check("画像 last_follow_up 已回写为本次跟进时间", delta < 2, f"偏差={delta:.1f}s")
    check(
        "active_status 未被自动改写（仍为人工值「活跃」）",
        fresh_c1.active_status == "活跃",
        f"实际={fresh_c1.active_status!r}",
    )

    # -------- 3.5 待跟进提醒（两类口径）--------
    section("3.5 到期巡检：计划到期 + 超期未跟进")
    due = db.list_due_follow_ups(due_days=14)
    by_id = {i["customer_id"]: i for i in due}
    check("待跟进清单非空", len(due) > 0, f"条数={len(due)}")
    check(
        "c1 命中「计划到期」（next_follow_up_at 已到点）",
        by_id.get(c1.customer_id, {}).get("due_type") == "计划到期",
        f"实际={by_id.get(c1.customer_id, {}).get('due_type')}",
    )
    check(
        "c2 命中「超期未跟进」（20 天 > 阈值 14 天）",
        by_id.get(c2.customer_id, {}).get("due_type") == "超期未跟进",
        f"实际={by_id.get(c2.customer_id, {}).get('due_type')}",
    )
    check("c2 超期天数计算正确（≈20 天）",
          abs((by_id.get(c2.customer_id, {}).get("days") or 0) - 20) <= 1,
          f"实际={by_id.get(c2.customer_id, {}).get('days')}")
    check("c3（已成交、刚跟进）不在待跟进清单", c3.customer_id not in by_id)

    # -------- 3.6 「跟过了就不再提醒」的复位逻辑 --------
    section("3.6 复位逻辑：跟踪后不再重复提醒")
    r1b = FollowUp(
        customer_id=c1.customer_id, channel="电话", content="已加微信并预约试听",
        result="有意向",
    )
    db.save_follow_up(r1b)
    due_after = {i["customer_id"]: i for i in db.list_due_follow_ups(due_days=14)}
    check(
        "c1 再次跟进后，不再命中「计划到期」",
        due_after.get(c1.customer_id, {}).get("due_type") != "计划到期",
        f"实际={due_after.get(c1.customer_id, {}).get('due_type')}",
    )

    # -------- 3.7 转化统计（分子/分母口径）--------
    section("3.7 转化统计：分母 = 跟进条数，分子 = 已成交/有意向")
    stats30 = db.follow_up_conversion_stats(days=30)
    check("统计窗口 = 30 天", stats30["window_days"] == 30)
    check("分母（跟进记录数）= 4", stats30["total"] == 4, f"实际={stats30['total']}")
    check("分子（已成交/有意向）= 3", stats30["converted"] == 3, f"实际={stats30['converted']}")
    check(
        "转化率 = 3 / 4 = 0.75",
        stats30["conversion_rate"] is not None
        and abs(stats30["conversion_rate"] - 0.75) < 1e-6,
        f"实际={stats30['conversion_rate']}",
    )
    check("覆盖客户数 = 3", stats30["covered_customers"] == 3,
          f"实际={stats30['covered_customers']}")
    check(
        "结果分布含「已成交 / 有意向 / 已联系」",
        {"已成交", "有意向", "已联系"} <= set(stats30["by_result"].keys()),
        f"实际={stats30['by_result']}",
    )

    stats_win = db.follow_up_conversion_stats(days=1)
    check(
        "窗口缩小到 1 天 → 分母随之缩小（口径含时间窗）",
        stats_win["total"] < stats30["total"],
        f"1天={stats_win['total']} vs 30天={stats30['total']}",
    )
    check(
        "分子集合口径 = FOLLOW_UP_CONVERTED_RESULTS",
        FOLLOW_UP_CONVERTED_RESULTS == {"已成交", "有意向"},
        f"实际={sorted(FOLLOW_UP_CONVERTED_RESULTS)}",
    )

    # -------- 3.8 可观测性埋点 --------
    section("3.8 埋点：follow_up_total / follow_up_converted_total")
    total_metric = metrics.get_counter_base("follow_up_total")
    conv_metric = metrics.get_counter_base("follow_up_converted_total")
    check("follow_up_total 已累加（= 4 次跟进）", total_metric == 4, f"实际={total_metric}")
    check("follow_up_converted_total 已累加（= 3 次转化）", conv_metric == 3,
          f"实际={conv_metric}")

    # -------- 3.9 无数据口径：不报 0 冒充 --------
    section("3.9 无数据口径：转化率返回 None 而非 0")
    for rec in db.list_follow_ups():
        db.delete_follow_up(rec.follow_up_id)
    check("跟进记录已清空", len(db.list_follow_ups()) == 0)
    stats_empty = db.follow_up_conversion_stats(days=30)
    check("分母为 0 时不报 0，而是 None", stats_empty["total"] == 0
          and stats_empty["conversion_rate"] is None,
          f"total={stats_empty['total']}, rate={stats_empty['conversion_rate']}")

    # -------- 3.10 工具层 / 接口层常量自洽 --------
    section("3.10 枚举常量自洽（渠道 / 结果）")
    check("渠道枚举包含 电话/微信/面谈", {"电话", "微信", "面谈"} <= set(FOLLOW_UP_CHANNELS))
    check("结果枚举包含 已联系/未接/有意向/已成交/已放弃",
          {"已联系", "未接", "有意向", "已成交", "已放弃"} <= set(FOLLOW_UP_RESULTS))
    check("转化口径是结果枚举的子集",
          FOLLOW_UP_CONVERTED_RESULTS < set(FOLLOW_UP_RESULTS))

    db.close()

    # -------- 汇总 --------
    print("\n" + "=" * 68)
    print(f"自检汇总：通过 {len(_PASSED)} 项 / 失败 {len(_FAILED)} 项")
    if _FAILED:
        print("失败用例：")
        for name in _FAILED:
            print(f"  - {name}")
    print("=" * 68)
    return 1 if _FAILED else 0


if __name__ == "__main__":
    code = 1
    try:
        code = main()
    finally:
        # 清理临时测试库（含 WAL / SHM 附属文件）
        shutil.rmtree(_TMP_DIR, ignore_errors=True)
    sys.exit(code)
