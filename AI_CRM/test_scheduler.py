"""
test_scheduler.py - 常驻定时调度 + 主动告警推送 自检脚本
==========================================================
覆盖（对应 B.1 ⑤ 兜底「未落地项」的验收）：
  1. alerts 表 + 三个索引真实存在
  2. upsert_alert 写入成功，且「客户+类型+业务日」去重生效（重复写不新增）
  3. 重复命中不重置已读状态（已读后再巡检仍保持 read）
  4. list_alerts 按状态 / 顾问过滤正确
  5. count_unread_alerts 计数正确
  6. mark_alert_read / mark_all_alerts_read 生效
  7. AlertScheduler.run_once() 能跑通并落告警（端到端）
  8. 调度器 start / stop 生命周期正确（running 状态翻转）
  9. 原因文本 → 告警类型归类正确
 10. 严重度映射正确

运行：
  python test_scheduler.py
"""

import sys
import uuid
from datetime import datetime

from memory_sqlite import SQLiteMemory
from scheduler import (
    AlertScheduler,
    _classify_reason,
    _severity_of,
    get_scheduler,
)

# ==================== 断言工具 ====================
_passed = 0
_failed = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    """单条断言：打印结果并累计计数"""
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  [PASS] {name}" + (f"  ({detail})" if detail else ""))
    else:
        _failed += 1
        print(f"  [FAIL] {name}" + (f"  ({detail})" if detail else ""))


def section(title: str) -> None:
    print("-" * 60)
    print(f"## {title}")


def main() -> int:
    print("=" * 60)
    print("test_scheduler.py - 常驻定时调度 + 主动告警推送 自检")
    print("=" * 60)

    db = SQLiteMemory()
    # 用独立测试客户 ID，避免污染真实数据
    test_cid = f"TEST_{uuid.uuid4().hex[:8]}"
    test_advisor = "ADV_TEST"

    # ==================== 1. 表与索引 ====================
    section("1. alerts 表 + 索引存在性")
    db.cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='alerts'"
    )
    check("alerts 表已建", db.cursor.fetchone() is not None)

    db.cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='alerts'"
    )
    idx_names = {r[0] for r in db.cursor.fetchall()}
    check("idx_alerts_status 存在", "idx_alerts_status" in idx_names)
    check("idx_alerts_advisor 存在", "idx_alerts_advisor" in idx_names)
    check("idx_alerts_created 存在", "idx_alerts_created" in idx_names)

    # ==================== 2. 写入 + 去重 ====================
    section("2. upsert_alert 写入 + 业务日去重")
    ok1 = db.upsert_alert(
        customer_id=test_cid,
        alert_type="有意向未跟进",
        reasons=["有意向未跟进(>10天或无跟进)"],
        customer_name="测试客户",
        advisor_id=test_advisor,
        severity="high",
        source="scheduler",
    )
    check("首次写入成功", ok1)

    before = len([a for a in db.list_alerts(limit=500) if a["customer_id"] == test_cid])
    ok2 = db.upsert_alert(
        customer_id=test_cid,
        alert_type="有意向未跟进",
        reasons=["有意向未跟进(>10天或无跟进)", "补充原因"],
        customer_name="测试客户",
        advisor_id=test_advisor,
        severity="high",
        source="scheduler",
    )
    after = len([a for a in db.list_alerts(limit=500) if a["customer_id"] == test_cid])
    check("重复写入成功（走 UPDATE）", ok2)
    check("同客户+同类型+同业务日不新增", before == after, f"{before} → {after}")

    # 不同告警类型应各自成条
    db.upsert_alert(
        customer_id=test_cid,
        alert_type="高风险低意向",
        reasons=["高风险低意向(>45天未激活/未转化)"],
        customer_name="测试客户",
        advisor_id=test_advisor,
        severity="low",
        source="scheduler",
    )
    multi = [a for a in db.list_alerts(limit=500) if a["customer_id"] == test_cid]
    check("不同告警类型各自成条", len(multi) == 2, f"共 {len(multi)} 条")

    # ==================== 3. 已读状态不被重复命中重置 ====================
    section("3. 已读状态保持（重复巡检不重置为未读）")
    target = [a for a in multi if a["alert_type"] == "有意向未跟进"][0]
    db.mark_alert_read(target["alert_id"])
    read_after = [
        a for a in db.list_alerts(status="read", limit=500)
        if a["alert_id"] == target["alert_id"]
    ]
    check("标记已读成功", len(read_after) == 1)

    # 再次 upsert 同一去重键
    db.upsert_alert(
        customer_id=test_cid,
        alert_type="有意向未跟进",
        reasons=["有意向未跟进(>10天或无跟进)"],
        customer_name="测试客户",
        advisor_id=test_advisor,
        severity="high",
        source="scheduler",
    )
    still_read = [
        a for a in db.list_alerts(status="read", limit=500)
        if a["alert_id"] == target["alert_id"]
    ]
    check("重复命中后仍为已读", len(still_read) == 1)

    # ==================== 4. 过滤查询 ====================
    section("4. list_alerts 过滤")
    unread_all = db.list_alerts(status="unread", limit=500)
    check("status=unread 过滤生效", all(a["status"] == "unread" for a in unread_all))
    by_adv = db.list_alerts(advisor_id=test_advisor, limit=500)
    check("advisor_id 过滤生效", all(a["advisor_id"] == test_advisor for a in by_adv))

    # ==================== 5. 未读计数 ====================
    section("5. count_unread_alerts")
    cnt_all = db.count_unread_alerts()
    cnt_adv = db.count_unread_alerts(advisor_id=test_advisor)
    check("全局未读数 ≥ 顾问未读数", cnt_all >= cnt_adv, f"{cnt_all} / {cnt_adv}")
    check("顾问未读数 = 该顾问未读条数", cnt_adv == len(
        [a for a in unread_all if a["advisor_id"] == test_advisor]
    ))

    # ==================== 6. 批量已读 ====================
    section("6. mark_all_alerts_read")
    affected = db.mark_all_alerts_read(advisor_id=test_advisor)
    check("批量已读返回受影响条数", affected >= 0, f"affected={affected}")
    check("批量后该顾问未读归零", db.count_unread_alerts(advisor_id=test_advisor) == 0)

    # ==================== 7. 端到端 run_once ====================
    section("7. AlertScheduler.run_once() 端到端")
    sched = AlertScheduler(interval_seconds=99999)
    result = sched.run_once()
    check("run_once 返回统计字典", isinstance(result, dict))
    check("统计含 scanned/created/skipped/errors",
          all(k in result for k in ("scanned", "created", "skipped", "errors")))
    check("run_count 累加", sched.run_count == 1)
    check("last_run_at 已记录", sched.last_run_at is not None)
    print(f"    本轮巡检：{result}")

    # ==================== 8. 生命周期 ====================
    section("8. start / stop 生命周期")
    sched2 = AlertScheduler(interval_seconds=99999)
    check("初始未运行", not sched2.running)
    started = sched2.start()
    check("start() 返回 True", started)
    check("启动后 running=True", sched2.running)
    check("重复 start() 返回 False（幂等）", not sched2.start())
    stopped = sched2.stop()
    check("stop() 返回 True", stopped)
    check("停止后 running=False", not sched2.running)

    # ==================== 9. 类型归类 ====================
    section("9. 原因文本 → 告警类型归类")
    check("有意向未跟进", _classify_reason("有意向未跟进(>10天或无跟进)") == "有意向未跟进")
    check("流失/沉睡风险", _classify_reason("流失/沉睡风险") == "流失沉睡风险")
    check("高风险低意向", _classify_reason("高风险低意向(>45天未激活/未转化)") == "高风险低意向")
    check("未知原因兜底", _classify_reason("其它奇怪原因") == "其它")

    # ==================== 10. 严重度映射 ====================
    section("10. 严重度映射")
    check("有意向未跟进 → high", _severity_of("有意向未跟进") == "high")
    check("流失沉睡风险 → high", _severity_of("流失沉睡风险") == "high")
    check("计划到期 → medium", _severity_of("计划到期") == "medium")
    check("高风险低意向 → low", _severity_of("高风险低意向") == "low")
    check("未知类型 → medium", _severity_of("未知类型") == "medium")

    # ==================== 11. 单例 ====================
    section("11. get_scheduler 单例")
    check("两次获取同一实例", get_scheduler() is get_scheduler())

    # ==================== 清理测试数据 ====================
    section("清理测试数据")
    db.cursor.execute("DELETE FROM alerts WHERE customer_id = ?", (test_cid,))
    db.conn.commit()
    left = [a for a in db.list_alerts(limit=500) if a["customer_id"] == test_cid]
    check("测试告警已清空", len(left) == 0)

    db.close()

    # ==================== 汇总 ====================
    print("=" * 60)
    print(f"结果：{_passed} 通过 / {_failed} 失败")
    print("=" * 60)
    return 0 if _failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
