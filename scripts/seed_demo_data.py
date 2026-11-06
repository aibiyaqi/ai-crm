"""
seed_demo_data.py - 生成面试演示数据（可重复执行，不产生重复记录）
====================================================================
内容：
  · 3 个演示账号（advisor01 / advisor02 / advisor03，密码 Crm@2026）
  · 8 个 K12 客户（不同年级/课程/状态/来源/顾问）
  · 10 条跟进记录（含今天到期、已超期、已成交/有意向）
  · 3 条主动告警（未读）

设计：
  · customer_id / follow_up_id 用 uuid5 由固定字符串生成 → 重复执行是 upsert，不会翻倍
  · 时间相对「现在」偏移，保证「待跟进提醒」里既有今天到期、也有已超期的
  · 只写演示数据，不改任何业务代码

用法：
  python scripts/seed_demo_data.py            # SQLite（默认）
  DB_ENGINE=mysql python scripts/seed_demo_data.py
"""

from __future__ import annotations

import os
import sys
import uuid
from datetime import datetime, timedelta

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "AI_CRM"))

from memory_sqlite import SQLiteMemory          # noqa: E402
from models import CustomerProfile, FollowUp   # noqa: E402
from auth import AuthService, PasswordManager   # noqa: E402

NS = uuid.UUID("6f1d4b1e-2f5a-4c7e-9d3a-8b6c2e4f1a90")

# 演示账号（仅本地演示用；生产环境必须改密码）
DEMO_PASSWORD = "Crm@2026"
ACCOUNTS = [
    ("advisor01", "13900000001"),
    ("advisor02", "13900000002"),
    ("advisor03", "13900000003"),
]


def cid(key: str) -> str:
    return str(uuid.uuid5(NS, f"customer:{key}"))


def fid(key: str) -> str:
    return str(uuid.uuid5(NS, f"followup:{key}"))


def ago(days: float = 0, hours: float = 0) -> datetime:
    return datetime.now() - timedelta(days=days, hours=hours)


def ahead(days: float = 0) -> datetime:
    return datetime.now() + timedelta(days=days)


CUSTOMERS = [
    dict(key="zhang-yinuo", name="张一诺", phone="13812345678", identity="家长",
         age=13, school_grade="初二", intent_course="数学一对一", matched_course="初二数学同步培优",
         intent_price="300-350/课时", hourly_fee="320/课时", active_status="活跃",
         advisor_id="advisor01", source="转介绍", tags=["高意向", "初二", "数学"],
         notes="关注中考提分，家长本人是重点中学老师，对师资要求高",
         created_days=46, last_follow_days=1),
    dict(key="li-zihan", name="李子涵", phone="13987654321", identity="学生",
         age=15, school_grade="初三", intent_course="英语小班", matched_course="中考英语冲刺小班",
         intent_price="180/课时", hourly_fee="180/课时", active_status="待跟进",
         advisor_id="advisor02", source="地推", tags=["初三", "英语", "小班"],
         notes="试听过一次，家长嫌校区远，正在对比另一家",
         created_days=21, last_follow_days=5),
    dict(key="wang-zixuan", name="王梓萱", phone="13755556666", identity="家长",
         age=17, school_grade="高二", intent_course="物理一对一", matched_course="高二物理竞赛班",
         intent_price="380-420/课时", hourly_fee="400/课时", active_status="活跃",
         advisor_id="advisor01", source="抖音", tags=["高意向", "高二", "物理", "竞赛"],
         notes="目标物理竞赛省一，需要竞赛背景老师",
         created_days=33, last_follow_days=12),
    dict(key="chen-haoran", name="陈浩然", phone="13644447777", identity="学生",
         age=18, school_grade="高三", intent_course="化学冲刺班", matched_course="高考化学押题班",
         intent_price="220/课时", hourly_fee="220/课时", active_status="待跟进",
         advisor_id="advisor03", source="官网", tags=["高三", "化学", "冲刺"],
         notes="距离高考 5 个月，想补有机化学模块",
         created_days=58, last_follow_days=6),
    dict(key="liu-siyuan", name="刘思远", phone="13533338888", identity="家长",
         age=12, school_grade="初一", intent_course="语文阅读", matched_course="初一语文阅读写作",
         intent_price="150/课时", hourly_fee="150/课时", active_status="沉睡",
         advisor_id="advisor02", source="老带新", tags=["初一", "语文", "价格敏感"],
         notes="预算有限，希望先上体验课",
         created_days=75, last_follow_days=22),
    dict(key="zhao-yutong", name="赵雨桐", phone="13422229999", identity="家长",
         age=11, school_grade="小升初", intent_course="数学思维", matched_course="小升初数学思维班",
         intent_price="200/课时", hourly_fee="200/课时", active_status="活跃",
         advisor_id="advisor01", source="社群", tags=["已成交", "小升初", "数学"],
         notes="已报名秋季班，家长转介绍意愿强",
         created_days=64, last_follow_days=4),
    dict(key="sun-jiayi", name="孙嘉懿", phone="13311110000", identity="学生",
         age=16, school_grade="高一", intent_course="英语口语", matched_course="高一外教口语",
         intent_price="260/课时", hourly_fee="260/课时", active_status="流失风险",
         advisor_id="advisor03", source="电话名单", tags=["高一", "英语", "流失风险"],
         notes="两次约课未到，电话未接",
         created_days=50, last_follow_days=15),
    dict(key="zhou-kexin", name="周可欣", phone="13200001111", identity="家长",
         age=13, school_grade="初二", intent_course="生物地理会考", matched_course="生地会考冲刺班",
         intent_price="170/课时", hourly_fee="170/课时", active_status="待跟进",
         advisor_id="advisor02", source="转介绍", tags=["初二", "生地会考"],
         notes="会考在明年 6 月，想提前一年准备",
         created_days=12, last_follow_days=2),
]

# (客户 key, 跟进编号, 距今天数, 方式, 结果, 内容, 下次跟进(相对今天, 负数=已过期))
FOLLOWUPS = [
    ("zhang-yinuo", "f1", 9, "电话", "已联系", "首次沟通，家长反馈孩子在数学几何模块薄弱", None),
    ("zhang-yinuo", "f2", 3, "微信", "有意向", "发了师资介绍和初二数学大纲，家长认可", 1),
    ("li-zihan", "f1", 8, "微信", "未接", "加微信后未回复", None),
    ("li-zihan", "f2", 5, "电话", "已联系", "家长嫌校区远，推荐了线上小班", -1),
    ("wang-zixuan", "f1", 12, "面谈", "有意向", "到校区试听，确认竞赛班型", None),
    ("chen-haoran", "f1", 6, "电话", "已联系", "家长希望先看近三年高考化学真题解析", -2),
    ("liu-siyuan", "f1", 22, "短信", "未接", "发送体验课邀约，无回复", None),
    ("zhao-yutong", "f1", 10, "微信", "有意向", "确认秋季班名额", None),
    ("zhao-yutong", "f2", 4, "面谈", "已成交", "已缴费报名秋季数学思维班", 15),
    ("zhou-kexin", "f1", 2, "微信", "已联系", "发送生地会考规划，家长约周末详谈", 1),
]

# (客户 key, 告警类型, 原因列表, 严重度)
# 说明：这里只放两条「人工录入」告警作为演示；
#       其余告警由「告警中心 → 立即巡检」按规则自动生成（事件驱动链路）。
ALERTS = [
    ("li-zihan", "高意向未跟进", ["近 5 天内有沟通但未推进", "已推荐线上小班待确认"], "high"),
    ("chen-haoran", "超期未跟进", ["计划跟进时间已过 2 天"], "medium"),
]


def main() -> int:
    db = SQLiteMemory()
    auth = AuthService(db)
    n_acct = n_c = n_f = n_a = 0

    for username, phone in ACCOUNTS:
        existing = db.get_account_by_username(username)
        if existing:
            db.update_account_password(
                existing.account_id, PasswordManager.hash_password(DEMO_PASSWORD)
            )
        else:
            ok, _, _ = auth.register(username, phone, DEMO_PASSWORD)
            n_acct += 1 if ok else 0

    advisors = {c["key"]: c["advisor_id"] for c in CUSTOMERS}
    names = {c["key"]: c["name"] for c in CUSTOMERS}

    try:
        for c in CUSTOMERS:
            key = c["key"]
            profile = CustomerProfile(
                customer_id=cid(key),
                name=c["name"],
                phone=c["phone"],
                identity=c["identity"],
                age=c["age"],
                school_grade=c["school_grade"],
                intent_course=c["intent_course"],
                matched_course=c["matched_course"],
                intent_price=c["intent_price"],
                hourly_fee=c["hourly_fee"],
                active_status=c["active_status"],
                advisor_id=c["advisor_id"],
                source=c["source"],
                tags=c["tags"],
                notes=c["notes"],
                created_at=ago(c["created_days"]),
                updated_at=ago(c["last_follow_days"]),
                last_follow_up=ago(c["last_follow_days"]),
            )
            if db.save_customer(profile):
                n_c += 1

        # 从旧到新写入，最后一次回写画像的 last_follow_up 才是最新的
        for key, fno, days, channel, result, content, next_days in sorted(
            FOLLOWUPS, key=lambda x: x[2], reverse=True
        ):
            rec = FollowUp(
                follow_up_id=fid(f"{key}:{fno}"),
                customer_id=cid(key),
                advisor_id=advisors.get(key),
                channel=channel,
                result=result,
                content=content,
                created_at=ago(days),
                next_follow_up_at=ahead(next_days) if next_days is not None else None,
            )
            if db.save_follow_up(rec):
                n_f += 1

        for key, alert_type, reasons, severity in ALERTS:
            if db.upsert_alert(
                customer_id=cid(key),
                alert_type=alert_type,
                reasons=reasons,
                customer_name=names.get(key),
                advisor_id=advisors.get(key),
                severity=severity,
                source="manual",
            ):
                n_a += 1
    finally:
        db.close()

    print(f"[OK] 账号 {n_acct} 个（顾问密码 {DEMO_PASSWORD}）/ 客户 {n_c} 个 / 跟进 {n_f} 条 / 告警 {n_a} 条")
    return 0


if __name__ == "__main__":
    sys.exit(main())
