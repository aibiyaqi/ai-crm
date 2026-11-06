# -*- coding: utf-8 -*-
"""
test_mysql_integration.py - MySQL 端到端集成测试
================================================
前置：DB_ENGINE=mysql 且连接可用（DB_MYSQL_* 环境变量 / .env）。
      未启用 MySQL 时自动 SKIP（不算失败）。

覆盖：
  1. 连接 / 引擎 / 版本
  2. 账户 insert + upsert + 唯一键冲突拒绝
  3. 客户画像（中文 / emoji / JSON tags / LIKE 参数化）
  4. 会话 / 消息 / 历史顺序与条数
  5. 跟进 / 告警去重 / 延迟消息
  6. 会话令牌
  7. 评测库（4 表）
  8. 检查点库（put / get / list / trim）
  9. NL2SQL 只读边界（可查询、拒写）
 10. 表与索引完整性

运行：
  DB_ENGINE=mysql python test_mysql_integration.py
"""

import os
import sys
import time
from datetime import datetime, timedelta

if os.getenv("DB_ENGINE", "sqlite").lower() != "mysql":
    print("[SKIP] 未启用 MySQL（DB_ENGINE != mysql），跳过集成测试")
    sys.exit(0)

from memory_sqlite import SQLiteMemory          # noqa: E402
from models import (                            # noqa: E402
    Account, CustomerProfile, FollowUp, User, Message, ChatSession,
)

FAILED = []
SUFFIX = str(int(time.time()))[-6:]


def check(name, cond, detail=""):
    if cond:
        print(f"  [OK]   {name}")
    else:
        print(f"  [FAIL] {name} {detail}")
        FAILED.append(name)


def main() -> int:
    print("\n[1] 连接与引擎")
    db = SQLiteMemory()
    check("engine=mysql", db._db.engine == "mysql")
    check("ping", db._db.ping())
    print("       MySQL:", db._db.query("SELECT VERSION() AS v")[0]["v"])

    print("\n[2] 账户")
    acc = Account(username=f"it_{SUFFIX}", password_hash="h",
                  phone=f"136{SUFFIX.zfill(8)}", email="a@b.c")
    check("首次写入", db.save_account(acc))
    acc.email = "new@b.c"
    check("二次写入走 upsert", db.save_account(acc))
    got = db.get_account_by_username(acc.username)
    check("按用户名查回", got and got.account_id == acc.account_id)
    check("邮箱已更新", got.email == "new@b.c", got.email)
    dup = Account(username=acc.username, password_hash="h2", phone="13699999999")
    check("唯一键冲突被拒绝", db.save_account(dup) is False)

    print("\n[3] 客户画像")
    p = CustomerProfile(name="集成测试客户", phone="13600000000", age=15,
                        school_grade="初三", intent_course="数学一对一",
                        tags=["高意向", "初三"], notes="中文 emoji 😀")
    check("写入", db.save_customer(p))
    p2 = db.get_customer(p.customer_id)
    check("中文/emoji 往返", p2 and p2.name == "集成测试客户" and "😀" in (p2.notes or ""))
    check("tags 往返", p2.tags == ["高意向", "初三"])
    p2.active_status = "待跟进"
    check("upsert 更新", db.save_customer(p2) and
          db.get_customer(p.customer_id).active_status == "待跟进")
    check("LIKE 搜索", len(db.search_customers("集成测试")) >= 1)

    print("\n[4] 会话 / 消息 / 历史")
    u = User(name="集成测试用户")
    check("写入用户", db.save_user(u))
    s = ChatSession(account_id=acc.account_id, title="集成测试会话")
    check("写入会话", db.save_session(s))
    for i in range(3):
        db.save_message(Message(session_id=s.session_id, user_id=u.user_id,
                                role="user", content=f"第{i+1}条消息"))
    hist = db.get_history(s.session_id, limit=2)
    check("历史条数限制", len(hist) == 2, len(hist))
    check("历史顺序正确", hist[-1].content == "第3条消息", hist[-1].content)

    print("\n[5] 跟进 / 告警 / 延迟消息")
    fu = FollowUp(customer_id=p.customer_id, advisor_id="adv-1", channel="微信",
                  content="已沟通", result="有意向",
                  next_follow_up_at=datetime.now() + timedelta(days=2))
    check("写入跟进", db.save_follow_up(fu))
    check("跟进列表", len(db.list_follow_ups(customer_id=p.customer_id)) == 1)
    check("告警写入", db.upsert_alert(p.customer_id, "高意向未跟进", ["2天未跟进"]))
    check("告警重复不报错", db.upsert_alert(p.customer_id, "高意向未跟进", ["2天未跟进"]))
    mine = [a for a in db.list_alerts() if a.get("customer_id") == p.customer_id]
    check("告警去重（该客户 1 条）", len(mine) == 1, len(mine))
    mid = db.schedule_follow_up_check(p.customer_id, delay_days=3)
    check("投递延迟消息", bool(mid))
    check("未到期不返回", all(m["msg_id"] != mid for m in db.fetch_due_delayed_messages()))
    check("撤销延迟消息", db.cancel_follow_up_check(p.customer_id) == 1)

    print("\n[6] 会话令牌")
    tok = f"it-tok-{SUFFIX}"
    check("写入", db.save_session_token(tok, acc.account_id, expires_days=1))
    check("校验", db.get_account_id_by_token(tok) == acc.account_id)
    check("删除", db.delete_session_token(tok))
    check("删除后失效", db.get_account_id_by_token(tok) is None)

    print("\n[7] 评测库")
    from eval_store import EvalStore
    store = EvalStore()
    ds_id = store.ensure_dataset("it_smoke", "v1", "retrieval", "集成测试")
    store.add_cases(ds_id, [("初三数学多少钱", {"relevant": ["doc#1"]})])
    check("用例读取", len(store.get_cases(ds_id)) >= 1)
    check("badcase 写入", isinstance(store.add_badcase("intent_miss", "集成测试"), int))

    print("\n[8] 检查点库")
    from checkpoint import SQLiteCheckpointer
    cp = SQLiteCheckpointer()
    thread = f"it-thread-{SUFFIX}"
    cp.put({"configurable": {"thread_id": thread}}, {"step": 1}, {"node": "agent"})
    cp.put({"configurable": {"thread_id": thread}}, {"step": 2}, {"node": "tools"})
    t = cp.get_tuple({"configurable": {"thread_id": thread}})
    check("读回最新检查点", t and t.checkpoint == {"step": 2})
    check("检查点列表", len(list(cp.list({"configurable": {"thread_id": thread}}))) == 2)
    check("清理线程", cp.delete_thread(thread) == 2)

    print("\n[9] NL2SQL 只读边界")
    from sql_guard import SqlGuard, open_readonly_connection
    check("查询放行", SqlGuard(max_rows=5).validate(
        "SELECT name FROM customer_profiles").ok)
    conn, mode = open_readonly_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT name FROM customer_profiles LIMIT 5")
        cur.fetchall()
        blocked = False
        try:
            cur.execute("DELETE FROM customer_profiles WHERE 1=0")
        except Exception:
            blocked = True
        check(f"只读连接可查询（{mode}）", True)
        check("只读连接拒绝写入", blocked)
    finally:
        conn.close()

    print("\n[10] 表与索引")
    names = {
        r["t"] for r in db._db.query(
            "SELECT table_name AS t FROM information_schema.tables "
            "WHERE table_schema = DATABASE()"
        )
    }
    expect = {
        "accounts", "password_reset_tokens", "login_logs", "session_tokens",
        "sessions", "users", "messages", "customer_profiles", "follow_ups",
        "alerts", "pending_todo", "delayed_messages", "eval_dataset",
        "eval_case", "eval_badcase", "eval_run", "checkpoints",
    }
    check("17 张表齐全", expect <= names, sorted(expect - names))

    db.close()
    return 0 if not FAILED else 1


if __name__ == "__main__":
    print("=" * 68)
    print("test_mysql_integration.py — MySQL 端到端集成测试")
    print("=" * 68)
    code = main()
    print("\n" + "=" * 68)
    print("[OK] 全部通过" if not FAILED else f"[FAIL] {len(FAILED)} 项失败")
    sys.exit(code)
