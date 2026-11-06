# -*- coding: utf-8 -*-
"""
test_history_trim.py — 上下文裁剪自检（Token 动态计数 + 单条超长拦截）
=====================================================================
覆盖：
  1. 按 Token 动态裁剪：候选 12 条，超 Token 上限时从旧到新裁掉
  2. 单条超长拦截：save_message 落库前截断 + 提示尾标
  3. 兜底：单条就超 Token 上限时，至少保留最近 N 轮
  4. Token 估算：中文 ≈ 1 字 1 Token

运行：python test_history_trim.py
"""
import sys
import uuid
import datetime

sys.path.insert(0, ".")

from memory_sqlite import SQLiteMemory
from models import Message, User, ChatSession, Account
from config import (
    MAX_HISTORY_TOKENS,
    MAX_SINGLE_MESSAGE_CHARS,
    MIN_HISTORY_ROUNDS_ON_OVERFLOW,
)

PASS = 0
FAIL = 0


def check(name: str, cond: bool, detail: str = ""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [OK] {name} {detail}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name} {detail}")


def main():
    print("=" * 60)
    print("上下文裁剪自检（Token 动态计数 + 单条超长拦截）")
    print("=" * 60)

    db = SQLiteMemory()
    suffix = uuid.uuid4().hex[:6]
    acc_id = f"acc_{suffix}"
    uid = f"u_{suffix}"
    sid = f"s_{suffix}"

    # -------- 准备：账户 / 用户 / 会话 --------
    db.save_account(Account(account_id=acc_id, username=f"t_{suffix}",
                            password_hash="x", role="user",
                            phone=f"139{suffix}"))
    db.save_user(User(user_id=uid, name="测试", phone="13800000000"))
    db.save_session(ChatSession(session_id=sid, account_id=acc_id,
                                user_id=uid, title="t"))

    # -------- 1. Token 估算口径 --------
    print("\n[1] Token 估算（中文 ≈ 1 字 1 Token）")
    est = db._estimate_tokens("测试内容" * 300)  # 1200 个中文字
    check("1200 中文字 ≈ 1200 Token", 1150 <= est <= 1250, f"→ {est}")
    check("空串 → 0", db._estimate_tokens("") == 0)

    # -------- 2. 按 Token 动态裁剪 --------
    print("\n[2] 按 Token 动态裁剪（候选 12 条）")
    # 前 2 条各 1200 Token（超长），其余 10 条很短
    for i in range(12):
        content = ("测试内容" * 300) if i < 2 else f"短消息{i}"
        db.save_message(Message(
            message_id=uuid.uuid4().hex, session_id=sid, user_id=uid,
            role="user" if i % 2 == 0 else "assistant",
            content=content,
            created_at=datetime.datetime.now().isoformat(),
        ))
    hist = db.get_history(sid)
    total_tokens = sum(db._estimate_tokens(m.content) for m in hist)
    check("裁剪后总 Token ≤ 上限", total_tokens <= MAX_HISTORY_TOKENS,
          f"→ {total_tokens} ≤ {MAX_HISTORY_TOKENS}")
    check("裁剪后条数 < 候选 12 条", len(hist) < 12, f"→ {len(hist)} 条")
    check("保留的是较新消息（含短消息11）",
          any("短消息11" in m.content for m in hist))

    # -------- 3. 单条超长拦截 --------
    print("\n[3] 单条超长拦截（落库前截断）")
    long_content = "超长" * 5000  # 10000 字 > MAX_SINGLE_MESSAGE_CHARS
    db.save_message(Message(
        message_id=uuid.uuid4().hex, session_id=sid, user_id=uid,
        role="user", content=long_content,
        created_at=datetime.datetime.now().isoformat(),
    ))
    all_msgs = db.get_all_session_messages(sid)
    last = all_msgs[-1]
    check("落库内容被截断", len(last.content) < len(long_content),
          f"→ {len(last.content)} < {len(long_content)}")
    check("截断后带提示尾标", "已截断" in last.content)
    check("截断长度 ≈ 上限 + 尾标",
          len(last.content) <= MAX_SINGLE_MESSAGE_CHARS + 40,
          f"→ {len(last.content)}")

    # -------- 4. 兜底：单条就超 Token 上限 --------
    print("\n[4] 兜底（单条超 Token 上限时至少保留最近 N 轮）")
    sid2 = f"s2_{suffix}"
    db.save_session(ChatSession(session_id=sid2, account_id=acc_id,
                                user_id=uid, title="t2"))
    # 只放 1 条巨长的（估算 Token 远超 MAX_HISTORY_TOKENS）
    db.save_message(Message(
        message_id=uuid.uuid4().hex, session_id=sid2, user_id=uid,
        role="user", content="巨长" * 3000,  # 6000 字，但会被截断到 4000
        created_at=datetime.datetime.now().isoformat(),
    ))
    hist2 = db.get_history(sid2)
    check("兜底后至少保留 1 条（不空转）", len(hist2) >= 1, f"→ {len(hist2)} 条")

    # -------- 清理 --------
    db.clear_history(sid)
    db.clear_history(sid2)

    print("\n" + "=" * 60)
    print(f"结果：PASS={PASS} / FAIL={FAIL}")
    print("=" * 60)
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
