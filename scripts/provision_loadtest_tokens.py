"""
provision_loadtest_tokens.py - 压测前置：批量创建账户与会话令牌
================================================================
背景：Locust 会 monkey-patch socket（gevent），在压测进程里直接连 MySQL
      容易触发 WinError 10053 / 连接被中断。把「造数据」与「压请求」拆开：
      本脚本在普通进程里预置 N 个账户 + token，写入 JSON；
      Locust 只读 JSON，不再碰数据库。

用法：
  DB_ENGINE=mysql python scripts/provision_loadtest_tokens.py --count 100 \
      --out .dev/loadtest_tokens.json

输出 JSON：["token1", "token2", ...]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "AI_CRM"))

from memory_sqlite import SQLiteMemory  # noqa: E402
from models import Account              # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--out", default=".dev/loadtest_tokens.json")
    parser.add_argument("--prefix", default="load_u")
    args = parser.parse_args()

    out_path = os.path.join(BASE_DIR, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    db = SQLiteMemory()
    tokens = []
    created = 0
    try:
        for i in range(args.count):
            account_id = f"{args.prefix}{i}"
            if db.get_account_by_id(account_id) is None:
                acct = Account(
                    account_id=account_id,
                    username=account_id,
                    password_hash="loadtest",
                    role="user",
                    phone=f"1{abs(hash(account_id)) % 10 ** 10:010d}",
                    email=f"{account_id}@loadtest.local",
                    is_active=True,
                )
                if not db.save_account(acct):
                    print(f"[WARN] 账户创建失败: {account_id}")
                    continue
                created += 1
            token = f"lt-{uuid.uuid4().hex}"
            if db.save_session_token(token, account_id, expires_days=1):
                tokens.append(token)
    finally:
        db.close()

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(tokens, fh)

    print(f"[OK] 账户新建 {created} 个，token 写入 {len(tokens)} 条 -> {out_path}")
    return 0 if tokens else 1


if __name__ == "__main__":
    sys.exit(main())
