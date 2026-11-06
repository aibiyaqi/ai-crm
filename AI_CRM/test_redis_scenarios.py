# -*- coding: utf-8 -*-
"""
test_redis_scenarios.py - Redis 六场景测试（fakeredis，可离线运行）
==================================================================
覆盖：
  1. 意图缓存（Redis 命中 / LRU 兜底）
  2. 会话令牌缓存（写穿 / 仅缓存命中 / 删除失效）
  3. 告警去重（同客户+类型+日只落一条）
  4. 延迟队列 ZSET（投递 / 到期取回 / 撤销）
  5. 分布式锁（互斥 / 释放 / 重入）
  6. 熔断降级（连续失败后直接走 fallback）

依赖：fakeredis（未安装则整体 SKIP，不算失败）
运行：python test_redis_scenarios.py
"""

import os
import sys
import tempfile
from datetime import datetime

os.environ["DB_ENGINE"] = "sqlite"

try:
    import fakeredis
except ImportError:
    print("[SKIP] 未安装 fakeredis，跳过 Redis 场景测试（pip install fakeredis）")
    sys.exit(0)

TMP_DB = os.path.join(tempfile.gettempdir(), "crm_redis_test.db")
if os.path.exists(TMP_DB):
    os.remove(TMP_DB)
os.environ["DB_SQLITE_PATH"] = TMP_DB

from redis_client import RedisManager, configure_client, get_redis, reset_redis  # noqa: E402

FAILED = []


def check(name: str, cond: bool, detail: str = ""):
    if cond:
        print(f"  [OK]   {name}")
    else:
        print(f"  [FAIL] {name} {detail}")
        FAILED.append(name)


def main() -> int:
    fake = fakeredis.FakeRedis(decode_responses=True)
    rd = configure_client(fake)
    check("fakeredis 注入", rd.backend == "fake" and rd.available)

    # -------- 1. 意图缓存 --------
    print("\n[1] 意图缓存")
    from intent_cache import IntentCache
    cache = IntentCache()
    check("backend=redis", cache.backend == "redis")
    check("首次 miss", cache.get("初三数学多少钱") is None)
    cache.set("初三数学多少钱", "rag_retrieve")
    check("写入后命中", cache.get("初三数学多少钱") == "rag_retrieve")
    check("键带前缀", fake.exists("crm:intent:" + __import__("hashlib").md5(
        "初三数学多少钱".encode()).hexdigest()) == 1)

    # -------- 2. 会话令牌 --------
    print("\n[2] 会话令牌")
    from memory_sqlite import SQLiteMemory
    from models import Account, CustomerProfile

    db = SQLiteMemory()
    acc = Account(username="redis_user", password_hash="h", phone="13800000009")
    db.save_account(acc)
    db.save_session_token("tok-1", acc.account_id, expires_days=1)
    check("Redis 已写穿", fake.get("crm:session:tok-1") == acc.account_id)
    # 删掉数据库行，只靠 Redis 仍能鉴权
    db.cursor.execute("DELETE FROM session_tokens WHERE token = ?", ("tok-1",))
    db.conn.commit()
    check("仅缓存命中", db.get_account_id_by_token("tok-1") == acc.account_id)
    db.delete_session_token("tok-1")
    check("删除后失效", db.get_account_id_by_token("tok-1") is None)

    # -------- 3. 告警去重 --------
    print("\n[3] 告警去重")
    p = CustomerProfile(name="王五", phone="13800000010", intent_course="物理")
    db.save_customer(p)
    r1 = db.upsert_alert(p.customer_id, "高意向未跟进", ["3天未跟进"])
    r2 = db.upsert_alert(p.customer_id, "高意向未跟进", ["3天未跟进"])
    check("两次调用均成功", r1 and r2)
    check("库里只有一条", len(db.list_alerts()) == 1, str(len(db.list_alerts())))
    day = datetime.now().strftime("%Y-%m-%d")
    check("去重键存在",
          fake.exists(f"crm:alert:dedup:{p.customer_id}|高意向未跟进|{day}") == 1)

    # -------- 4. 延迟队列 --------
    print("\n[4] 延迟队列 ZSET")
    mid = db.schedule_follow_up_check(p.customer_id, delay_days=0)
    check("ZSET 已投递", mid in fake.zrange("crm:delay:follow_up", 0, -1))
    due = db.fetch_due_delayed_messages()
    check("到期可取回", any(d["msg_id"] == mid for d in due), str(due))
    db.cancel_follow_up_check(p.customer_id)
    check("撤销后 ZSET 清空", fake.zrange("crm:delay:follow_up", 0, -1) == [])
    check("库里状态为 cancelled", db.fetch_due_delayed_messages() == [])

    # -------- 5. 分布式锁 --------
    print("\n[5] 分布式锁")
    tok1 = rd.acquire_lock("scan:coarse:2026-11-06", 30)
    tok2 = rd.acquire_lock("scan:coarse:2026-11-06", 30)
    check("首次抢到", bool(tok1))
    check("第二次抢不到", tok2 is None)
    check("释放成功", rd.release_lock("scan:coarse:2026-11-06", tok1))
    check("释放后可再抢", bool(rd.acquire_lock("scan:coarse:2026-11-06", 30)))
    check("错误 token 不能释放",
          rd.release_lock("scan:coarse:2026-11-06", "wrong-token") is False)

    # -------- 6. 熔断降级 --------
    print("\n[6] 熔断降级")
    class Broken:
        def __getattr__(self, name):
            def _boom(*a, **k):
                raise RuntimeError("redis down")
            return _boom

    reset_redis()
    bad = RedisManager(enabled=True, auto_connect=False)
    bad._client = Broken()
    check("异常不向上抛",
          bad.run("get", lambda c: c.get("x"), default="FALLBACK") == "FALLBACK")
    for _ in range(5):
        bad.run("get", lambda c: c.get("x"))
    check("熔断已打开", bad.health()["breaker_open"])
    check("熔断期直接降级",
          bad.run("get", lambda c: c.get("x"), default="DEGRADED") == "DEGRADED")

    db.close()
    return 0 if not FAILED else 1


if __name__ == "__main__":
    print("=" * 68)
    print("test_redis_scenarios.py — Redis 六场景")
    print("=" * 68)
    code = main()
    print("\n" + "=" * 68)
    if FAILED:
        print(f"[FAIL] {len(FAILED)} 项未通过: {', '.join(FAILED)}")
    else:
        print("[OK] 全部通过")
    sys.exit(code)
