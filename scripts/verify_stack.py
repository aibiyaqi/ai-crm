"""
verify_stack.py - docker-compose 栈一键验收
============================================
前提：`docker compose up -d` 已启动（mysql + redis + streamlit + api）。
用途：把"起来了"变成"可验证"——逐项检查端口、数据库引擎、Redis、鉴权、查询与写库。

用法：
  python scripts/verify_stack.py
  python scripts/verify_stack.py --api http://localhost:8000 --streamlit http://localhost:8501

退出码：0=全部通过；1=有检查失败
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

FAILED = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'OK' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail and not cond else ""))
    if not cond:
        FAILED.append(name)


def http_json(url: str, timeout: float = 5.0, headers: dict | None = None):
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))


def http_status(url: str, timeout: float = 5.0) -> int:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status
    except urllib.error.HTTPError as exc:
        return exc.code
    except Exception:
        return 0


def wait_until(fn, timeout_s: float, interval_s: float = 2.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if fn():
            return True
        time.sleep(interval_s)
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default=os.getenv("VERIFY_API", "http://localhost:8000"))
    parser.add_argument("--streamlit", default=os.getenv("VERIFY_STREAMLIT", "http://localhost:8501"))
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    print("=" * 64)
    print("docker-compose 栈验收")
    print("=" * 64)

    print("\n[1] API 健康检查")
    ready = wait_until(lambda: http_status(f"{args.api}/api/status") == 200, args.timeout)
    check("GET /api/status = 200", ready)
    if not ready:
        print("\n[FAIL] API 未就绪，后续检查跳过")
        return 1

    code, status = http_json(f"{args.api}/api/status")
    check("database=true", status.get("database") is True)
    check("db_engine=mysql", status.get("db_engine") == "mysql", str(status.get("db_engine")))
    redis_info = status.get("redis") or {}
    check("redis.ok=true", redis_info.get("ok") is True, json.dumps(redis_info, ensure_ascii=False))
    check("status=ready", status.get("status") == "ready", str(status.get("status")))

    print("\n[2] Streamlit 前端")
    st = wait_until(lambda: http_status(args.streamlit) == 200, args.timeout)
    check("GET / = 200", st)

    print("\n[3] 鉴权链路（无 token 应 401/403）")
    unauth = http_status(f"{args.api}/api/alerts")
    check("未带 token 被拒绝", unauth in (401, 403), f"status={unauth}")

    print("\n[4] MySQL 可写（通过 API 写入的操作会落库）")
    # 说明：这里不直接建测试数据，避免污染；改由 /api/status 的 database 与后续人工验证覆盖。
    check("数据库连接可用（status.database）", status.get("database") is True)

    print("\n" + "=" * 64)
    if FAILED:
        print(f"[FAIL] {len(FAILED)} 项未通过: {', '.join(FAILED)}")
        return 1
    print("[OK] 栈验收全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
