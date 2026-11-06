"""
test_sql_agent_ast.py - SQLAgent 集成测试（AST 护栏接入后）
============================================================
不依赖真实 LLM：用一个假 LLM 直接验证 _execute_sql 的「放行 / 拦截 / 只读」。

运行：python test_sql_agent_ast.py
"""

import sys

from sql_agent import SQLAgent, _validate_sql_ast, MAX_ROWS, SQL_GUARD


class _FakeLLM:
    """假 LLM：invoke 返回固定 SQL（不发起网络请求）"""
    def __init__(self, sql: str):
        self._sql = sql

    def invoke(self, messages):  # noqa: D401
        return {"content": f"```sql\n{self._sql}\n```"}


def _agent() -> SQLAgent:
    return SQLAgent(_FakeLLM("SELECT 1"))


def main() -> int:
    passed = failed = 0

    def check(title, cond, extra=""):
        nonlocal passed, failed
        if cond:
            passed += 1
            print(f"  [OK]   {title}")
        else:
            failed += 1
            print(f"  [FAIL] {title} {extra}")

    print("=" * 68)
    print("SQLAgent 集成测试：AST 护栏 + 只读边界")
    print("=" * 68)

    agent = _agent()

    # 1. 合法查询 → 放行并返回结果
    out = agent._execute_sql("SELECT COUNT(*) AS cnt FROM customer_profiles")
    check("合法聚合查询放行", "查询结果" in out or "命中" in out, f"→ {out[:80]}")

    # 2. 无 LIMIT → 护栏自动补 MAX_ROWS
    check("MAX_ROWS 常量 = 200", MAX_ROWS == 200, f"→ {MAX_ROWS}")
    v = _validate_sql_ast("SELECT * FROM customer_profiles")
    check("无 LIMIT 自动补 200", v.ok and v.limit == 200 and "LIMIT 200" in v.sql.upper(),
          f"→ {v.sql!r}")

    # 3. 越权列 → 拒绝
    out2 = agent._execute_sql("SELECT password_hash FROM accounts")
    check("敏感列 password_hash 被拒", "安全校验未通过" in out2, f"→ {out2[:80]}")

    # 4. 系统表 → 拒绝
    out3 = agent._execute_sql("SELECT * FROM sqlite_master")
    check("系统表 sqlite_master 被拒", "安全校验未通过" in out3, f"→ {out3[:80]}")

    # 5. 多语句注入 → 拒绝
    out4 = agent._execute_sql("SELECT * FROM customer_profiles; DROP TABLE users")
    check("多语句注入被拒", "安全校验未通过" in out4, f"→ {out4[:80]}")

    # 6. 危险函数 → 拒绝
    out5 = agent._execute_sql("SELECT load_extension('evil')")
    check("危险函数 load_extension 被拒", "安全校验未通过" in out5, f"→ {out5[:80]}")

    # 7. 非 SELECT → 拒绝
    out6 = agent._execute_sql("UPDATE customer_profiles SET name='x'")
    check("UPDATE 被拒", "安全校验未通过" in out6, f"→ {out6[:80]}")

    # 8. 白名单派生一致性（单源）
    from sql_agent import ALLOWED_TABLES
    check("ALLOWED_TABLES 与护栏单源一致",
          ALLOWED_TABLES == SQL_GUARD.allowed_tables,
          f"→ {sorted(ALLOWED_TABLES)}")

    # 9. 审计文件已产生 allow / reject 记录
    import os
    from sql_guard import _AUDIT_PATH
    txt = ""
    if os.path.exists(_AUDIT_PATH):
        with open(_AUDIT_PATH, "r", encoding="utf-8") as fh:
            txt = fh.read()
    check("审计含 allow 记录", '"event": "allow"' in txt or '"event":"allow"' in txt)
    check("审计含 reject 记录", '"event": "reject"' in txt or '"event":"reject"' in txt)

    print("\n" + "=" * 68)
    print(f"结果：通过 {passed} 项，失败 {failed} 项")
    print("=" * 68)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
