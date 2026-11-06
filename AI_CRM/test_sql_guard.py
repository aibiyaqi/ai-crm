"""
test_sql_guard.py - SQL AST 护栏回归测试
=========================================
覆盖：
  A. AST 白名单：Statement / Table / Column / Function
  B. Resource Limit：LIMIT 补入 + 超限收敛
  C. Read-only Boundary：只读连接拒写
  D. Timeout：progress_handler 超时中断

运行：python test_sql_guard.py
"""

import os
import sqlite3
import sys
import tempfile

from sql_guard import (
    SqlGuard,
    validate_sql_ast,
    open_readonly_connection,
    audit_sql,
)

# 关闭 sqlglot 交叉校验，保证测试只验证本模块自身的解析器
GUARD = SqlGuard(max_rows=200, use_sqlglot=False)

_passed = 0
_failed = 0


def check(title: str, cond: bool, extra: str = "") -> None:
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  [OK]   {title}")
    else:
        _failed += 1
        print(f"  [FAIL] {title} {extra}")


# ==================== A. 应当放行的合法只读查询 ====================
ALLOW_CASES = [
    "SELECT name FROM customer_profiles",
    "SELECT * FROM customer_profiles LIMIT 10",
    "SELECT COUNT(*) AS 总数 FROM customer_profiles WHERE active_status='活跃'",
    "SELECT school_grade, COUNT(*) AS 人数 FROM customer_profiles "
    "GROUP BY school_grade ORDER BY 人数 DESC",
    "SELECT a.name, u.phone FROM customer_profiles a JOIN users u ON a.name = u.name",
    "SELECT * FROM customer_profiles WHERE active_status IN (SELECT role FROM accounts)",
    "WITH t AS (SELECT customer_id FROM customer_profiles) SELECT COUNT(*) AS 总 FROM t",
    "SELECT strftime('%Y-%m', created_at) AS 月份, COUNT(*) AS 数量 "
    "FROM customer_profiles GROUP BY 月份",
    "SELECT * FROM login_logs WHERE action = 'fail' ORDER BY created_at DESC",
    "SELECT ROUND(AVG(age), 1) AS 平均年龄 FROM customer_profiles",
]

# ==================== B. 应当拦截的越权 / 危险 SQL ====================
BLOCK_CASES = [
    # (SQL, 期望出现在 error 里的关键字片段)
    ("SELECT password_hash FROM accounts", "列"),
    ("SELECT * FROM sqlite_master", "表"),
    ("SELECT * FROM password_reset_tokens", "表"),
    ("SELECT * FROM session_tokens", "表"),
    ("SELECT * FROM customer_profiles; DROP TABLE users", "多语句"),
    ("DROP TABLE users", "只读"),
    ("UPDATE customer_profiles SET name='x'", "只读"),
    ("SELECT load_extension('evil')", "函数"),
    ("SELECT writefile('x', content) FROM messages", "函数"),
    ("SELECT * FROM customer_profiles UNION SELECT * FROM sqlite_master", "集合运算"),
    ("SELECT note FROM customer_profiles", "列"),
]


def test_allow() -> None:
    print("\n[A] 合法只读查询应当放行")
    for sql in ALLOW_CASES:
        res = GUARD.validate(sql)
        check(f"放行: {sql[:64]}", res.ok, f"→ {res.error}")


def test_block() -> None:
    print("\n[B] 越权 / 危险 SQL 应当拦截")
    for sql, frag in BLOCK_CASES:
        if not frag:
            continue
        res = GUARD.validate(sql)
        ok = (not res.ok) and (frag in res.error)
        check(f"拦截: {sql[:64]}", ok, f"→ ok={res.ok} error={res.error}")


def test_limit() -> None:
    print("\n[C] Resource Limit：LIMIT 补入 / 超限收敛")
    r1 = GUARD.validate("SELECT * FROM customer_profiles")
    check("无 LIMIT → 自动补 LIMIT 200",
          r1.ok and r1.limit == 200 and r1.limit_injected
          and r1.sql.rstrip().upper().endswith("LIMIT 200"),
          f"→ sql={r1.sql!r}")

    r2 = GUARD.validate("SELECT * FROM customer_profiles LIMIT 9999")
    check("LIMIT 9999 → 收敛为 200",
          r2.ok and r2.limit == 200 and r2.limit_clamped
          and r2.sql.rstrip().upper().endswith("LIMIT 200"),
          f"→ sql={r2.sql!r}")

    r3 = GUARD.validate("SELECT * FROM customer_profiles LIMIT 20")
    check("LIMIT 20 → 原样保留",
          r3.ok and r3.limit == 20 and not r3.limit_injected
          and not r3.limit_clamped and "LIMIT 20" in r3.sql.upper(),
          f"→ sql={r3.sql!r}")

    r4 = GUARD.validate("SELECT * FROM customer_profiles LIMIT ?")
    check("LIMIT 参数化 → 拒绝（不给非确定上限）",
          (not r4.ok) and ("LIMIT" in r4.error), f"→ {r4.error}")


def test_structure() -> None:
    print("\n[D] AST 结构采集（表 / 列 / 函数）")
    r = GUARD.validate(
        "SELECT COUNT(*) AS 数量, school_grade FROM customer_profiles "
        "WHERE active_status='活跃' GROUP BY school_grade"
    )
    check("表 = {customer_profiles}", r.tables == {"customer_profiles"}, f"→ {r.tables}")
    check("函数含 count", "count" in r.functions, f"→ {r.functions}")
    check("列含 school_grade/active_status",
          {"school_grade", "active_status"} <= r.columns, f"→ {r.columns}")


def test_readonly_boundary() -> None:
    print("\n[E] Read-only Execution Boundary（只读连接拒写）")
    tmpdir = tempfile.mkdtemp(prefix="sqlguard_")
    db_path = os.path.join(tmpdir, "t.db")

    # 先建库写一行
    wconn = sqlite3.connect(db_path)
    wconn.execute("CREATE TABLE customer_profiles (customer_id TEXT, name TEXT)")
    wconn.execute("INSERT INTO customer_profiles VALUES ('C1', '张三')")
    wconn.commit()
    wconn.close()

    conn, mode = open_readonly_connection(db_path, timeout_ms=5000)
    check(f"只读连接建立成功 ({mode})", conn is not None)

    got = conn.execute("SELECT COUNT(*) FROM customer_profiles").fetchone()[0]
    check("只读连接可正常 SELECT", got == 1, f"→ count={got}")

    write_blocked = False
    try:
        conn.execute("INSERT INTO customer_profiles VALUES ('C2', '李四')")
        conn.commit()
    except sqlite3.OperationalError:
        write_blocked = True
    except sqlite3.DatabaseError:
        write_blocked = True
    check("只读连接写入被数据库拒绝", write_blocked)

    drop_blocked = False
    try:
        conn.execute("DROP TABLE customer_profiles")
    except sqlite3.DatabaseError:
        drop_blocked = True
    check("只读连接 DROP 被拒绝", drop_blocked)

    conn.close()
    try:
        os.remove(db_path)
        os.rmdir(tmpdir)
    except OSError:
        pass


def test_timeout() -> None:
    print("\n[F] Timeout：progress_handler 超时中断")
    tmpdir = tempfile.mkdtemp(prefix="sqlguard_t_")
    db_path = os.path.join(tmpdir, "t.db")
    sqlite3.connect(db_path).close()

    conn, _ = open_readonly_connection(db_path, timeout_ms=1)   # 给 1ms 预算
    heavy = (
        "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM c WHERE x<5000000) "
        "SELECT COUNT(*) FROM c"
    )
    interrupted = False
    try:
        conn.execute(heavy).fetchall()
    except sqlite3.OperationalError as exc:
        interrupted = "interrupt" in str(exc).lower()
    except sqlite3.DatabaseError:
        interrupted = True
    else:
        # 极端情况下跑完了也算超时机制未触发失败
        interrupted = False
    check("超长查询被超时中断", interrupted)
    conn.close()
    try:
        os.remove(db_path)
        os.rmdir(tmpdir)
    except OSError:
        pass


def test_audit() -> None:
    print("\n[G] 审计日志写入")
    from sql_guard import _AUDIT_PATH
    before = os.path.getsize(_AUDIT_PATH) if os.path.exists(_AUDIT_PATH) else 0
    audit_sql("allow", "SELECT 1", True, "selftest", tables=["-"])
    after = os.path.getsize(_AUDIT_PATH) if os.path.exists(_AUDIT_PATH) else 0
    check(f"审计文件追加成功 ({_AUDIT_PATH})", after > before, f"→ {before}→{after}")


def main() -> int:
    print("=" * 68)
    print("SQL AST 护栏回归测试")
    print("=" * 68)
    test_allow()
    test_block()
    test_limit()
    test_structure()
    test_readonly_boundary()
    test_timeout()
    test_audit()
    print("\n" + "=" * 68)
    print(f"结果：通过 {_passed} 项，失败 {_failed} 项")
    print("=" * 68)
    return 0 if _failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
