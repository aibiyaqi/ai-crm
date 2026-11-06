# -*- coding: utf-8 -*-
"""
test_db_mysql_dialect.py - MySQL 方言适配层单测（不依赖真实数据库）
==================================================================
覆盖：
  1. adapt_sql_to_mysql：占位符 / 字面量 % / 字符串里的 ? / 注释 / upsert 改写
  2. split_sql_statements：多语句拆分（跳过字符串/注释里的分号）
  3. CompatRow：row[0] 与 row["col"] 双访问、dict()/keys()/迭代
  4. MySQL 建表脚本：关键语法约束（无 AUTOINCREMENT / 有 ENGINE+CHARSET）
  5. sql_guard 的 MySQL 方言：DATE_FORMAT 放行、strftime 拒绝、information_schema 拒绝
运行：
  python test_db_mysql_dialect.py
"""

import os
import re
import sys

# 必须在导入 config/sql_guard 之前设置：本文件验证 MySQL 方言分支
os.environ["DB_ENGINE"] = "mysql"

from db import (  # noqa: E402
    CompatRow,
    adapt_sql_to_mysql,
    split_sql_statements,
)
from sql_guard import SqlGuard  # noqa: E402

FAILED = []


def check(name: str, condition: bool, detail: str = ""):
    if condition:
        print(f"  [OK]   {name}")
    else:
        print(f"  [FAIL] {name} {detail}")
        FAILED.append(name)


def test_adapt():
    print("\n[1] adapt_sql_to_mysql")
    check(
        "? → %s",
        adapt_sql_to_mysql("SELECT * FROM t WHERE a = ? AND b = ?") ==
        "SELECT * FROM t WHERE a = %s AND b = %s",
    )
    check(
        "字符串里的 ? 不动",
        adapt_sql_to_mysql("SELECT '?' AS q, a FROM t WHERE b = ?") ==
        "SELECT '?' AS q, a FROM t WHERE b = %s",
    )
    check(
        "字面量 % 转义为 %%",
        adapt_sql_to_mysql("SELECT * FROM t WHERE a LIKE '%x%' AND b = ?") ==
        "SELECT * FROM t WHERE a LIKE '%%x%%' AND b = %s",
    )
    check(
        "无参数时不做 % 转义",
        adapt_sql_to_mysql("SELECT DATE_FORMAT(d, '%Y-%m') FROM t", has_params=False) ==
        "SELECT DATE_FORMAT(d, '%Y-%m') FROM t",
    )
    check(
        "注释里的 ? 不动",
        adapt_sql_to_mysql("SELECT a FROM t -- 这是?注释\nWHERE b = ?") ==
        "SELECT a FROM t -- 这是?注释\nWHERE b = %s",
    )
    adapted = adapt_sql_to_mysql(
        "INSERT INTO accounts (account_id, username) VALUES (?, ?) "
        "ON CONFLICT(account_id) DO UPDATE SET username = excluded.username"
    )
    check("ON CONFLICT → 行别名 upsert", "AS new ON DUPLICATE KEY UPDATE" in adapted)
    check("excluded. → new.", "username = new.username" in adapted)
    check("upsert 占位符已转换", adapted.count("%s") == 2, adapted)


def test_split():
    print("\n[2] split_sql_statements")
    script = """
    CREATE TABLE a (id INT);
    -- 注释里的分号; 不算
    CREATE TABLE b (name VARCHAR(64) DEFAULT ';');
    CREATE INDEX idx ON b (name);
    """
    stmts = split_sql_statements(script)
    check("拆出 3 条语句", len(stmts) == 3, str(stmts))
    check("字符串里的分号未被拆开", any("DEFAULT ';'" in s for s in stmts))
    check("空脚本返回空列表", split_sql_statements("  ;;  ") == [])


def test_row():
    print("\n[3] CompatRow")
    row = CompatRow(["customer_id", "name", "age"], ("c-1", "张三", 14))
    check("下标访问 row[0]", row[0] == "c-1")
    check("列名访问 row['name']", row["name"] == "张三")
    check("dict(row)", dict(row) == {"customer_id": "c-1", "name": "张三", "age": 14})
    check("keys()", row.keys() == ["customer_id", "name", "age"])
    check("迭代产出值", list(row) == ["c-1", "张三", 14])
    check("get 默认值", row.get("missing", "x") == "x")
    check("长度", len(row) == 3)
    try:
        row["nope"]
        check("不存在的列抛 IndexError", False)
    except IndexError:
        check("不存在的列抛 IndexError", True)


def test_schema_files():
    print("\n[4] MySQL 建表脚本")
    base = os.path.dirname(os.path.abspath(__file__))
    files = {
        "001_init.sql": 12,
        "002_eval_store.sql": 4,
        "003_checkpoint.sql": 1,
    }
    for fname, expect_tables in files.items():
        path = os.path.join(base, "sql", "mysql", fname)
        with open(path, "r", encoding="utf-8") as fh:
            sql = fh.read()
        tables = re.findall(r"CREATE TABLE IF NOT EXISTS\s+`?(\w+)`?", sql, re.I)
        # 去掉行注释后再做语法约束检查（注释里可能提到不允许的写法）
        code = re.sub(r"--[^\n]*", "", sql)
        check(f"{fname} 建表数 = {expect_tables}", len(tables) == expect_tables, str(tables))
        check(f"{fname} 无 AUTOINCREMENT", "AUTOINCREMENT" not in code.upper())
        check(f"{fname} 无 CREATE INDEX IF NOT EXISTS",
              "CREATE INDEX IF NOT EXISTS" not in code.upper())
        check(f"{fname} 显式引擎+字符集", sql.upper().count("ENGINE=INNODB") == len(tables))

    # 可选：sqlglot 静态语法校验（装了才跑）
    try:
        import sqlglot
        for fname in files:
            with open(os.path.join(base, "sql", "mysql", fname), encoding="utf-8") as fh:
                stmts = split_sql_statements(fh.read())
            bad = []
            for stmt in stmts:
                try:
                    sqlglot.parse_one(stmt, read="mysql")
                except Exception as exc:
                    bad.append(str(exc)[:60])
            check(f"sqlglot 语法校验 {fname}", not bad, "; ".join(bad))
    except ImportError:
        print("  [SKIP] 未安装 sqlglot，跳过第二解析器校验")


def test_guard():
    print("\n[5] sql_guard · MySQL 方言")
    g = SqlGuard(max_rows=200)
    r = g.validate(
        "SELECT DATE_FORMAT(created_at,'%Y-%m') AS 月份, COUNT(*) AS 数量 "
        "FROM customer_profiles GROUP BY 1"
    )
    check("DATE_FORMAT 放行", r.ok, r.error)

    r = g.validate("SELECT name, IFNULL(phone,'-') AS 电话 FROM customer_profiles")
    check("IFNULL 放行", r.ok, r.error)

    r = g.validate("SELECT strftime('%Y-%m', created_at) FROM customer_profiles")
    check("strftime 在 MySQL 下拒绝", not r.ok and "strftime" in r.error)

    r = g.validate("SELECT * FROM information_schema.tables")
    check("information_schema 拒绝", not r.ok)

    r = g.validate("SELECT SLEEP(10)")
    check("SLEEP 拒绝", not r.ok)

    r = g.validate("SELECT * FROM customer_profiles LIMIT 9999")
    check("LIMIT 收敛到 200", r.ok and r.sql.rstrip().upper().endswith("LIMIT 200"))

    r = g.validate("SELECT * FROM `accounts`")
    check("反引号标识符放行", r.ok, r.error)

    r = g.validate("DELETE FROM customer_profiles")
    check("写操作拒绝", not r.ok)


if __name__ == "__main__":
    print("=" * 68)
    print("test_db_mysql_dialect.py — MySQL 方言适配层单测")
    print("=" * 68)
    test_adapt()
    test_split()
    test_row()
    test_schema_files()
    test_guard()

    print("\n" + "=" * 68)
    if FAILED:
        print(f"[FAIL] {len(FAILED)} 项未通过: {', '.join(FAILED)}")
        sys.exit(1)
    print("[OK] 全部通过")
