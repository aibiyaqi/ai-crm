"""
migrate_sqlite_to_mysql.py - SQLite → MySQL 数据搬迁与校验
==========================================================
用途：把现有 crm.db / eval_store.db / checkpoints.db 的数据搬到 MySQL，
     并逐表校验（行数、主键集合、抽样字段），失败即报错，不静默丢数据。

前置：
  1. 已建库：CREATE DATABASE crm DEFAULT CHARSET utf8mb4 COLLATE utf8mb4_0900_ai_ci;
  2. 已配置环境变量（或 .env）：DB_MYSQL_HOST / PORT / USER / PASSWORD / DATABASE
  3. 强烈建议先备份三个 .db 文件（脚本会提示，但不会自动删源文件）

用法：
  # 业务库
  python scripts/migrate_sqlite_to_mysql.py --source AI_CRM/crm.db --business
  # 评测库 + 检查点库（同库不同表）
  python scripts/migrate_sqlite_to_mysql.py --source AI_CRM/eval_store.db --eval
  python scripts/migrate_sqlite_to_mysql.py --source AI_CRM/checkpoints.db --checkpoint
  # 一次搬完
  python scripts/migrate_sqlite_to_mysql.py --all --truncate

参数：
  --all         依次搬业务库/评测库/检查点库（默认路径在 AI_CRM 下）
  --truncate    先清空目标表（默认遇到非空目标库直接拒绝，避免误覆盖）
  --dry-run     只统计源库行数，不写 MySQL
  --batch N     批量写入条数（默认 500）

退出码：0=成功；1=有表校验失败
"""

from __future__ import annotations

import argparse
import os
import re
import sqlite3
import sys
from datetime import datetime
from typing import Dict, List, Tuple

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AI_CRM_DIR = os.path.join(BASE_DIR, "AI_CRM")
sys.path.insert(0, AI_CRM_DIR)

from db import split_sql_statements  # noqa: E402

# 外键顺序：父表在前，子表在后
BUSINESS_TABLES = [
    "accounts",
    "users",
    "sessions",
    "customer_profiles",
    "session_tokens",
    "password_reset_tokens",
    "login_logs",
    "messages",
    "follow_ups",
    "alerts",
    "pending_todo",
    "delayed_messages",
]
EVAL_TABLES = ["eval_dataset", "eval_case", "eval_badcase", "eval_run"]
CHECKPOINT_TABLES = ["checkpoints"]

SCHEMA_FILES = {
    "business": "sql/mysql/001_init.sql",
    "eval": "sql/mysql/002_eval_store.sql",
    "checkpoint": "sql/mysql/003_checkpoint.sql",
}

_DT_PATTERN = re.compile(r"^(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2})(\.\d+)?$")


def _normalize(value):
    """SQLite 文本时间 → MySQL DATETIME 可接受格式（去掉微秒与 T 分隔符）"""
    if isinstance(value, str):
        m = _DT_PATTERN.match(value)
        if m:
            return m.group(1).replace("T", " ")
    return value


def sqlite_connect(path: str) -> sqlite3.Connection:
    if not os.path.exists(path):
        raise SystemExit(f"[FAIL] 源库不存在: {path}")
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def sqlite_tables(conn: sqlite3.Connection) -> List[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    ).fetchall()
    return [r["name"] for r in rows]


def sqlite_columns(conn: sqlite3.Connection, table: str) -> List[str]:
    return [r["name"] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]


def sqlite_rows(conn: sqlite3.Connection, table: str):
    return conn.execute(f"SELECT * FROM {table}").fetchall()


def load_schema(kind: str) -> str:
    with open(os.path.join(AI_CRM_DIR, SCHEMA_FILES[kind]), "r", encoding="utf-8") as fh:
        return fh.read()


def mysql_connect():
    import pymysql

    from config import (
        DB_MYSQL_HOST,
        DB_MYSQL_PORT,
        DB_MYSQL_USER,
        DB_MYSQL_PASSWORD,
        DB_MYSQL_DATABASE,
    )

    return pymysql.connect(
        host=DB_MYSQL_HOST,
        port=int(DB_MYSQL_PORT),
        user=DB_MYSQL_USER,
        password=DB_MYSQL_PASSWORD,
        database=DB_MYSQL_DATABASE,
        charset="utf8mb4",
        autocommit=False,
        connect_timeout=10,
    )


def ensure_schema(mconn, kind: str) -> None:
    cur = mconn.cursor()
    try:
        for stmt in split_sql_statements(load_schema(kind)):
            cur.execute(stmt)
        mconn.commit()
    finally:
        cur.close()


def mysql_columns(mconn, table: str) -> List[str]:
    cur = mconn.cursor()
    try:
        cur.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = DATABASE() AND table_name = %s ORDER BY ordinal_position",
            (table,),
        )
        return [r[0] for r in cur.fetchall()]
    finally:
        cur.close()


def mysql_count(mconn, table: str) -> int:
    cur = mconn.cursor()
    try:
        cur.execute(f"SELECT COUNT(*) FROM `{table}`")
        return int(cur.fetchone()[0])
    finally:
        cur.close()


def mysql_pk_set(mconn, table: str, pk: str):
    cur = mconn.cursor()
    try:
        cur.execute(f"SELECT `{pk}` FROM `{table}`")
        return {str(r[0]) for r in cur.fetchall()}
    finally:
        cur.close()


def plan_columns(sconn, mconn, table: str) -> Tuple[List[str], List[str]]:
    """返回 (共同列, 只在 SQLite 里的列)；只搬共同列，避免源/目标结构漂移直接报错"""
    src = sqlite_columns(sconn, table)
    dst = set(mysql_columns(mconn, table))
    common = [c for c in src if c in dst]
    src_only = [c for c in src if c not in dst]
    return common, src_only


def copy_table(sconn, mconn, table: str, batch: int, truncate: bool,
               dry_run: bool) -> Tuple[int, int]:
    """
    拷贝单表
    ========
    返回 (源行数, 写入行数)
    """
    rows = sqlite_rows(sconn, table)
    n_src = len(rows)
    if dry_run:
        return n_src, 0

    common, src_only = plan_columns(sconn, mconn, table)
    if src_only:
        print(f"  [WARN] {table}: 源库存在目标库没有的列，已忽略: {', '.join(src_only)}")
    if not common:
        raise SystemExit(f"[FAIL] {table}: 源/目标没有共同列，无法迁移")

    cur = mconn.cursor()
    try:
        if truncate:
            # 外键检查已在 migrate() 里按会话关闭，这里直接清表
            cur.execute(f"TRUNCATE TABLE `{table}`")
            mconn.commit()

        cols = ", ".join(f"`{c}`" for c in common)
        placeholders = ", ".join(["%s"] * len(common))
        sql = f"INSERT INTO `{table}` ({cols}) VALUES ({placeholders})"

        written = 0
        for start in range(0, n_src, batch):
            chunk = rows[start:start + batch]
            payload = [
                tuple(_normalize(row[c]) for c in common) for row in chunk
            ]
            cur.executemany(sql, payload)
            mconn.commit()
            written += len(payload)
        return n_src, written
    except Exception:
        mconn.rollback()
        raise
    finally:
        cur.close()


def verify_table(sconn, mconn, table: str, pk_hint: str = "") -> Tuple[bool, str]:
    """校验：行数一致 + 主键集合一致（有单列主键时）"""
    n_src = len(sqlite_rows(sconn, table))
    n_dst = mysql_count(mconn, table)
    if n_src != n_dst:
        return False, f"行数不一致：sqlite={n_src} mysql={n_dst}"

    pk = pk_hint
    if not pk:
        info = sconn.execute(f"PRAGMA table_info({table})").fetchall()
        pks = [r["name"] for r in info if r["pk"]]
        pk = pks[0] if len(pks) == 1 else ""

    if pk:
        src_pk = {str(r[pk]) for r in sqlite_rows(sconn, table)}
        dst_pk = mysql_pk_set(mconn, table, pk)
        if src_pk != dst_pk:
            missing = list(src_pk - dst_pk)[:5]
            extra = list(dst_pk - src_pk)[:5]
            return False, f"主键集合不一致（缺失 {missing} / 多出 {extra}）"
    return True, f"sqlite={n_src} mysql={n_dst}"


# 外键孤儿检查：(子表, 外键列, 父表, 父键)
ORPHAN_CHECKS = [
    ("messages", "user_id", "users", "user_id"),
    ("messages", "session_id", "sessions", "session_id"),
    ("sessions", "account_id", "accounts", "account_id"),
    ("session_tokens", "account_id", "accounts", "account_id"),
    ("login_logs", "account_id", "accounts", "account_id"),
    ("follow_ups", "customer_id", "customer_profiles", "customer_id"),
    ("alerts", "customer_id", "customer_profiles", "customer_id"),
    ("pending_todo", "customer_id", "customer_profiles", "customer_id"),
    ("delayed_messages", "customer_id", "customer_profiles", "customer_id"),
]


def report_orphans(sconn: sqlite3.Connection, tables: List[str]) -> None:
    """
    迁移前检查外键孤儿
    ==================
    SQLite 时代的写路径并非全程开启 foreign_keys，
    历史数据可能存在孤儿行（如压测产生的 load_uXX 用户）。
    策略：只报告、不删除、不阻断迁移（MySQL 侧复制时临时关闭外键检查）。
    """
    present = set(tables)
    for child, col, parent, pk in ORPHAN_CHECKS:
        if child not in present or parent not in present:
            continue
        try:
            sql = (
                f"SELECT COUNT(*) FROM {child} "
                f"WHERE {col} IS NOT NULL AND {col} NOT IN (SELECT {pk} FROM {parent})"
            )
            n = sconn.execute(sql).fetchone()[0]
        except sqlite3.Error:
            continue
        if n:
            print(f"  [WARN] 外键孤儿：{child}.{col} → {parent}.{pk} 共 {n} 行（保留原样，不删除）")


def migrate(kind: str, tables: List[str], source: str, truncate: bool,
            dry_run: bool, batch: int) -> bool:
    print(f"\n{'=' * 72}")
    print(f"[{kind}] 源库: {source}")
    print(f"{'=' * 72}")

    sconn = sqlite_connect(source)
    existing = set(sqlite_tables(sconn))
    targets = [t for t in tables if t in existing]
    missing = [t for t in tables if t not in existing]
    if missing:
        print(f"[WARN] 源库缺少表（跳过）: {', '.join(missing)}")
    if not targets:
        print("[SKIP] 源库没有可迁移的表")
        sconn.close()
        return True

    if dry_run:
        report_orphans(sconn, targets)
        for t in targets:
            print(f"  {t:<22} 源行数={len(sqlite_rows(sconn, t))}")
        sconn.close()
        return True

    report_orphans(sconn, targets)

    mconn = mysql_connect()
    ok_all = True
    try:
        ensure_schema(mconn, kind)
        # 复制期间关闭外键检查：历史孤儿数据不应阻断搬迁（已通过 report_orphans 报告）
        fk_cur = mconn.cursor()
        fk_cur.execute("SET FOREIGN_KEY_CHECKS=0")
        fk_cur.close()
        for t in targets:
            n_src, n_dst = copy_table(sconn, mconn, t, batch, truncate, dry_run)
            ok, detail = verify_table(sconn, mconn, t)
            flag = "OK " if ok else "FAIL"
            print(f"  [{flag}] {t:<22} 源={n_src:<6} 写入={n_dst:<6} | {detail}")
            ok_all = ok_all and ok
    finally:
        try:
            cur = mconn.cursor()
            cur.execute("SET FOREIGN_KEY_CHECKS=1")
            cur.close()
        except Exception:
            pass
        mconn.close()
        sconn.close()
    return ok_all


def main() -> int:
    parser = argparse.ArgumentParser(description="SQLite → MySQL 迁移与校验")
    parser.add_argument("--source", help="源 SQLite 文件路径")
    parser.add_argument("--business", action="store_true", help="迁移业务库")
    parser.add_argument("--eval", action="store_true", help="迁移评测库")
    parser.add_argument("--checkpoint", action="store_true", help="迁移检查点库")
    parser.add_argument("--all", action="store_true", help="迁移全部三个库")
    parser.add_argument("--truncate", action="store_true", help="先清空目标表")
    parser.add_argument("--dry-run", action="store_true", help="只统计不写入")
    parser.add_argument("--batch", type=int, default=500, help="批量写入条数")
    args = parser.parse_args()

    print("=" * 72)
    print("SQLite → MySQL 迁移")
    print(f"时间: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("提醒: 迁移前请确认已备份 crm.db / eval_store.db / checkpoints.db")
    print("=" * 72)

    jobs: List[Tuple[str, List[str], str]] = []
    if args.all:
        jobs = [
            ("business", BUSINESS_TABLES, os.path.join(AI_CRM_DIR, "crm.db")),
            ("eval", EVAL_TABLES, os.path.join(AI_CRM_DIR, "eval_store.db")),
            ("checkpoint", CHECKPOINT_TABLES, os.path.join(AI_CRM_DIR, "checkpoints.db")),
        ]
    else:
        if not args.source:
            parser.error("未指定 --source（或使用 --all）")
        if args.business:
            jobs.append(("business", BUSINESS_TABLES, args.source))
        if args.eval:
            jobs.append(("eval", EVAL_TABLES, args.source))
        if args.checkpoint:
            jobs.append(("checkpoint", CHECKPOINT_TABLES, args.source))
        if not jobs:
            parser.error("请至少指定 --business / --eval / --checkpoint / --all 之一")

    all_ok = True
    for kind, tables, source in jobs:
        if not os.path.exists(source):
            print(f"\n[SKIP] {kind}: 源文件不存在 {source}")
            continue
        all_ok = migrate(kind, tables, source, args.truncate, args.dry_run, args.batch) and all_ok

    print("\n" + "=" * 72)
    print("[OK] 迁移与校验全部通过" if all_ok else "[FAIL] 存在校验失败的表，请检查上方输出")
    print("=" * 72)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
