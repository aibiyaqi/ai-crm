"""
eval_store.py - 评测数据管理库（测试集 / Ground Truth / badcase + 版本管理）
================================================================================
背景（对应二面整改「评测体系」）：
  项目原来只有线上运行日志，评测用例是「写死在脚本里的常量」：
    · 想换一版测试集，只能改代码；
    · 线上发现的 badcase 没有沉淀路径，面试时也说不清「怎么迭代」。
  本模块把「测试集 / 标注 / badcase」落成 SQLite 表，提供：
    1. 测试集版本管理（同一 name 多 version，可指定 active）
    2. Ground Truth 用例的入库 / 读取（带 source：manual / online）
    3. badcase 库：线上线索沉淀 → 人工分诊 → 回流进测试集（闭环）
    4. 评测运行记录（哪版数据集、跑了什么指标、得分多少）

设计原则：
  - 纯标准库、独立 SQLite 文件（默认 eval_store.db），不污染业务库 crm.db；
  - 线程安全：每次操作使用独立短连接 + WAL + busy_timeout；
  - 幂等建表（IF NOT EXISTS），可重复调用 init_schema()。

用法：
  from eval_store import EvalStore
  store = EvalStore()
  ds_id = store.ensure_dataset("retrieval_golden", "v1", "retrieval")
  store.add_cases(ds_id, [("初三数学一对一学费", {"relevant": ["doc#1"]})])
  cases = store.get_cases(ds_id)

自检：
  python eval_store.py
"""

# ==================== 1. 导入依赖 ====================
from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

from db import get_database, load_sql_resource


# 进程内建表去重：同一引擎+库只执行一次 DDL
_SCHEMA_READY: set = set()


# ==================== 2. 表结构 ====================
_SCHEMA = """
-- 测试集版本：同一 name 可有多 version（支持「用例集演进」）
CREATE TABLE IF NOT EXISTS eval_dataset (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL,           -- 数据集名，如 retrieval_golden
    version     TEXT    NOT NULL,           -- 版本号，如 v1 / v2
    kind        TEXT    NOT NULL,           -- intent | retrieval | safety | e2e
    note        TEXT    DEFAULT '',
    is_active   INTEGER NOT NULL DEFAULT 1, -- 是否为该 name 的当前启用版本
    created_at  TEXT    NOT NULL,
    UNIQUE(name, version)
);

-- 测试用例（Ground Truth）
CREATE TABLE IF NOT EXISTS eval_case (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id  INTEGER NOT NULL,
    query       TEXT    NOT NULL,
    expected    TEXT    DEFAULT '{}',       -- JSON：期望值（意图/相关文档/答案要点）
    source      TEXT    DEFAULT 'manual',   -- manual | online | synthetic
    meta        TEXT    DEFAULT '{}',       -- JSON：其它标注信息
    created_at  TEXT    NOT NULL,
    FOREIGN KEY(dataset_id) REFERENCES eval_dataset(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_eval_case_dataset ON eval_case(dataset_id);

-- badcase 库（线上线索 → 分诊 → 回流）
CREATE TABLE IF NOT EXISTS eval_badcase (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    kind        TEXT    NOT NULL,           -- intent_miss | rag_empty | sql_reject | hallucination | other
    query       TEXT    NOT NULL,
    expected    TEXT    DEFAULT '',
    actual      TEXT    DEFAULT '',
    source      TEXT    DEFAULT 'online',   -- online | manual
    status      TEXT    NOT NULL DEFAULT 'open',  -- open | triaged | fixed | wontfix
    note        TEXT    DEFAULT '',
    created_at  TEXT    NOT NULL,
    updated_at  TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_eval_badcase_status ON eval_badcase(status);

-- 评测运行记录（哪版数据集、什么指标、多少分）
CREATE TABLE IF NOT EXISTS eval_run (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id  INTEGER,
    dataset     TEXT    DEFAULT '',         -- 冗余 name@version，便于直接展示
    metric      TEXT    NOT NULL,           -- recall@k / hit@k / mrr@k / intent_acc ...
    score       REAL    NOT NULL,
    detail      TEXT    DEFAULT '{}',       -- JSON：分项细节
    created_at  TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_eval_run_dataset ON eval_run(dataset_id);
"""


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _json(value: Any) -> str:
    """把任意对象安全地序列化成 JSON 字符串（不可序列化时降级为字符串）"""
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return json.dumps(str(value), ensure_ascii=False)


def _unjson(text: str, default: Any) -> Any:
    """把 JSON 字符串解析回对象（解析失败返回默认值）"""
    if not text:
        return default
    try:
        return json.loads(text)
    except Exception:
        return default


# ==================== 3. 数据访问对象 ====================
class EvalStore:
    """评测数据管理库（测试集 / Ground Truth / badcase / 运行记录）"""

    def __init__(self, path: Optional[str] = None, busy_timeout_ms: int = 5000):
        self.path = path or os.getenv(
            "EVAL_DB_PATH",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_store.db"),
        )
        self.busy_timeout_ms = busy_timeout_ms
        # 引擎切换位：mysql 时与业务库共用连接池（表结构见 sql/mysql/002_eval_store.sql）
        self._db = get_database()
        self.init_schema()

    # -------- 3.1 连接与建表 --------
    def connect(self):
        """每次操作使用独立短连接（SQLite）/ 从池中借还（MySQL）"""
        if self._db.engine == "mysql":
            return self._db.connect()
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute(f"PRAGMA busy_timeout={self.busy_timeout_ms};")
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    def init_schema(self) -> None:
        """幂等建表（按引擎选择建表脚本；进程内只执行一次）"""
        key = f"{self._db.engine}:{self.path}"
        if key in _SCHEMA_READY:
            return
        if self._db.engine == "mysql":
            self._db.executescript(load_sql_resource("sql/mysql/002_eval_store.sql"))
            _SCHEMA_READY.add(key)
            return
        conn = self.connect()
        try:
            conn.executescript(_SCHEMA)
            conn.commit()
            _SCHEMA_READY.add(key)
        finally:
            conn.close()

    # -------- 3.2 测试集版本管理 --------
    def ensure_dataset(
        self, name: str, version: str, kind: str = "retrieval", note: str = ""
    ) -> int:
        """
        获取或创建数据集版本，并把它标记为 active（同 name 下互斥）
        返回：dataset_id
        """
        conn = self.connect()
        try:
            row = conn.execute(
                "SELECT id FROM eval_dataset WHERE name=? AND version=?",
                (name, version),
            ).fetchone()
            if row is None:
                # 同 name 的其它版本取消 active（同一时刻只启用一个版本）
                conn.execute(
                    "UPDATE eval_dataset SET is_active=0 WHERE name=?", (name,)
                )
                cur = conn.execute(
                    "INSERT INTO eval_dataset(name, version, kind, note, is_active, created_at) "
                    "VALUES(?,?,?,?,1,?)",
                    (name, version, kind, note, _now()),
                )
                conn.commit()
                return int(cur.lastrowid)
            conn.execute(
                "UPDATE eval_dataset SET is_active=0 WHERE name=?", (name,)
            )
            conn.execute(
                "UPDATE eval_dataset SET is_active=1 WHERE id=?", (row["id"],)
            )
            conn.commit()
            return int(row["id"])
        finally:
            conn.close()

    def get_dataset(self, name: str, version: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """按 name(+version) 取数据集；version 为空时取 active 版本"""
        conn = self.connect()
        try:
            if version:
                row = conn.execute(
                    "SELECT * FROM eval_dataset WHERE name=? AND version=?",
                    (name, version),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT * FROM eval_dataset WHERE name=? AND is_active=1 "
                    "ORDER BY id DESC LIMIT 1",
                    (name,),
                ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def list_datasets(self) -> List[Dict[str, Any]]:
        conn = self.connect()
        try:
            rows = conn.execute(
                "SELECT d.*, (SELECT COUNT(*) FROM eval_case c WHERE c.dataset_id=d.id) "
                "AS case_count FROM eval_dataset d ORDER BY d.name, d.id DESC"
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    # -------- 3.3 测试用例 --------
    def add_cases(
        self,
        dataset_id: int,
        cases: Sequence[Tuple[str, Any]],
        source: str = "manual",
    ) -> int:
        """
        批量写入用例
        参数：
          cases : [(query, expected), ...]；expected 可为 dict / str / None
        返回：写入条数
        """
        conn = self.connect()
        try:
            n = 0
            for query, expected in cases:
                conn.execute(
                    "INSERT INTO eval_case(dataset_id, query, expected, source, meta, created_at) "
                    "VALUES(?,?,?,?,?,?)",
                    (dataset_id, str(query), _json(expected or {}), source, "{}", _now()),
                )
                n += 1
            conn.commit()
            return n
        finally:
            conn.close()

    def get_cases(self, dataset_id: int) -> List[Dict[str, Any]]:
        conn = self.connect()
        try:
            rows = conn.execute(
                "SELECT * FROM eval_case WHERE dataset_id=? ORDER BY id", (dataset_id,)
            ).fetchall()
            out: List[Dict[str, Any]] = []
            for r in rows:
                item = dict(r)
                item["expected"] = _unjson(item.get("expected", ""), {})
                item["meta"] = _unjson(item.get("meta", ""), {})
                out.append(item)
            return out
        finally:
            conn.close()

    def load_dataset_cases(
        self, name: str, version: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """便捷入口：按 name(+version) 直接取用例（供 eval_metrics 调用）"""
        ds = self.get_dataset(name, version)
        if not ds:
            return []
        return self.get_cases(int(ds["id"]))

    # -------- 3.4 badcase 库 --------
    def add_badcase(
        self,
        kind: str,
        query: str,
        expected: str = "",
        actual: str = "",
        source: str = "online",
        note: str = "",
    ) -> int:
        """记录一条 badcase（线上线索 / 人工发现）"""
        ts = _now()
        conn = self.connect()
        try:
            cur = conn.execute(
                "INSERT INTO eval_badcase(kind, query, expected, actual, source, status, "
                "note, created_at, updated_at) VALUES(?,?,?,?,?,?,?,?,?)",
                (kind, query, expected, actual, source, "open", note, ts, ts),
            )
            conn.commit()
            return int(cur.lastrowid)
        finally:
            conn.close()

    def list_badcases(
        self, status: Optional[str] = None, kind: Optional[str] = None, limit: int = 100
    ) -> List[Dict[str, Any]]:
        clauses, params = [], []
        if status:
            clauses.append("status=?")
            params.append(status)
        if kind:
            clauses.append("kind=?")
            params.append(kind)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)
        conn = self.connect()
        try:
            rows = conn.execute(
                f"SELECT * FROM eval_badcase {where} ORDER BY id DESC LIMIT ?", params
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def update_badcase(
        self, badcase_id: int, status: Optional[str] = None, note: Optional[str] = None
    ) -> bool:
        """更新 badcase 分诊状态 / 备注"""
        sets, params = [], []
        if status:
            sets.append("status=?")
            params.append(status)
        if note is not None:
            sets.append("note=?")
            params.append(note)
        if not sets:
            return False
        sets.append("updated_at=?")
        params.append(_now())
        params.append(badcase_id)
        conn = self.connect()
        try:
            cur = conn.execute(
                f"UPDATE eval_badcase SET {', '.join(sets)} WHERE id=?", params
            )
            conn.commit()
            return cur.rowcount > 0
        finally:
            conn.close()

    def promote_badcase(self, badcase_id: int, dataset_id: int) -> bool:
        """
        把 badcase 回流进测试集（闭环的关键一步）
        ==========================================
        用例的 expected 取 badcase.expected（可能为空 → 标记为待人工标注）。
        """
        conn = self.connect()
        try:
            row = conn.execute(
                "SELECT * FROM eval_badcase WHERE id=?", (badcase_id,)
            ).fetchone()
            if row is None:
                return False
            expected = row["expected"] or {"needs_label": True}
            conn.execute(
                "INSERT INTO eval_case(dataset_id, query, expected, source, meta, created_at) "
                "VALUES(?,?,?,?,?,?)",
                (
                    dataset_id,
                    row["query"],
                    _json(expected),
                    "online",
                    _json({"from_badcase": badcase_id, "kind": row["kind"]}),
                    _now(),
                ),
            )
            conn.execute(
                "UPDATE eval_badcase SET status='triaged', updated_at=? WHERE id=?",
                (_now(), badcase_id),
            )
            conn.commit()
            return True
        finally:
            conn.close()

    # -------- 3.5 运行记录 --------
    def record_run(
        self,
        metric: str,
        score: float,
        dataset_id: Optional[int] = None,
        dataset: str = "",
        detail: Optional[dict] = None,
    ) -> int:
        conn = self.connect()
        try:
            cur = conn.execute(
                "INSERT INTO eval_run(dataset_id, dataset, metric, score, detail, created_at) "
                "VALUES(?,?,?,?,?,?)",
                (dataset_id, dataset, metric, float(score), _json(detail or {}), _now()),
            )
            conn.commit()
            return int(cur.lastrowid)
        finally:
            conn.close()

    def list_runs(self, dataset_id: Optional[int] = None, limit: int = 50) -> List[Dict[str, Any]]:
        conn = self.connect()
        try:
            if dataset_id is not None:
                rows = conn.execute(
                    "SELECT * FROM eval_run WHERE dataset_id=? ORDER BY id DESC LIMIT ?",
                    (dataset_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM eval_run ORDER BY id DESC LIMIT ?", (limit,)
                ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    # -------- 3.6 导出 --------
    def export_dataset(self, name: str, version: Optional[str] = None) -> Dict[str, Any]:
        """导出某个数据集（含用例），用于备份 / 版本对比"""
        ds = self.get_dataset(name, version)
        if not ds:
            return {}
        return {"dataset": ds, "cases": self.get_cases(int(ds["id"]))}

    def stats(self) -> Dict[str, Any]:
        """概览统计（供监控大盘展示）"""
        conn = self.connect()
        try:
            datasets = conn.execute("SELECT COUNT(*) AS n FROM eval_dataset").fetchone()["n"]
            cases = conn.execute("SELECT COUNT(*) AS n FROM eval_case").fetchone()["n"]
            badcases = conn.execute("SELECT COUNT(*) AS n FROM eval_badcase").fetchone()["n"]
            open_badcases = conn.execute(
                "SELECT COUNT(*) AS n FROM eval_badcase WHERE status='open'"
            ).fetchone()["n"]
            runs = conn.execute("SELECT COUNT(*) AS n FROM eval_run").fetchone()["n"]
            return {
                "datasets": datasets,
                "cases": cases,
                "badcases": badcases,
                "open_badcases": open_badcases,
                "runs": runs,
            }
        finally:
            conn.close()


# ==================== 4. 自检 ====================
if __name__ == "__main__":  # pragma: no cover
    import tempfile

    tmp = os.path.join(tempfile.gettempdir(), "eval_store_selftest.db")
    if os.path.exists(tmp):
        os.remove(tmp)
    store = EvalStore(tmp)

    ds_id = store.ensure_dataset("retrieval_golden", "v1", "retrieval", "自检数据集")
    store.add_cases(ds_id, [
        ("初三数学一对一学费", {"relevant": ["doc#1", "doc#2"]}),
        ("你们有哪些班型", {"relevant": ["doc#3"]}),
    ])
    # 再建一个版本，验证 active 切换
    ds_v2 = store.ensure_dataset("retrieval_golden", "v2", "retrieval", "第二版")

    bc_id = store.add_badcase("intent_miss", "我想了解一下你们的课程体系", expected="rag_retrieve")
    store.promote_badcase(bc_id, ds_v2)
    store.record_run("recall@3", 0.83, dataset_id=ds_id, dataset="retrieval_golden@v1",
                     detail={"k": 3})

    print("=" * 68)
    print("eval_store.py 自检")
    print("=" * 68)
    print("datasets     :", store.list_datasets())
    print("active       :", store.get_dataset("retrieval_golden"))
    print("cases(v1)    :", len(store.get_cases(ds_id)))
    print("cases(v2)    :", len(store.get_cases(ds_v2)))
    print("badcases     :", store.list_badcases())
    print("runs         :", store.list_runs())
    print("stats        :", store.stats())
    print("=" * 68)
    os.remove(tmp)
