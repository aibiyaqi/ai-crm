"""
db.py - 数据库访问层（DAO 抽象层）
=====================================
职责：
  1. 定义统一的 Database 抽象接口（连接 / 查询 / 执行 / 事务）
  2. SQLiteDatabase：本地开发 / 回滚实现（WAL + busy_timeout + foreign_keys）
  3. MySQLDatabase：生产实现（连接池 + 方言适配 + 兼容 Row + 断连/死锁重试）
  4. get_database() 工厂：按 config.DB_ENGINE 返回对应实现 —— 引擎切换位

设计取舍（为什么不改业务 SQL）：
  业务层有 158 处 execute、1146 个 "?" 占位，全是 SQLite 写法。
  改业务 SQL 的回归风险远大于"把方言差异收敛到 DAO 一层"：
    · "?" → "%s"（跳过字符串字面量/注释里的问号）
    · 字面量 "%" → "%%"（PyMySQL 用 % 做参数格式化，不转义会炸）
    · ON CONFLICT 由业务 SQL 保留，DAO 层不做文本替换（见各表 upsert 用 MySQL 分支）
    · 行对象统一为 CompatRow（同时支持 row[0] 与 row["col"]）

配置切换：改 DB_ENGINE（sqlite / mysql），业务代码零改动。
"""

# ==================== 1. 导入依赖 ====================
from abc import ABC, abstractmethod
import os
import random
import re
import sqlite3
import time
from contextlib import contextmanager
from typing import Any, Callable, Iterator, Optional, Sequence

from config import (
    DB_ENGINE,
    DB_SQLITE_PATH,
    DB_MYSQL_HOST,
    DB_MYSQL_PORT,
    DB_MYSQL_USER,
    DB_MYSQL_PASSWORD,
    DB_MYSQL_DATABASE,
    DB_MYSQL_POOL_SIZE,
    DB_MYSQL_MAX_OVERFLOW,
    DB_MYSQL_POOL_RECYCLE,
    DB_MYSQL_CHARSET,
    DB_MYSQL_TIMEZONE,
    DB_MYSQL_CONNECT_TIMEOUT,
    DB_MYSQL_READ_TIMEOUT,
    DB_MYSQL_WRITE_TIMEOUT,
    DB_MYSQL_READONLY_USER,
    DB_MYSQL_READONLY_PASSWORD,
    DB_MYSQL_SCHEMA_FILE,
)
from logger import logger

try:  # 指标埋点（可选依赖：缺失时降级为 no-op）
    from metrics import incr as _metric_incr
except Exception:  # pragma: no cover
    def _metric_incr(*args, **kwargs):  # type: ignore[misc]
        return None


# ==================== 2. 重试（只对可恢复错误重试） ====================
_MAX_RETRIES: int = int(os.getenv("DB_MAX_RETRIES", "5"))
_RETRY_BASE_S: float = float(os.getenv("DB_RETRY_BASE_S", "0.05"))
_RETRY_MAX_S: float = float(os.getenv("DB_RETRY_MAX_S", "1.0"))

# SQLite：写事务互相等待，busy_timeout 到期后抛 locked
_LOCK_HINTS = ("database is locked", "database table is locked", "locked")
# MySQL：1213 死锁 / 1205 锁等待超时 / 2006 服务端断开 / 2013 连接丢失 / 1040 连接数满
_MYSQL_RETRY_CODES = {1213, 1205, 2006, 2013, 1040}


def _is_sqlite_lock_error(exc: BaseException) -> bool:
    """判断是否为 SQLite 锁冲突（只对 OperationalError 判定，避免误判）"""
    if not isinstance(exc, sqlite3.OperationalError):
        return False
    msg = str(exc).lower()
    return any(h in msg for h in _LOCK_HINTS)


def _is_mysql_retryable(exc: BaseException) -> bool:
    """判断 MySQL 异常是否可重试（连接类 + 锁类）"""
    code = getattr(exc, "args", (None,))[0]
    return isinstance(code, int) and code in _MYSQL_RETRY_CODES


def _run_with_retry(
    fn: Callable[[], Any],
    op: str = "sql",
    retryable: Callable[[BaseException], bool] = _is_sqlite_lock_error,
) -> Any:
    """
    执行数据库操作，遇到可恢复错误时指数退避重试
    ==========================================
    参数：
      fn        : 无参可调用对象（真正执行 SQL 的闭包）
      op        : 操作名（用于日志 / 埋点标签）
      retryable : 判定函数（SQLite 锁 / MySQL 断连与死锁）
    异常：非可恢复异常、或重试耗尽时原样抛出
    """
    attempt = 0
    while True:
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - 需先按类型判定，再决定是否重试
            if not retryable(exc) or attempt >= _MAX_RETRIES:
                raise
            delay = min(_RETRY_MAX_S, _RETRY_BASE_S * (2 ** attempt))
            delay += random.uniform(0, delay * 0.25)
            attempt += 1
            _metric_incr("db_retry_total", op=op)
            logger.warning(
                f"[WARN] 数据库可恢复错误（{op}），第 {attempt}/{_MAX_RETRIES} 次重试，"
                f"{delay * 1000:.0f}ms 后重试: {exc}"
            )
            time.sleep(delay)


# ==================== 3. 兼容行对象（SQLite Row 语义） ====================
class CompatRow:
    """
    同时支持 row[0] 与 row["col"] 的行对象
    =========================================
    业务代码里两种写法都有（_row_to_* 用列名、迁移/令牌查询用下标），
    直接换 PyMySQL DictCursor 会打断下标访问，故提供兼容行。
    行为对齐 sqlite3.Row：迭代产出值、keys() 返回列名、dict(row) 可用。
    """

    __slots__ = ("_keys", "_values", "_index")

    def __init__(self, keys: Sequence[str], values: Sequence[Any]):
        self._keys = list(keys)
        self._values = tuple(values)
        self._index = {k: i for i, k in enumerate(self._keys)}

    def keys(self) -> list:
        return list(self._keys)

    def values(self) -> tuple:
        return self._values

    def items(self) -> list:
        return list(zip(self._keys, self._values))

    def get(self, key: str, default: Any = None) -> Any:
        idx = self._index.get(key)
        return default if idx is None else self._values[idx]

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, str):
            idx = self._index.get(key)
            if idx is None:
                raise IndexError(f"列不存在: {key}")
            return self._values[idx]
        return self._values[key]

    def __contains__(self, key: Any) -> bool:
        return isinstance(key, str) and key in self._index

    def __iter__(self) -> Iterator[Any]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __repr__(self) -> str:
        pairs = ", ".join(f"{k}={v!r}" for k, v in zip(self._keys, self._values))
        return f"<Row {pairs}>"

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, CompatRow):
            return self._keys == other._keys and self._values == other._values
        if isinstance(other, tuple):
            return self._values == other
        return NotImplemented


# ==================== 4. SQL 方言适配 ====================
def split_sql_statements(script: str) -> list:
    """
    按分号拆分多条 SQL（跳过字符串字面量与注释里的分号）
    =====================================================
    MySQL 没有 sqlite3.executescript，建表/迁移脚本需自行拆分。
    """
    statements: list = []
    buf: list = []
    i, n = 0, len(script)
    state = "normal"
    while i < n:
        ch = script[i]
        if state == "normal":
            if ch == "'":
                state = "squote"
                buf.append(ch)
            elif ch == '"':
                state = "dquote"
                buf.append(ch)
            elif ch == "`":
                state = "btick"
                buf.append(ch)
            elif ch == "-" and i + 1 < n and script[i + 1] == "-":
                state = "line"
                buf.append(ch)
            elif ch == "/" and i + 1 < n and script[i + 1] == "*":
                state = "block"
                buf.append(ch)
            elif ch == ";":
                stmt = "".join(buf).strip()
                if stmt:
                    statements.append(stmt)
                buf = []
            else:
                buf.append(ch)
        elif state == "squote":
            buf.append(ch)
            if ch == "\\" and i + 1 < n:
                buf.append(script[i + 1])
                i += 1
            elif ch == "'":
                if i + 1 < n and script[i + 1] == "'":
                    buf.append("'")
                    i += 1
                else:
                    state = "normal"
        elif state == "dquote":
            buf.append(ch)
            if ch == "\\" and i + 1 < n:
                buf.append(script[i + 1])
                i += 1
            elif ch == '"':
                if i + 1 < n and script[i + 1] == '"':
                    buf.append('"')
                    i += 1
                else:
                    state = "normal"
        elif state == "btick":
            buf.append(ch)
            if ch == "`":
                if i + 1 < n and script[i + 1] == "`":
                    buf.append("`")
                    i += 1
                else:
                    state = "normal"
        elif state == "line":
            buf.append(ch)
            if ch == "\n":
                state = "normal"
        elif state == "block":
            buf.append(ch)
            if ch == "*" and i + 1 < n and script[i + 1] == "/":
                buf.append("/")
                i += 1
                state = "normal"
        i += 1
    tail = "".join(buf).strip()
    if tail:
        statements.append(tail)
    return statements


def adapt_sql_to_mysql(sql: str, has_params: bool = True) -> str:
    """
    SQLite 方言 SQL → MySQL 方言 SQL（占位符 + upsert 结构）
    =======================================================
    1) "?" → "%s"（跳过字符串字面量 / 反引号标识符 / 注释里的问号）
    2) 有参数时把字面量 "%" 转义为 "%%"（否则 PyMySQL 的 % 格式化会抛
       "unsupported format character"）
    3) upsert：`ON CONFLICT(col) DO UPDATE SET ... excluded.x`
              → `AS new ON DUPLICATE KEY UPDATE ... new.x`
       （MySQL ≥ 8.0.19 的行别名语法；VALUES() 已在 8.0.20 标记废弃）
    注意：PRAGMA / CREATE INDEX IF NOT EXISTS 等结构性差异不在这里处理，
          由建表脚本与仓储方法的方言分支负责。
    """
    out: list = []
    i, n = 0, len(sql)
    state = "normal"
    while i < n:
        ch = sql[i]
        if state == "normal":
            if ch == "'":
                state = "squote"
                out.append(ch)
            elif ch == '"':
                state = "dquote"
                out.append(ch)
            elif ch == "`":
                state = "btick"
                out.append(ch)
            elif ch == "-" and i + 1 < n and sql[i + 1] == "-":
                state = "line"
                out.append(ch)
            elif ch == "/" and i + 1 < n and sql[i + 1] == "*":
                state = "block"
                out.append(ch)
            elif ch == "?":
                out.append("%s" if has_params else "?")
            elif ch == "%" and has_params:
                out.append("%%")
            else:
                out.append(ch)
        elif state == "squote":
            out.append(ch)
            if ch == "\\" and i + 1 < n:
                out.append(sql[i + 1])
                i += 1
            elif ch == "%" and has_params:
                # PyMySQL 的 % 格式化作用于整条 SQL（含字符串字面量），必须一起转义
                out[-1] = "%%"
            elif ch == "'":
                if i + 1 < n and sql[i + 1] == "'":
                    out.append("'")
                    i += 1
                else:
                    state = "normal"
        elif state == "dquote":
            out.append(ch)
            if ch == "\\" and i + 1 < n:
                out.append(sql[i + 1])
                i += 1
            elif ch == "%" and has_params:
                out[-1] = "%%"
            elif ch == '"':
                if i + 1 < n and sql[i + 1] == '"':
                    out.append('"')
                    i += 1
                else:
                    state = "normal"
        elif state == "btick":
            out.append(ch)
            if ch == "%" and has_params:
                out[-1] = "%%"
            elif ch == "`":
                if i + 1 < n and sql[i + 1] == "`":
                    out.append("`")
                    i += 1
                else:
                    state = "normal"
        elif state == "line":
            out.append(ch)
            if ch == "%" and has_params:
                out[-1] = "%%"
            elif ch == "\n":
                state = "normal"
        elif state == "block":
            out.append(ch)
            if ch == "%" and has_params:
                out[-1] = "%%"
            elif ch == "*" and i + 1 < n and sql[i + 1] == "/":
                out.append("/")
                i += 1
                state = "normal"
        i += 1

    adapted = "".join(out)
    if not has_params:
        return adapted

    # upsert 结构：ON CONFLICT(k) DO UPDATE SET ...  →  AS new ON DUPLICATE KEY UPDATE ...
    adapted = re.sub(
        r"ON\s+CONFLICT\s*\([^)]*\)\s+DO\s+UPDATE\s+SET",
        "AS new ON DUPLICATE KEY UPDATE",
        adapted,
        flags=re.IGNORECASE,
    )
    adapted = re.sub(r"\bexcluded\.", "new.", adapted, flags=re.IGNORECASE)
    adapted = re.sub(r"\bINSERT\s+OR\s+IGNORE\b", "INSERT IGNORE", adapted, flags=re.IGNORECASE)
    return adapted


# ==================== 5. 抽象接口 ====================
class Database(ABC):
    """
    数据库访问抽象接口
    ====================
    业务/仓库层只依赖本接口，不依赖具体引擎（SQLite / MySQL）。
    迁移数据库时，只需新增对应实现并修改 DB_ENGINE 配置。
    """

    #: 引擎标识（sqlite / mysql）
    engine: str = "abstract"
    #: SQL 方言（供 sql_guard / sql_agent 生成对应语句）
    dialect: str = "sqlite"

    @abstractmethod
    def connect(self) -> Any:
        """创建并返回一个已配置（PRAGMA/参数）的连接"""

    @abstractmethod
    def query(self, sql: str, params: Sequence[Any] = ()) -> list:
        """执行 SELECT，返回全部行（参数化，防注入）"""

    @abstractmethod
    def query_one(self, sql: str, params: Sequence[Any] = ()) -> Optional[Any]:
        """执行 SELECT，返回首行或 None"""

    @abstractmethod
    def execute(self, sql: str, params: Sequence[Any] = ()) -> int:
        """执行 INSERT/UPDATE/DELETE，返回影响行数（自动提交）"""

    @abstractmethod
    def executescript(self, sql: str) -> None:
        """执行多语句脚本（建表 / 迁移）"""

    @abstractmethod
    def transaction(self) -> Iterator["Database"]:
        """事务上下文管理器：正常结束 commit，异常 rollback"""

    @abstractmethod
    def close(self) -> None:
        """关闭连接"""

    def ping(self) -> bool:
        """健康检查（可选实现）"""
        try:
            return self.query_one("SELECT 1") is not None
        except Exception:
            return False


# ==================== 6. SQLite 实现（本地开发 / 回滚） ====================
class SQLiteDatabase(Database):
    """
    SQLite 实现
    ==============
    连接优化（有代码证据）：
      - PRAGMA journal_mode=WAL;          （WAL 模式：读写不互斥）
      - PRAGMA busy_timeout=5000;         （锁等待 5 秒而非立刻报错）
      - PRAGMA foreign_keys=ON;           （外键约束）
      - PRAGMA synchronous=NORMAL;        （平衡性能与可靠性）
    """

    engine = "sqlite"
    dialect = "sqlite"

    def __init__(self, path: Optional[str] = None, busy_timeout_ms: int = 5000):
        self.path = path or DB_SQLITE_PATH
        self.busy_timeout_ms = busy_timeout_ms
        self._conn: Optional[sqlite3.Connection] = None

    def connect(self) -> sqlite3.Connection:
        """创建连接并应用 SQLite 连接优化 PRAGMA"""
        conn = sqlite3.connect(
            self.path,
            check_same_thread=False,               # 允许多线程访问
            detect_types=sqlite3.PARSE_DECLTYPES,  # 自动解析类型
        )
        conn.row_factory = sqlite3.Row
        # journal_mode 切换需要短暂排它锁，可能与其他写事务冲突 → 同样做重试
        _run_with_retry(lambda: conn.execute("PRAGMA journal_mode=WAL;"), op="pragma_wal")
        conn.execute(f"PRAGMA busy_timeout={self.busy_timeout_ms};")
        conn.execute("PRAGMA foreign_keys=ON;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _get_conn(self) -> sqlite3.Connection:
        """惰性获取（首次创建后复用），与 memory_sqlite 单连接模式兼容"""
        if self._conn is None:
            self._conn = self.connect()
        return self._conn

    def query(self, sql: str, params: Sequence[Any] = ()) -> list:
        cur = _run_with_retry(
            lambda: self._get_conn().execute(sql, params), op="query"
        )
        return cur.fetchall()

    def query_one(self, sql: str, params: Sequence[Any] = ()) -> Optional[Any]:
        cur = _run_with_retry(
            lambda: self._get_conn().execute(sql, params), op="query_one"
        )
        return cur.fetchone()

    def execute(self, sql: str, params: Sequence[Any] = ()) -> int:
        def _do() -> int:
            conn = self._get_conn()
            cur = conn.execute(sql, params)
            conn.commit()          # 提交也可能因锁冲突失败 → 一并重试
            return cur.rowcount

        return _run_with_retry(_do, op="execute")

    def executescript(self, sql: str) -> None:
        def _do() -> None:
            conn = self._get_conn()
            conn.executescript(sql)
            conn.commit()

        _run_with_retry(_do, op="executescript")

    @contextmanager
    def transaction(self) -> Iterator["Database"]:
        conn = self._get_conn()
        try:
            yield self
            _run_with_retry(conn.commit, op="commit")
        except Exception:
            conn.rollback()
            raise

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None


# ==================== 7. MySQL 实现（生产） ====================
def _make_compat_cursor(pymysql_module):
    """
    生成 PyMySQL 兼容游标：fetchone/fetchall 返回 CompatRow
    =====================================================
    1) 行对象兼容 sqlite3.Row（row[0] 与 row["col"] 都能用）
    2) execute/executemany 自动做方言适配（"?" → "%s"、upsert 结构）
       —— 这样 eval_store / checkpoint / sql_guard 里的裸游标代码不用改。
    """

    class _CompatCursor(pymysql_module.cursors.Cursor):
        def _wrap(self, row):
            if row is None or not self.description:
                return row
            return CompatRow([d[0] for d in self.description], row)

        def execute(self, query, args=None):
            return super().execute(adapt_sql_to_mysql(query, bool(args)), args)

        def executemany(self, query, args):
            seq = list(args)
            return super().executemany(adapt_sql_to_mysql(query, bool(seq)), seq)

        def fetchone(self):
            return self._wrap(super().fetchone())

        def fetchall(self):
            return [self._wrap(r) for r in super().fetchall()]

        def fetchmany(self, size=None):
            rows = super().fetchmany(size) if size else super().fetchmany()
            return [self._wrap(r) for r in rows]

    return _CompatCursor


class MySQLConnectionAdapter:
    """
    MySQL 连接的 sqlite3 风格适配器
    ================================
    原代码里有两种写法：
      · conn.cursor().execute(...)     —— 标准 DB-API
      · conn.execute(...)              —— sqlite3 的便捷方法（checkpoint / eval_store 在用）
    PyMySQL 连接没有后者，这里补齐，避免为了迁库重写这两个模块。
    """

    def __init__(self, conn):
        self._conn = conn

    # -------- sqlite3 便捷方法 --------
    def execute(self, sql: str, params: Sequence[Any] = ()):
        cur = self._conn.cursor()
        cur.execute(sql, tuple(params) or None)
        return cur

    def executemany(self, sql: str, seq):
        cur = self._conn.cursor()
        cur.executemany(sql, seq)
        return cur

    def executescript(self, script: str) -> None:
        """MySQL 无 executescript：按分号拆分逐条执行"""
        cur = self._conn.cursor()
        try:
            for stmt in split_sql_statements(script):
                cur.execute(stmt)
            self._conn.commit()
        finally:
            cur.close()

    # -------- 标准 DB-API --------
    def cursor(self):
        return self._conn.cursor()

    def commit(self) -> None:
        self._conn.commit()

    def rollback(self) -> None:
        self._conn.rollback()

    def close(self) -> None:
        self._conn.close()

    @property
    def raw(self):
        return self._conn


class MySQLDatabase(Database):
    """
    MySQL 实现（P1 落地）
    ======================
    要点：
      1. DBUtils.PooledDB 连接池：多线程 / 多会话共享池，不共享单条连接
      2. 方言适配：adapt_sql_to_mysql("?" → "%s"、字面量 "%" → "%%")
      3. 兼容 Row：_CompatCursor → CompatRow
      4. autocommit=True：与现有"每条语句后 commit"的写法语义一致
      5. 重试：死锁 1213 / 锁等待 1205 / 断连 2006·2013 / 连接数满 1040
      6. init_command 固定 time_zone，避免应用本地时间与服务端时区错位
    """

    engine = "mysql"
    dialect = "mysql"

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None,
    ):
        try:
            import pymysql  # 延迟导入：纯 SQLite 环境无需安装
            from dbutils.pooled_db import PooledDB
        except ImportError as exc:  # pragma: no cover - 依赖缺失时给出明确指引
            raise RuntimeError(
                "DB_ENGINE=mysql 需要安装依赖：pip install pymysql dbutils cryptography"
            ) from exc

        self._pymysql = pymysql
        self._compat_cursor = _make_compat_cursor(pymysql)
        self.host = host or DB_MYSQL_HOST
        self.port = port or DB_MYSQL_PORT
        self.user = user or DB_MYSQL_USER
        self.password = DB_MYSQL_PASSWORD if password is None else password
        self.database = database or DB_MYSQL_DATABASE
        self._pool = PooledDB(
            creator=pymysql,
            maxconnections=DB_MYSQL_POOL_SIZE + DB_MYSQL_MAX_OVERFLOW,
            mincached=1,
            maxcached=DB_MYSQL_POOL_SIZE,
            blocking=True,
            ping=1,                      # 取连接前探活，自动剔除断连
            host=self.host,
            port=int(self.port),
            user=self.user,
            password=self.password,
            database=self.database,
            charset=DB_MYSQL_CHARSET,
            cursorclass=self._compat_cursor,
            autocommit=True,
            connect_timeout=DB_MYSQL_CONNECT_TIMEOUT,
            read_timeout=DB_MYSQL_READ_TIMEOUT,
            write_timeout=DB_MYSQL_WRITE_TIMEOUT,
            init_command=f"SET time_zone='{DB_MYSQL_TIMEZONE}'",
        )
        logger.info(
            f"[DB] MySQL 连接池就绪: {self.host}:{self.port}/{self.database} "
            f"(pool={DB_MYSQL_POOL_SIZE}+{DB_MYSQL_MAX_OVERFLOW})"
        )

    # -------- 7.1 连接 --------
    def connect(self):
        """从池中取一条连接（用完 close() 归还），返回 sqlite3 风格适配器"""
        return MySQLConnectionAdapter(self._pool.connection())

    @contextmanager
    def _acquire(self):
        conn = self._pool.connection()
        cursor = None
        try:
            cursor = conn.cursor()
            yield conn, cursor
        finally:
            if cursor is not None:
                try:
                    cursor.close()
                except Exception:
                    pass
            conn.close()

    # -------- 7.2 查询 --------
    def query(self, sql: str, params: Sequence[Any] = ()) -> list:
        def _do() -> list:
            with self._acquire() as (_conn, cursor):
                # 方言适配在 _CompatCursor.execute 内完成，这里传原始 SQL
                cursor.execute(sql, params or None)
                return cursor.fetchall()

        return _run_with_retry(_do, op="query", retryable=_is_mysql_retryable)

    def query_one(self, sql: str, params: Sequence[Any] = ()) -> Optional[Any]:
        def _do():
            with self._acquire() as (_conn, cursor):
                cursor.execute(sql, params or None)
                return cursor.fetchone()

        return _run_with_retry(_do, op="query_one", retryable=_is_mysql_retryable)

    # -------- 7.3 写入 --------
    def execute(self, sql: str, params: Sequence[Any] = ()) -> int:
        def _do() -> int:
            with self._acquire() as (conn, cursor):
                cursor.execute(sql, params or None)
                # autocommit=True 时 commit 是空操作；显式调用兼容关闭 autocommit 的场景
                conn.commit()
                return cursor.rowcount

        return _run_with_retry(_do, op="execute", retryable=_is_mysql_retryable)

    def executescript(self, sql: str) -> None:
        statements = split_sql_statements(sql)
        for stmt in statements:
            def _do(stmt=stmt) -> None:
                with self._acquire() as (conn, cursor):
                    cursor.execute(stmt)
                    conn.commit()

            _run_with_retry(_do, op="executescript", retryable=_is_mysql_retryable)

    # -------- 7.4 事务 --------
    @contextmanager
    def transaction(self) -> Iterator["Database"]:
        conn = self._pool.connection()
        cursor = None
        try:
            conn.autocommit(False)
            cursor = conn.cursor()
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            if cursor is not None:
                try:
                    cursor.close()
                except Exception:
                    pass
            try:
                conn.autocommit(True)   # 归还池前恢复默认，避免污染下一位借用者
            except Exception:
                pass
            conn.close()

    # -------- 7.5 其它 --------
    def close(self) -> None:
        """关闭连接池（进程退出 / 测试清理时调用）"""
        try:
            self._pool.close()
        except Exception:
            pass

    def ping(self) -> bool:
        try:
            return self.query_one("SELECT 1 AS ok") is not None
        except Exception:
            return False

    def server_version(self) -> str:
        row = self.query_one("SELECT VERSION() AS v")
        return str(row["v"]) if row else ""


# ==================== 8. MySQL 只读连接（NL2SQL 执行边界） ====================
def open_mysql_readonly_connection(timeout_ms: int = 3000):
    """
    打开「会话只读」的 MySQL 连接
    ==============================
    1) 优先使用独立只读账号（DB_MYSQL_READONLY_USER，库级 GRANT SELECT）
    2) 未配置只读账号时，用主账号 + SET SESSION TRANSACTION READ ONLY 兜底
    3) SET SESSION MAX_EXECUTION_TIME：与服务端侧超时对齐（毫秒）
    4) 不经连接池：避免只读会话状态污染共享连接
    返回：(connection, mode_desc)
    """
    import pymysql

    user = DB_MYSQL_READONLY_USER or DB_MYSQL_USER
    password = DB_MYSQL_READONLY_PASSWORD if DB_MYSQL_READONLY_USER else DB_MYSQL_PASSWORD
    conn = pymysql.connect(
        host=DB_MYSQL_HOST,
        port=int(DB_MYSQL_PORT),
        user=user,
        password=password,
        database=DB_MYSQL_DATABASE,
        charset=DB_MYSQL_CHARSET,
        cursorclass=_make_compat_cursor(pymysql),
        autocommit=True,
        connect_timeout=DB_MYSQL_CONNECT_TIMEOUT,
        read_timeout=DB_MYSQL_READ_TIMEOUT,
        write_timeout=DB_MYSQL_WRITE_TIMEOUT,
        init_command=f"SET time_zone='{DB_MYSQL_TIMEZONE}'",
    )
    try:
        cur = conn.cursor()
        try:
            cur.execute("SET SESSION TRANSACTION READ ONLY")
            cur.execute(f"SET SESSION MAX_EXECUTION_TIME={max(100, int(timeout_ms))}")
        finally:
            cur.close()
    except Exception:
        conn.close()
        raise
    mode = "mysql-ro-account" if DB_MYSQL_READONLY_USER else "mysql-ro-session"
    return conn, mode


# ==================== 9. 工厂：引擎切换位 ====================
_instances: dict = {}


def get_database() -> Database:
    """
    按 config.DB_ENGINE 返回数据库实例（迁移切换点）
    ===================================================
    业务代码只调用 get_database()，不关心底层是 SQLite 还是 MySQL；
    切换数据库只需修改环境变量 DB_ENGINE。
    """
    global _instances
    if DB_ENGINE not in _instances:
        if DB_ENGINE == "sqlite":
            _instances[DB_ENGINE] = SQLiteDatabase()
        elif DB_ENGINE == "mysql":
            _instances[DB_ENGINE] = MySQLDatabase()
        else:
            raise ValueError(f"未知 DB_ENGINE: {DB_ENGINE}（支持 sqlite / mysql）")
        logger.info(
            f"[DB] 已选择数据库引擎: {_instances[DB_ENGINE].engine}"
            f"（切换位：DB_ENGINE={DB_ENGINE}）"
        )
    return _instances[DB_ENGINE]


def reset_database() -> None:
    """清空引擎实例缓存（测试 / 运行时切库用）"""
    for db in _instances.values():
        try:
            db.close()
        except Exception:
            pass
    _instances.clear()


def load_sql_resource(rel_path: str) -> str:
    """读取 SQL 资源文件（相对 AI_CRM 目录）"""
    base = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base, rel_path)
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def load_mysql_schema() -> str:
    """读取 MySQL 建表脚本（相对 AI_CRM 目录）"""
    return load_sql_resource(DB_MYSQL_SCHEMA_FILE)
