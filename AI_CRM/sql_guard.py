"""
sql_guard.py - SQL 安全护栏（AST 解析版）
==========================================
把 SQL 安全边界从「字符串黑名单」升级为「AST 白名单」。
这是二面复盘中确认的生产级方向，本模块把它真正落地为可运行代码。

分层防线（每一层都对应一个可讲清的设计理由）：
  L1  Statement Allowlist : 只允许 1 条 SELECT（含 WITH ... SELECT）
  L2  Table Allowlist     : 只允许业务表白名单
  L3  Column Allowlist    : 只允许白名单列；敏感列（密码/令牌）永久拒绝
  L4  Function Allowlist  : 只允许安全的标量/聚合函数（挡 load_extension 等）
  L5  Resource Limit      : 强制 LIMIT，且不超过 max_rows（防全表拉取）
  L6  Read-only Boundary  : 只读连接 file:...?mode=ro&uri=True + PRAGMA query_only
  L7  Timeout & Audit     : progress_handler 超时中断 + 结构化审计日志

关键结论（回答面试追问用）：
  - 参数化查询只解决「SQL 模板已定时，外部输入的 Value 不被解释为 SQL 语法」；
    当整条 SQL 由 LLM 生成时，表名/列名/JOIN/UNION/函数/ORDER BY/LIMIT 仍是
    模型生成的结构，参数化无法证明这些结构安全。
  - 因此真正的安全边界是「解析成 AST 后按白名单校验」，而不是「检测字符串里
    有没有危险关键字」——黑名单永远会漏（大小写、注释、编码、方言差异）。

依赖：仅标准库。若环境里装了 sqlglot，会额外做一次「第二解析器」交叉校验
      （可选，装了更好，不装不影响运行）。
"""

# ==================== 1. 导入依赖 ====================
from __future__ import annotations

import json
import os
import re
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Set, Tuple
from urllib.parse import quote

try:  # 允许在无 logger 的最小环境下独立运行/自测
    from logger import logger
except Exception:  # pragma: no cover
    import logging

    logger = logging.getLogger("sql_guard")

try:  # 指标埋点（可观测性增强）；缺失时降级为 no-op，保证独立自测可运行
    from metrics import incr as _metric_incr
except Exception:  # pragma: no cover
    def _metric_incr(*args, **kwargs):  # type: ignore[misc]
        return None

# 当前 SQL 方言：跟随 DB_ENGINE（sqlite / mysql）。
# 用途：函数白名单、第二解析器 sqlglot 的 read=、只读连接方式。
try:
    from config import DB_ENGINE as _DB_ENGINE
    SQL_DIALECT: str = _DB_ENGINE if _DB_ENGINE in ("sqlite", "mysql") else "sqlite"
except Exception:  # pragma: no cover - 独立运行 sql_guard 自测时无 config
    SQL_DIALECT = "sqlite"


# ==================== 2. 词法 / 语法常量 ====================
# 标识符起始字符：Unicode 字母或下划线（兼容中文列别名，如 `AS 客户数`）
_IDENT_START = re.compile(r"[^\W\d]", re.UNICODE)
_IDENT_BODY = re.compile(r"\w", re.UNICODE)

# SELECT 子句里会正常出现的关键字（这些不做「列引用」处理）
KEYWORDS: Set[str] = {
    "SELECT", "FROM", "WHERE", "ON", "USING", "AS", "AND", "OR", "NOT",
    "NULL", "IS", "IN", "LIKE", "GLOB", "BETWEEN", "EXISTS",
    "GROUP", "BY", "ORDER", "HAVING", "LIMIT", "OFFSET", "DISTINCT",
    "UNION", "INTERSECT", "EXCEPT", "ALL",
    "CASE", "WHEN", "THEN", "ELSE", "END", "ASC", "DESC",
    "WITH", "RECURSIVE", "CAST",
    "JOIN", "LEFT", "RIGHT", "INNER", "OUTER", "FULL", "CROSS", "NATURAL",
    # MySQL 方言关键字（避免被当成列名误判）
    "REGEXP", "RLIKE", "XOR", "DIV", "MOD", "BINARY", "INTERVAL", "SEPARATOR",
}

# 非只读 / 危险语句关键字：出现在任何位置都直接拒绝
FORBIDDEN_KEYWORDS: Set[str] = {
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "TRUNCATE",
    "ATTACH", "DETACH", "PRAGMA", "VACUUM", "REINDEX",
    "GRANT", "REVOKE", "EXEC", "EXECUTE",
    "BEGIN", "COMMIT", "ROLLBACK", "SAVEPOINT", "RELEASE", "ANALYZE",
}

# 表引用解析时，遇到这些关键字说明「表/别名列表结束」
_TABLE_STOP: Set[str] = {
    "ON", "USING", "WHERE", "GROUP", "ORDER", "HAVING", "LIMIT",
    "UNION", "INTERSECT", "EXCEPT",
    "JOIN", "LEFT", "RIGHT", "INNER", "OUTER", "FULL", "CROSS", "NATURAL",
}

# 允许调用的函数白名单（挡掉 load_extension/readfile/writefile/randomblob 等）
FUNCTION_ALLOWLIST: Set[str] = {
    # 聚合
    "count", "sum", "avg", "min", "max", "total", "group_concat",
    # 标量 · 字符串
    "length", "lower", "upper", "substr", "substring", "trim", "ltrim", "rtrim",
    "replace", "instr", "hex", "char", "unicode", "printf",
    # 标量 · 条件/类型
    "coalesce", "ifnull", "nullif", "cast", "typeof",
    # 标量 · 数值
    "abs", "round", "ceil", "floor",
    # 日期
    "date", "time", "datetime", "strftime", "julianday", "unixepoch",
}

# MySQL 方言函数白名单（切库后 NL2SQL 会生成 DATE_FORMAT/NOW() 这类函数）
# 原则：只放"只读、无副作用、不触达文件系统"的函数；
#       LOAD_FILE / SLEEP / BENCHMARK / GET_LOCK 等一律不放。
FUNCTION_ALLOWLIST_MYSQL: Set[str] = {
    # 聚合
    "count", "sum", "avg", "min", "max", "group_concat",
    # 字符串
    "length", "char_length", "character_length", "lower", "upper", "trim",
    "ltrim", "rtrim", "replace", "locate", "instr", "substring", "substr",
    "left", "right", "concat", "concat_ws", "substring_index", "lpad", "rpad",
    "hex", "md5", "format",
    # 条件 / 类型
    "coalesce", "ifnull", "nullif", "if", "isnull", "cast", "convert",
    # 数值
    "abs", "round", "ceil", "ceiling", "floor", "truncate", "mod", "pow",
    "power", "sqrt", "greatest", "least",
    # 日期时间
    "now", "curdate", "curtime", "current_date", "current_time",
    "current_timestamp", "date", "time", "year", "month", "day", "hour",
    "minute", "second", "date_format", "date_add", "date_sub", "datediff",
    "timestampdiff", "unix_timestamp", "from_unixtime", "last_day",
    # JSON（tags 字段是 JSON 文本）
    "json_extract", "json_unquote", "json_length", "json_contains",
    "json_valid", "json_type",
}

# 系统库/系统表：任何方言都不允许 NL2SQL 访问
FORBIDDEN_SCHEMAS: Set[str] = {
    "information_schema", "mysql", "performance_schema", "sys",
}

# ==================== 3. 表 / 列白名单（唯一事实源，与建表语句对齐） ====================
# 说明：只登记「业务可查表」；password_reset_tokens / session_tokens 不在白名单内
#       （认证令牌表根本不对外开放查询）。
TABLE_COLUMNS: Dict[str, Set[str]] = {
    "users": {
        "user_id", "name", "phone", "created_at", "updated_at",
    },
    "messages": {
        "message_id", "session_id", "user_id", "role", "content", "created_at",
    },
    "customer_profiles": {
        "customer_id", "name", "company", "position", "phone", "email",
        "notes", "tags", "created_at", "updated_at",
        # K12 教育业务字段
        "age", "identity", "school_grade", "intent_course", "matched_course",
        "intent_price", "hourly_fee", "active_status", "advisor_id", "source",
        "last_follow_up",
    },
    "accounts": {
        "account_id", "username", "role", "phone", "email", "is_active",
        "last_login", "created_at", "updated_at",
    },
    "sessions": {
        "session_id", "account_id", "title", "created_at", "updated_at",
    },
    "login_logs": {
        "log_id", "account_id", "username", "action", "ip_address",
        "detail", "created_at",
    },
}

# 单源派生：允许访问的表 = 表-列白名单的键
ALLOWED_TABLES: Set[str] = set(TABLE_COLUMNS.keys())

# 敏感列：即使出现在白名单表里也永久拒绝（纵深防御）
FORBIDDEN_COLUMNS: Set[str] = {
    "password_hash", "password", "passwd", "pwd",
    "token", "code", "secret", "api_key", "salt",
}


# ==================== 4. 词法分析（SQL → Token 流） ====================
@dataclass
class Token:
    """SQL 词法单元"""
    kind: str          # ident | qident | string | number | param | punct | comment
    value: str         # 归一化值（qident 已去掉引号、已还原 '' 转义）
    pos: int = 0       # 在原 SQL 中的起始下标（用于 LIMIT 改写）


def tokenize(sql: str) -> Tuple[List[Token], Optional[str]]:
    """
    把 SQL 切成 Token 流（手写词法分析器）
    ======================================
    返回：(tokens, error)
      error=None 表示成功；否则为词法错误原因。

    覆盖：空白 / 行注释 / 块注释 / 字符串字面量 / 引号标识符 /
          数字 / 标识符（含中文）/ 参数占位符 / 标点
    """
    tokens: List[Token] = []
    i, n = 0, len(sql)
    while i < n:
        ch = sql[i]

        # -------- 空白 --------
        if ch.isspace():
            i += 1
            continue

        # -------- 行注释 -- ... --------
        if ch == "-" and i + 1 < n and sql[i + 1] == "-":
            tokens.append(Token("comment", "--", i))
            j = sql.find("\n", i)
            i = n if j == -1 else j + 1
            continue

        # -------- 块注释 /* ... */ --------
        if ch == "/" and i + 1 < n and sql[i + 1] == "*":
            j = sql.find("*/", i + 2)
            if j == -1:
                return tokens, "未闭合的块注释 /*"
            tokens.append(Token("comment", "/*", i))
            i = j + 2
            continue

        # -------- 字符串字面量 '...' --------
        if ch == "'":
            j = i + 1
            while j < n:
                if sql[j] == "'":
                    if j + 1 < n and sql[j + 1] == "'":   # '' 转义
                        j += 2
                        continue
                    break
                j += 1
            if j >= n:
                return tokens, "未闭合的字符串字面量 '"
            tokens.append(Token("string", sql[i:j + 1], i))
            i = j + 1
            continue

        # -------- 引号标识符 "..." / `...` --------
        if ch in ('"', "`"):
            q = ch
            j = i + 1
            while j < n:
                if sql[j] == q:
                    if j + 1 < n and sql[j + 1] == q:      # "" 转义
                        j += 2
                        continue
                    break
                j += 1
            if j >= n:
                return tokens, f"未闭合的引号标识符 {q}"
            tokens.append(Token("qident", sql[i + 1:j].replace(q * 2, q), i))
            i = j + 1
            continue

        # -------- 数字 --------
        if ch.isdigit():
            j = i
            while j < n and (sql[j].isdigit() or sql[j] == "."):
                j += 1
            tokens.append(Token("number", sql[i:j], i))
            i = j
            continue

        # -------- 标识符 / 关键字（含中文） --------
        if _IDENT_START.match(ch):
            j = i + 1
            while j < n and (_IDENT_BODY.match(sql[j]) or sql[j] == "$"):
                j += 1
            tokens.append(Token("ident", sql[i:j], i))
            i = j
            continue

        # -------- 位置参数 ? --------
        if ch == "?":
            tokens.append(Token("param", "?", i))
            i += 1
            continue

        # -------- 命名参数 :name / @name / $name --------
        if ch in (":", "@", "$") and i + 1 < n and _IDENT_START.match(sql[i + 1]):
            j = i + 1
            while j < n and _IDENT_BODY.match(sql[j]):
                j += 1
            tokens.append(Token("param", sql[i:j], i))
            i = j
            continue

        # -------- 标点 --------
        if ch in "(),;.*=<>!+-/%|~[]":
            tokens.append(Token("punct", ch, i))
            i += 1
            continue

        # -------- 无法识别 --------
        return tokens, f"存在无法识别的字符: {ch!r}"

    return tokens, None


# ==================== 5. 校验结果 / 内部扫描结果 ====================
@dataclass
class ValidationResult:
    """SQL 校验结果（既给执行器用，也给审计日志用）"""
    ok: bool
    sql: str                                        # 可能被改写过的 SQL（含 LIMIT）
    error: str = ""
    tables: Set[str] = field(default_factory=set)
    columns: Set[str] = field(default_factory=set)
    functions: Set[str] = field(default_factory=set)
    limit: Optional[int] = None                     # 最终生效的 LIMIT
    limit_injected: bool = False                    # 是否由护栏补上 LIMIT
    limit_clamped: bool = False                     # 是否因超限被收敛

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "error": self.error,
            "tables": sorted(self.tables),
            "columns": sorted(self.columns),
            "functions": sorted(self.functions),
            "limit": self.limit,
            "limit_injected": self.limit_injected,
            "limit_clamped": self.limit_clamped,
        }


@dataclass
class _ScanResult:
    """一次遍历收集到的结构信息（跨子查询合并）"""
    tables: Set[str] = field(default_factory=set)
    table_alias: Dict[str, str] = field(default_factory=dict)   # 别名/表名 → 真实表名
    columns: Set[str] = field(default_factory=set)
    qualified: Set[Tuple[str, str]] = field(default_factory=set)  # (限定符, 列)
    aliases: Set[str] = field(default_factory=set)               # SELECT ... AS x / 表别名
    cte_names: Set[str] = field(default_factory=set)             # WITH x AS (...)
    funcs: Set[str] = field(default_factory=set)
    dangerous: Optional[str] = None
    has_set_op: bool = False
    error: Optional[str] = None

    def merge(self, other: "_ScanResult") -> None:
        self.tables |= other.tables
        self.table_alias.update(other.table_alias)
        self.columns |= other.columns
        self.qualified |= other.qualified
        self.aliases |= other.aliases
        self.cte_names |= other.cte_names
        self.funcs |= other.funcs
        if other.dangerous and not self.dangerous:
            self.dangerous = other.dangerous
        self.has_set_op = self.has_set_op or other.has_set_op


# ==================== 6. 审计日志 ====================
_AUDIT_PATH = os.environ.get(
    "SQL_AUDIT_LOG",
    os.path.join(os.environ.get("LOG_DIR", "logfile"), "sql_audit.log"),
)


def audit_sql(event: str, sql: str, ok: bool, detail: str = "", **extra) -> None:
    """
    结构化 SQL 审计（谁、什么时候、执行了什么、被允许/拒绝、为什么）
    ==============================================================
    写入两处：logger（随应用日志滚动）+ 独立审计文件（append-only JSONL）。
    审计失败绝不影响主流程（用 try 包住）。
    """
    record = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "event": event,          # allow | reject | error
        "ok": ok,
        "detail": detail,
        "sql": " ".join((sql or "").split())[:800],
    }
    record.update(extra)
    line = json.dumps(record, ensure_ascii=False)
    try:
        logger.info(f"[AUDIT][SQL] {line}")
    except Exception:  # pragma: no cover
        pass
    try:
        os.makedirs(os.path.dirname(_AUDIT_PATH) or ".", exist_ok=True)
        with open(_AUDIT_PATH, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except Exception:  # pragma: no cover
        pass


# ==================== 7. 只读执行边界（含超时） ====================
def open_readonly_connection(
    db_path: Optional[str] = None,
    busy_timeout_ms: int = 5000,
    timeout_ms: int = 3000,
) -> Tuple[sqlite3.Connection, str]:
    """
    打开「只读 + 超时」连接（Read-only Execution Boundary）
    ======================================================
    1) file:{path}?mode=ro&uri=True  → 文件层只读（写入直接报错，而非靠 SQL 文本判断）
    2) PRAGMA query_only=ON          → 连接层只读（双保险）
    3) PRAGMA busy_timeout=N         → 锁等待上限
    4) set_progress_handler          → 执行超时中断（防慢查询拖死进程）

    返回：(connection, mode_desc)；失败抛异常，由调用方回退到普通连接。
    """
    # -------- MySQL：只读账号 + 会话只读 + 执行超时 --------
    if SQL_DIALECT == "mysql":
        from db import open_mysql_readonly_connection
        return open_mysql_readonly_connection(timeout_ms=timeout_ms)

    if not db_path:
        raise ValueError("缺少 SQLite 数据库路径，无法建立只读连接")
    abs_path = os.path.abspath(db_path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"数据库文件不存在: {abs_path}")

    uri = quote(abs_path.replace("\\", "/"), safe="/:")
    conn = sqlite3.connect(
        f"file:{uri}?mode=ro",
        uri=True,
        check_same_thread=False,
    )
    conn.row_factory = sqlite3.Row
    conn.execute(f"PRAGMA busy_timeout={int(busy_timeout_ms)};")
    try:
        conn.execute("PRAGMA query_only=ON;")
    except Exception:  # pragma: no cover
        pass

    # -------- 超时中断：每 2000 条 VM 指令检查一次截止时间 --------
    deadline = time.monotonic() + max(0.1, timeout_ms / 1000.0)

    def _deadline_reached() -> int:
        return 1 if time.monotonic() > deadline else 0

    conn.set_progress_handler(_deadline_reached, 2000)
    return conn, f"ro-uri (timeout={timeout_ms}ms)"


# ==================== 8. AST 护栏 ====================
class SqlGuard:
    """
    SQL AST 护栏
    =============
    用法：
        guard = SqlGuard(max_rows=200)
        res = guard.validate("SELECT name FROM customer_profiles")
        if not res.ok:
            print(res.error)
        else:
            cur.execute(res.sql)   # res.sql 已保证带 LIMIT
    """

    def __init__(
        self,
        allowed_tables: Optional[Sequence[str]] = None,
        table_columns: Optional[Dict[str, Set[str]]] = None,
        function_allowlist: Optional[Sequence[str]] = None,
        forbidden_columns: Optional[Sequence[str]] = None,
        max_rows: int = 200,
        max_subquery_depth: int = 3,
        allow_set_operations: bool = False,
        use_sqlglot: bool = True,
    ):
        self.table_columns: Dict[str, Set[str]] = {
            k: set(v) for k, v in (table_columns or TABLE_COLUMNS).items()
        }
        self.allowed_tables: Set[str] = set(
            allowed_tables or self.table_columns.keys()
        )
        self.function_allowlist: Set[str] = set(
            function_allowlist
            or (FUNCTION_ALLOWLIST_MYSQL if SQL_DIALECT == "mysql" else FUNCTION_ALLOWLIST)
        )
        self.forbidden_columns: Set[str] = set(
            forbidden_columns or FORBIDDEN_COLUMNS
        )
        self.max_rows = int(max_rows)
        self.max_subquery_depth = int(max_subquery_depth)
        self.allow_set_operations = bool(allow_set_operations)
        self.use_sqlglot = bool(use_sqlglot)

    # -------- 8.1 工具函数 --------
    @staticmethod
    def _next_sig(toks: List[Token], k: int) -> int:
        """返回 k 之后第一个非注释 Token 的下标"""
        n = len(toks)
        while k < n and toks[k].kind == "comment":
            k += 1
        return k

    @staticmethod
    def _match_paren(toks: List[Token], i: int) -> int:
        """i 指向 '(' 时，返回匹配 ')' 的下标；不匹配返回 -1"""
        depth = 0
        for j in range(i, len(toks)):
            t = toks[j]
            if t.kind == "punct" and t.value == "(":
                depth += 1
            elif t.kind == "punct" and t.value == ")":
                depth -= 1
                if depth == 0:
                    return j
        return -1

    @staticmethod
    def _split_statements(toks: List[Token]) -> List[List[Token]]:
        """按顶层分号切分语句（返回去空后的语句列表）"""
        stmts: List[List[Token]] = []
        cur: List[Token] = []
        depth = 0
        for t in toks:
            if t.kind == "punct" and t.value == "(":
                depth += 1
            elif t.kind == "punct" and t.value == ")":
                depth -= 1
            if t.kind == "punct" and t.value == ";" and depth == 0:
                stmts.append(cur)
                cur = []
                continue
            cur.append(t)
        stmts.append(cur)
        return [s for s in stmts if any(x.kind != "comment" for x in s)]

    # -------- 8.2 表引用解析 --------
    def _consume_one_table(self, toks: List[Token], k: int, depth: int,
                           acc: _ScanResult) -> int:
        """
        解析单个表引用（表名 / 子查询），并登记别名。
        返回：该表引用结束后的下标；解析失败返回 -1（acc.error 已置位）。
        """
        n = len(toks)
        if k >= n:
            acc.error = "FROM/JOIN 之后缺少表"
            return -1
        tk = toks[k]

        # -------- 情形 A：子查询 / 括号连接表 --------
        if tk.kind == "punct" and tk.value == "(":
            j = self._match_paren(toks, k)
            if j < 0:
                acc.error = "括号不匹配"
                return -1
            self._walk(toks[k + 1:j], depth + 1, acc)
            if acc.error:
                return -1
            after = j + 1
            a = self._next_sig(toks, after)
            if a < n and toks[a].kind == "ident" and toks[a].value.upper() == "AS":
                a2 = self._next_sig(toks, a + 1)
                if a2 < n and toks[a2].kind in ("ident", "qident"):
                    acc.aliases.add(toks[a2].value.lower())
                    return a2 + 1
                return a + 1
            if (a < n and toks[a].kind in ("ident", "qident")
                    and toks[a].value.upper() not in KEYWORDS
                    and toks[a].value.upper() not in _TABLE_STOP):
                acc.aliases.add(toks[a].value.lower())
                return a + 1
            return after

        # -------- 情形 B：普通表名（可能带 schema 前缀） --------
        if tk.kind in ("ident", "qident"):
            parts = [tk.value.lower()]
            j = self._next_sig(toks, k + 1)
            while j < n and toks[j].kind == "punct" and toks[j].value == ".":
                j2 = self._next_sig(toks, j + 1)
                if j2 < n and toks[j2].kind in ("ident", "qident"):
                    parts.append(toks[j2].value.lower())
                    j = self._next_sig(toks, j2 + 1)
                else:
                    break
            table = parts[-1]

            # 表值函数（如 pragma_table_info('x')）一律拒绝
            if j < n and toks[j].kind == "punct" and toks[j].value == "(":
                acc.error = f"不允许表值函数: {table}()"
                return -1

            acc.tables.add(table)
            acc.table_alias[table] = table

            # 可选别名：AS x  /  裸 x
            if j < n and toks[j].kind == "ident" and toks[j].value.upper() == "AS":
                j2 = self._next_sig(toks, j + 1)
                if j2 < n and toks[j2].kind in ("ident", "qident"):
                    acc.table_alias[toks[j2].value.lower()] = table
                    return j2 + 1
                return j + 1
            if (j < n and toks[j].kind in ("ident", "qident")
                    and toks[j].value.upper() not in KEYWORDS
                    and toks[j].value.upper() not in _TABLE_STOP):
                acc.table_alias[toks[j].value.lower()] = table
                return j + 1
            return j

        acc.error = f"无法解析表引用: {tk.value!r}"
        return -1

    def _consume_table_list(self, toks: List[Token], i: int, depth: int,
                            acc: _ScanResult) -> int:
        """
        解析 FROM / JOIN 后的表列表（支持逗号分隔的多表；JOIN 由外层循环续接）。
        返回：列表结束后的下标；失败返回 -1。
        """
        n = len(toks)
        k = self._next_sig(toks, i + 1)
        k = self._consume_one_table(toks, k, depth, acc)
        if k < 0:
            return -1
        # 逗号分隔的后续表
        while True:
            j = self._next_sig(toks, k)
            if j < n and toks[j].kind == "punct" and toks[j].value == ",":
                k = self._consume_one_table(toks, self._next_sig(toks, j + 1),
                                            depth, acc)
                if k < 0:
                    return -1
                continue
            return k

    # -------- 8.3 主遍历（递归处理子查询） --------
    def _walk(self, toks: List[Token], depth: int, acc: _ScanResult) -> None:
        """遍历 Token 流，收集表/列/函数/别名信息"""
        if depth > self.max_subquery_depth:
            acc.error = f"子查询嵌套超过上限（{self.max_subquery_depth}）"
            return

        n = len(toks)
        i = 0
        while i < n:
            t = toks[i]

            # -------- 注释：忽略 --------
            if t.kind == "comment":
                i += 1
                continue

            # -------- 括号组：函数参数 / 子查询 / 表达式 --------
            if t.kind == "punct" and t.value == "(":
                j = self._match_paren(toks, i)
                if j < 0:
                    acc.error = "括号不匹配"
                    return
                self._walk(toks[i + 1:j], depth + 1, acc)
                if acc.error:
                    return
                # 关闭括号后的别名（子查询 AS x 已在表引用处处理，这里兜底）
                k = self._next_sig(toks, j + 1)
                if (k < n and toks[k].kind == "ident"
                        and toks[k].value.upper() == "AS"):
                    k2 = self._next_sig(toks, k + 1)
                    if k2 < n and toks[k2].kind in ("ident", "qident"):
                        acc.aliases.add(toks[k2].value.lower())
                        i = k2 + 1
                        continue
                i = j + 1
                continue

            if t.kind in ("ident", "qident"):
                up = t.value.upper()
                low = t.value.lower()

                # -------- 危险语句关键字 --------
                if t.kind == "ident" and up in FORBIDDEN_KEYWORDS:
                    acc.dangerous = up
                    i += 1
                    continue

                # -------- 集合运算 --------
                if t.kind == "ident" and up in ("UNION", "INTERSECT", "EXCEPT"):
                    acc.has_set_op = True
                    i += 1
                    continue

                # -------- CTE：ident [ (列清单) ] AS ( --------
                if t.kind == "ident" and up == "WITH":
                    k = self._next_sig(toks, i + 1)
                    if (k < n and toks[k].kind == "ident"
                            and toks[k].value.upper() == "RECURSIVE"):
                        k = self._next_sig(toks, k + 1)
                    if k < n and toks[k].kind in ("ident", "qident"):
                        name = toks[k].value.lower()
                        m = self._next_sig(toks, k + 1)
                        if m < n and toks[m].kind == "punct" and toks[m].value == "(":
                            mm = self._match_paren(toks, m)
                            inner_first = self._next_sig(toks, m + 1)
                            is_body = (mm >= 0 and inner_first < n
                                       and toks[inner_first].kind in ("ident", "qident")
                                       and toks[inner_first].value.upper() in ("SELECT", "WITH"))
                            if not is_body:      # 列清单 → 跳过
                                acc.cte_names.add(name)
                                i = (mm + 1) if mm >= 0 else m + 1
                                continue
                        acc.cte_names.add(name)
                        acc.aliases.add(name)
                        i = k + 1
                        continue
                    i += 1
                    continue

                # -------- FROM / JOIN：解析表列表 --------
                if t.kind == "ident" and up in ("FROM", "JOIN"):
                    nxt = self._consume_table_list(toks, i, depth, acc)
                    if nxt < 0:
                        return
                    i = nxt
                    continue

                # -------- AS 别名：不作为列 --------
                if t.kind == "ident" and up == "AS":
                    k = self._next_sig(toks, i + 1)
                    if k < n and toks[k].kind in ("ident", "qident"):
                        acc.aliases.add(toks[k].value.lower())
                        i = k + 1
                        continue
                    i += 1
                    continue

                # -------- 函数调用？ident 紧跟 '(' --------
                # 顺序很重要：白名单函数优先识别（COUNT/SUM/CAST…），
                # 关键字（IN/EXISTS/CAST…）次之——避免把 `IN (` 误判成函数 `in`；
                # 只有「既非白名单函数、又非关键字」的调用才登记为函数，
                # 从而被 Function Allowlist 拦下（如 load_extension）。
                k = self._next_sig(toks, i + 1)
                is_call = (k < n and toks[k].kind == "punct"
                           and toks[k].value == "(")
                if is_call and low in self.function_allowlist:
                    acc.funcs.add(low)
                    i += 1
                    continue

                # -------- 普通关键字 --------
                if t.kind == "ident" and up in KEYWORDS:
                    i += 1
                    continue

                # -------- 未知函数调用：登记后由白名单拒绝 --------
                if is_call:
                    acc.funcs.add(low)
                    i += 1
                    continue

                # -------- 限定符 t.col：当前是表/别名 --------
                if k < n and toks[k].kind == "punct" and toks[k].value == ".":
                    i += 1
                    continue

                # -------- 列引用（含 t.col 形式） --------
                prev = i - 1
                while prev >= 0 and toks[prev].kind == "comment":
                    prev -= 1
                if (prev >= 0 and toks[prev].kind == "punct"
                        and toks[prev].value == "."):
                    pprev = prev - 1
                    while pprev >= 0 and toks[pprev].kind == "comment":
                        pprev -= 1
                    if pprev >= 0 and toks[pprev].kind in ("ident", "qident"):
                        acc.qualified.add((toks[pprev].value.lower(), low))
                acc.columns.add(low)
                i += 1
                continue

            i += 1

    # -------- 8.4 LIMIT 处理（Resource Limit） --------
    def _apply_limit(self, toks: List[Token], sql: str
                     ) -> Tuple[Optional[str], Optional[str], bool, bool, Optional[int]]:
        """
        强制 / 收敛 LIMIT。
        返回：(改写后 SQL, 错误, 是否补入, 是否收敛, 最终 LIMIT)
        """
        n = len(toks)
        depth = 0
        limit_idx: Optional[int] = None
        for idx in range(n):
            t = toks[idx]
            if t.kind == "punct" and t.value == "(":
                depth += 1
            elif t.kind == "punct" and t.value == ")":
                depth -= 1
            elif depth == 0 and t.kind == "ident" and t.value.upper() == "LIMIT":
                limit_idx = idx

        base = sql.rstrip().rstrip(";").rstrip()

        # -------- 没有 LIMIT：自动补上 --------
        if limit_idx is None:
            return f"{base} LIMIT {self.max_rows}", None, True, False, self.max_rows

        # -------- 有 LIMIT：必须是整数字面量，且不超过上限 --------
        k = self._next_sig(toks, limit_idx + 1)
        if k >= n or toks[k].kind != "number":
            return None, "LIMIT 必须是整数字面量（不接受参数或表达式）", False, False, None
        try:
            val = int(float(toks[k].value))
        except ValueError:
            return None, f"LIMIT 值无法解析: {toks[k].value}", False, False, None

        if val > self.max_rows:
            rewritten = (sql[:toks[k].pos] + str(self.max_rows)
                         + sql[toks[k].pos + len(toks[k].value):])
            return (rewritten.rstrip().rstrip(";"), None, False, True, self.max_rows)
        return base, None, False, False, val

    # -------- 8.5 可选：sqlglot 第二解析器交叉校验 --------
    @staticmethod
    def _cross_check_sqlglot(sql: str) -> Optional[str]:
        """装了 sqlglot 才生效；作为「第二解析器」交叉验证，不装则跳过"""
        try:
            import sqlglot
            from sqlglot import exp
        except Exception:
            return None
        try:
            exprs = sqlglot.parse(sql, read=SQL_DIALECT)
        except Exception as exc:  # pragma: no cover
            return f"第二解析器(sqlglot) 解析失败: {exc}"
        if len(exprs) != 1:
            return f"第二解析器(sqlglot): 语句数={len(exprs)}（必须为 1）"
        node = exprs[0]
        if not isinstance(node, (exp.Select, exp.Union, exp.Intersect, exp.Except)):
            return f"第二解析器(sqlglot): 非 SELECT（{type(node).__name__}）"
        return None

    # -------- 8.6 对外主入口 --------
    def validate(self, sql: str) -> ValidationResult:
        """
        校验并规范化 SQL（对外入口，附带指标埋点）
        ================================
        返回 ValidationResult；ok=True 时 result.sql 已带合规 LIMIT。
        埋点：sql_check_total{verdict=allow|reject, stage=ast}
              → 用于计算 SQL 安全校验拒绝率（二面被追问的指标之一）。
        """
        res = self._validate_core(sql)
        _metric_incr(
            "sql_check_total",
            verdict="allow" if res.ok else "reject",
            stage="ast",
        )
        return res

    def _validate_core(self, sql: str) -> ValidationResult:
        """
        核心校验逻辑（AST 护栏）
        ================
        返回 ValidationResult；ok=True 时 result.sql 已带合规 LIMIT。
        """
        res = ValidationResult(ok=False, sql=sql or "")
        if not sql or not sql.strip():
            res.error = "空 SQL"
            return res

        # -------- L0（可选）：第二解析器交叉校验 --------
        if self.use_sqlglot:
            err = self._cross_check_sqlglot(sql)
            if err:
                res.error = err
                return res

        # -------- 词法 --------
        toks, err = tokenize(sql)
        if err:
            res.error = err
            return res

        # -------- L1：Statement Allowlist（只允许单条 SELECT） --------
        stmts = self._split_statements(toks)
        if not stmts:
            res.error = "空 SQL"
            return res
        if len(stmts) > 1:
            res.error = f"禁止多语句（检测到 {len(stmts)} 条）"
            return res
        seg = stmts[0]

        first = self._next_sig(seg, 0)
        if first >= len(seg):
            res.error = "空语句"
            return res
        ft = seg[first]
        if ft.kind not in ("ident", "qident"):
            res.error = f"非法语句起始: {ft.value!r}"
            return res
        fup = ft.value.upper()
        if fup in FORBIDDEN_KEYWORDS:
            res.error = f"Statement Allowlist 拒绝：{fup} 非只读语句"
            return res
        if fup not in ("SELECT", "WITH"):
            res.error = f"Statement Allowlist 仅允许 SELECT（收到 {fup}）"
            return res

        # -------- 结构遍历 --------
        acc = _ScanResult()
        self._walk(seg, 0, acc)
        if acc.error:
            res.error = acc.error
            return res
        if acc.dangerous:
            res.error = f"检测到非只读关键字：{acc.dangerous}"
            return res
        if acc.has_set_op and not self.allow_set_operations:
            res.error = "禁止集合运算（UNION / INTERSECT / EXCEPT）"
            return res

        # -------- L2：Table Allowlist --------
        unknown = acc.tables - self.allowed_tables - acc.cte_names
        if unknown:
            res.error = f"访问未授权表：{', '.join(sorted(unknown))}"
            return res

        # -------- L2b：系统库防护（information_schema / mysql / performance_schema / sys）--------
        # 说明：SQLite 靠 sqlite_ 前缀 + 白名单拦截；MySQL 还需显式拦系统库，
        #       否则 information_schema.tables 可能被当成普通限定表绕过白名单。
        bad_schemas = {
            t.value.lower() for t in seg
            if t.kind in ("ident", "qident") and t.value.lower() in FORBIDDEN_SCHEMAS
        }
        if bad_schemas:
            res.error = f"禁止访问系统库：{', '.join(sorted(bad_schemas))}"
            return res

        # -------- L4：Function Allowlist --------
        bad_funcs = {f for f in acc.funcs if f not in self.function_allowlist}
        if bad_funcs:
            res.error = f"调用未授权函数：{', '.join(sorted(bad_funcs))}"
            return res

        # -------- L3：Column Allowlist --------
        allowed_cols: Set[str] = set(acc.aliases) | set(acc.cte_names) | {"rowid"}
        for tbl in acc.tables:
            allowed_cols |= self.table_columns.get(tbl, set())

        bad_cols: Set[str] = set()

        # 3.1 敏感列：任何形式都拒绝
        for c in acc.columns:
            if c in self.forbidden_columns:
                bad_cols.add(c)

        # 3.2 限定列 t.col：优先按 t 解析到真实表
        for qualifier, col in acc.qualified:
            if col in self.forbidden_columns:
                bad_cols.add(col)
                continue
            table = acc.table_alias.get(qualifier)
            if table and table in self.table_columns:
                if col not in self.table_columns[table] and col not in acc.aliases:
                    bad_cols.add(f"{qualifier}.{col}")

        # 3.3 非限定列：必须存在于任一被引用表
        for col in acc.columns:
            if col in self.forbidden_columns:
                continue          # 已在 3.1 记录
            if col in acc.aliases or col in acc.cte_names or col == "rowid":
                continue
            if acc.tables:
                if any(col in self.table_columns.get(t, set()) for t in acc.tables):
                    continue
            bad_cols.add(col)

        if bad_cols:
            res.error = f"访问未授权列：{', '.join(sorted(bad_cols))}"
            return res

        # -------- L5：Resource Limit --------
        new_sql, lerr, injected, clamped, limit_val = self._apply_limit(seg, sql)
        if lerr:
            res.error = lerr
            return res

        res.ok = True
        res.sql = new_sql or sql
        res.tables = set(acc.tables)
        res.columns = set(acc.columns)
        res.functions = set(acc.funcs)
        res.limit = limit_val
        res.limit_injected = injected
        res.limit_clamped = clamped
        return res


# ==================== 9. 模块级默认护栏（开箱即用） ====================
DEFAULT_GUARD = SqlGuard(max_rows=200)


def validate_sql_ast(sql: str, guard: Optional[SqlGuard] = None) -> ValidationResult:
    """便捷函数：用默认（或指定）护栏校验 SQL"""
    return (guard or DEFAULT_GUARD).validate(sql)


if __name__ == "__main__":  # pragma: no cover
    # 简易自检：python sql_guard.py
    _cases = [
        "SELECT name FROM customer_profiles",
        "SELECT COUNT(*) AS 总数 FROM customer_profiles WHERE active_status='活跃'",
        "SELECT * FROM customer_profiles LIMIT 9999",
        "SELECT password_hash FROM accounts",
        "SELECT * FROM sqlite_master",
        "SELECT * FROM customer_profiles; DROP TABLE users",
        "SELECT load_extension('x')",
        "SELECT a.name FROM customer_profiles a JOIN users u ON a.name=u.name",
        "SELECT * FROM customer_profiles WHERE active_status IN (SELECT role FROM accounts)",
    ]
    for _sql in _cases:
        _r = validate_sql_ast(_sql)
        _flag = "PASS" if _r.ok else "BLOCK"
        print(f"[{_flag}] {_sql}")
        print(f"        -> error={_r.error or '-'} limit={_r.limit} "
              f"tables={sorted(_r.tables)}")
