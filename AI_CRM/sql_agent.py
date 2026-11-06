"""
sql_agent.py - SQL Agent 模块
====================================
功能分区：
  1. 自然语言 → SQL 转换（调用 LLM 生成 SQL）
  2. SQL 执行器（安全执行查询，只读模式）
  3. 查询结果格式化（表格 → 自然语言描述）

安全设计（两层 + 边界 + 审计）：
  【第一层】AST 白名单（真正的安全边界 · 已落地，见 sql_guard.py）
    [OK] Statement Allowlist：只允许 1 条 SELECT（含 WITH ... SELECT）
    [OK] Table Allowlist：只允许业务表白名单
    [OK] Column Allowlist：只允许白名单列；密码/令牌等敏感列永久拒绝
    [OK] Function Allowlist：只允许安全函数（挡 load_extension 等）
    [OK] Resource Limit：强制 LIMIT，且不超过 MAX_ROWS
  【第二层】字符串黑名单（历史实现 · 兼容性双保险）
    [OK] 禁止危险关键字（DROP/ALTER/INSERT 等）
    [OK] 禁止多语句执行（分号检测）
    [OK] 禁止 SQL 注释绕过（-- 和 /* */）
    [OK] 表名白名单（ALLOWED_TABLES，仅业务表）
  【执行边界】
    [OK] Read-only Boundary：只读连接（file:...?mode=ro）+ PRAGMA query_only
    [OK] Timeout：busy_timeout + progress_handler 超时中断
    [OK] Audit：允许 / 拒绝 / 异常全部落审计日志
  【写入链路】
    [OK] 结构化意图链路（服务端模板 + 参数化 SQL，Prompt 不作为安全边界）

为什么参数化还不够：
  参数化查询只保证「SQL 模板已定时，外部输入的 Value 不被解释为 SQL 语法」。
  当整条 SQL 由 LLM 生成时，表名 / 列名 / JOIN / UNION / 子查询 / 函数 /
  ORDER BY / LIMIT 这些「结构」同样是模型生成的，参数化无法为其背书；
  黑名单式的字符串检测也永远会漏。因此必须解析成 AST 后按白名单逐项校验。
"""

# ==================== 1. 导入依赖 ====================
import re                   # 正则表达式，用于提取 SQL
import time                 # 计时（审计：单次查询耗时）
from typing import Optional, List  # 类型提示

from db import get_database  # DAO 抽象层：统一连接/PRAGMA/引擎切换位
from logger import logger    # 日志记录器
# 客户字段更新权限白名单（服务端信任边界，与 Prompt 解耦）
from models import AUTO_UPDATE_FIELDS, MANUAL_REVIEW_FIELDS
# AST 安全护栏（生产级方向 · 已落地）：Statement/Table/Column/Function 白名单
#                            + Resource Limit + 只读连接 + 超时 + 审计
from sql_guard import (
    SqlGuard,
    TABLE_COLUMNS,
    validate_sql_ast,
    open_readonly_connection,
    audit_sql,
    SQL_DIALECT,
)
# 运行时指标埋点（可观测性增强：SQL 执行耗时 / 行数 / 拦截率）
from metrics import incr, observe


# ==================== 1b. SQL 安全校验函数 ====================
# 危险关键字黑名单（不区分大小写，全词匹配）
# 这些关键字在 SELECT 语句中也不应出现
DANGEROUS_KEYWORDS = [
    "DROP ", "ALTER ", "INSERT ", "UPDATE ", "DELETE ",
    "EXEC ", "EXECUTE ", "CREATE ", "TRUNCATE ", "REPLACE ",
    "ATTACH ", "DETACH ", "REINDEX ", "VACUUM", "PRAGMA",
    "GRANT ", "REVOKE ", "DENY ",
]

# 允许的表名白名单（防止访问系统表或其他数据库）
# 单一事实源：来自 sql_guard.TABLE_COLUMNS（表-列白名单的键集合），
#            避免「文档里一套、代码里一套」的漂移。
ALLOWED_TABLES = set(TABLE_COLUMNS.keys())

# 系统表前缀（禁止访问）：SQLite / PostgreSQL / MySQL 系统库
FORBIDDEN_TABLE_PREFIXES = (
    "sqlite_", "pg_", "mysql_",
    "information_schema", "performance_schema", "sys",
)

# 表名引用提取正则：匹配 FROM / JOIN 之后紧跟的表名
_TABLE_REF_PATTERN = re.compile(
    r"\b(?:FROM|JOIN)\s+([A-Za-z_][A-Za-z0-9_]*)",
    re.IGNORECASE,
)


def _extract_table_refs(sql_no_quotes: str) -> set:
    """
    提取 SQL 中 FROM / JOIN 引用的表名（统一小写）
    ============================================
    仅做粗粒度提取，用于表名白名单准入。真正的安全边界是
    "AST 白名单（sql_guard）+ 服务端模板 + 参数化 SQL"，
    此处为第二层字符串黑名单的辅助提取。
    """
    return {m.group(1).lower() for m in _TABLE_REF_PATTERN.finditer(sql_no_quotes)}


def _validate_sql_safe(sql: str) -> Optional[str]:
    """
    SQL 安全校验（第二层：字符串黑名单）
    ====================================
    返回：
      None — 校验通过
      str — 错误信息（校验失败原因）

    检查项：
      1. 是否以 SELECT 开头（忽略大小写和前后空白）
      2. 是否包含危险关键字
      3. 是否包含多语句（分号绕过检测）
      4. 是否包含 SQL 注释（注释绕过检测）
      5. 是否访问系统表
      6. 是否访问白名单外的表

    说明：本函数是历史实现，作为兼容性双保险保留；
          第一层 AST 白名单（_validate_sql_ast）才是真正的安全边界。
    """
    # -------- 检查 1: 必须以 SELECT 开头 --------
    sql_stripped = sql.strip()
    if not sql_stripped.upper().startswith("SELECT"):
        return "[FAIL] 只允许执行 SELECT 查询语句"
    
    # -------- 检查 2: 移文字符串字面量后再检查 --------
    # 移除单引号字符串（避免字符串内容误触发关键字检查）
    sql_no_strings = re.sub(r"'[^']*'", "", sql_stripped)
    # 移除双引号标识符
    sql_no_quotes = re.sub(r'"[^"]*"', "", sql_no_strings)
    
    # 统一转大写用于关键词匹配
    sql_upper = sql_no_quotes.upper()
    
    # -------- 检查 3: 禁止危险关键字 --------
    for kw in DANGEROUS_KEYWORDS:
        # 前后加空格确保全词匹配（避免误匹配列名中的子串）
        if f" {kw}" in f" {sql_upper} " or sql_upper.startswith(kw):
            return f"[FAIL] 禁止使用 '{kw.strip()}' 操作（仅允许只读查询）"
    
    # -------- 检查 4: 禁止多语句（通过分号检测）-------
    # 注意：SQLite 允许多语句由 sqlite3 的 executescript 执行
    # 但 cursor.execute() 只执行第一条语句
    # 然而为了多层防护，仍检测分号
    if ";" in sql_no_quotes.rstrip(";"):
        return "[FAIL] 禁止执行多条SQL语句"
    
    # -------- 检查 5: 禁止 SQL 注释（防止注释绕过）-------
    if "--" in sql_no_strings:
        return "[FAIL] SQL 中包含注释（--），存在注入风险"
    if "/*" in sql_no_strings:
        return "[FAIL] SQL 中包含块注释（/* */），存在注入风险"
    
    # -------- 检查 6: 禁止访问系统表 --------
    for prefix in FORBIDDEN_TABLE_PREFIXES:
        if prefix.upper() in sql_upper:
            return f"[FAIL] 禁止访问系统表（{prefix}*）"
    
    # -------- 检查 7: 表名白名单（仅允许业务表）--------
    referenced_tables = _extract_table_refs(sql_no_quotes)
    unauthorized = referenced_tables - ALLOWED_TABLES
    if unauthorized:
        return f"[FAIL] 访问了未授权的表: {', '.join(sorted(unauthorized))}"
    
    return None  # 校验通过


# ==================== 1b-2. AST 白名单护栏（第一层 · 生产级方向已落地） ====================
# 与上面字符串黑名单的关系：
#   黑名单（_validate_sql_safe）是历史实现，作为兼容性双保险保留；
#   真正的安全边界是 AST 白名单 —— 把 SQL 解析成语法树后，按
#   Statement / Table / Column / Function 白名单 + LIMIT 资源上限逐项校验。
#
# 为什么必须上 AST（面试要能讲清）：
#   LLM 生成整条 SQL 时，「参数化」只能保护 Value，保护不了表名 / 列名 /
#   JOIN / UNION / 子查询 / 函数 / ORDER BY / LIMIT 这些「结构」；
#   而「字符串里有没有危险关键字」的黑名单永远会漏（大小写、注释、方言差异）。
#   所以安全边界只能是：解析成 AST → 与白名单比对。
MAX_ROWS = 200                       # 单次查询最多返回行数（Resource Limit）
SQL_GUARD = SqlGuard(max_rows=MAX_ROWS)


def _validate_sql_ast(sql: str):
    """
    AST 白名单校验（第一层 · 真正安全边界）
    ========================================
    返回 ValidationResult：
      ok=True  → result.sql 已带合规 LIMIT（补入或收敛到 ≤ MAX_ROWS）
      ok=False → result.error 为拒绝原因
    """
    return validate_sql_ast(sql, guard=SQL_GUARD)


# ==================== 1c. 结构化意图链路（Schema 校验 + 业务白名单） ====================
# 设计要点（对应二面整改「Prompt 不是安全边界」）：
#   1) Prompt 只负责「理解」，不承担安全职责；
#   2) LLM 输出结构化意图 {action, customer_id, field, value}；
#   3) 服务端做 Schema 校验 + 业务字段白名单校验；
#   4) 通过后由服务端模板生成参数化 SQL（用户值永不拼接）；
#   5) 高风险字段（金额/状态/归属）转人工审核，不自动落库。
STRUCTURED_ACTIONS = {"query_customer", "count_customer", "update_customer"}

# 允许参与查询/筛选的字段白名单（服务端硬编码）
STRUCTURED_QUERY_FIELDS = {
    "customer_id", "name", "phone", "school_grade", "intent_course",
    "matched_course", "active_status", "advisor_id", "source", "identity",
}


def validate_structured_intent(intent: dict) -> Optional[str]:
    """
    校验 LLM 输出的结构化意图（服务端信任边界）
    ==========================================
    返回：
      None — 校验通过
      str  — 失败原因
    """
    if not isinstance(intent, dict):
        return "[FAIL] 结构化意图必须是 JSON 对象"

    action = str(intent.get("action", "")).strip()
    if action not in STRUCTURED_ACTIONS:
        return f"[FAIL] 未授权的操作类型: {action or '(空)'}"

    # -------- 更新类：字段必须落在可写白名单内 --------
    if action == "update_customer":
        if not intent.get("customer_id"):
            return "[FAIL] 更新操作必须提供 customer_id"
        field = str(intent.get("field", "")).strip()
        writable = AUTO_UPDATE_FIELDS | MANUAL_REVIEW_FIELDS
        if field not in writable:
            return f"[FAIL] 字段 '{field}' 不在可写白名单内"
        if intent.get("value") is None:
            return "[FAIL] 更新操作必须提供 value"

    # -------- 查询类：筛选字段必须在查询白名单内 --------
    if action in ("query_customer", "count_customer"):
        field = intent.get("field")
        if field is not None and str(field) not in STRUCTURED_QUERY_FIELDS:
            return f"[FAIL] 查询字段 '{field}' 不在白名单内"

    return None


# ==================== 2. SQL Agent 类 ====================
class SQLAgent:
    """
    SQL Agent 类
    =============
    功能：将自然语言问题转换为 SQL 查询，并执行返回结果
    
    使用方式：
      agent = SQLAgent(llm)    # 传入 QwenLLM 实例
      result = agent.query("查询所有客户")  # 自然语言查询
      
    安全措施（两层校验 + 执行边界）：
      [OK] AST 白名单：Statement/Table/Column/Function（_validate_sql_ast）
      [OK] 字符串黑名单：危险关键字/注释/多语句/系统表（_validate_sql_safe）
      [OK] Resource Limit：LIMIT 强制收敛到 ≤ MAX_ROWS
      [OK] Read-only Boundary：只读连接 + PRAGMA query_only
      [OK] Timeout：busy_timeout=5000ms + progress_handler 超时中断
      [OK] Audit：允许/拒绝/异常全部落审计日志
    """
    
    def __init__(self, llm):
        """
        初始化 SQL Agent
        参数：
          llm : QwenLLM 实例，用于自然语言 → SQL 转换
        """
        from llms import QwenLLM  # 类型提示用
        self.llm = llm
        
        # 系统提示词（告诉 LLM 如何生成 SQL）
        # 包含表结构信息，让 LLM 知道有哪些表和字段
        self.system_prompt = self._build_system_prompt()
    
    def _build_system_prompt(self) -> str:
        """按当前 SQL 方言选择提示词（MySQL / SQLite）"""
        if SQL_DIALECT == "mysql":
            return self._build_system_prompt_mysql()
        return self._build_system_prompt_sqlite()

    def _build_system_prompt_sqlite(self) -> str:
        """
        构建系统提示词（SQLite 方言）
        ================
        包含数据库表结构，LLM 据此生成正确的 SQL
        注意：Prompt 只负责「让模型答得对」，不承担安全职责
              —— 安全由服务端 AST 白名单强制保证。
        """
        return """你是一个SQL查询助手。请根据用户的问题，生成SQLite SQL查询语句。

## 数据库表结构

### users（用户表）
- user_id TEXT PRIMARY KEY — 用户ID
- name TEXT — 姓名
- phone TEXT — 手机号
- created_at TIMESTAMP — 创建时间

### messages（消息表）
- message_id TEXT PRIMARY KEY — 消息ID
- user_id TEXT — 用户ID
- role TEXT — 角色(user/assistant)
- content TEXT — 消息内容
- created_at TIMESTAMP — 发送时间

### customer_profiles（客户画像表 · K12 教育 CRM）
- customer_id TEXT PRIMARY KEY — 客户ID
- name TEXT — 客户姓名
- phone TEXT — 电话
- email TEXT — 邮箱
- age INTEGER — 年龄（学生客户）
- identity TEXT — 身份（学生/家长/成人）
- school_grade TEXT — 就读年级（如 初二/高三）
- intent_course TEXT — 意向课程
- matched_course TEXT — 适配课程（需顾问确认）
- intent_price TEXT — 意向价格/预算（高风险字段）
- hourly_fee TEXT — 课时费（高风险字段）
- active_status TEXT — 活跃状态（活跃/待跟进/沉睡/流失风险）
- advisor_id TEXT — 归属课程顾问ID
- source TEXT — 客户来源渠道
- last_follow_up TIMESTAMP — 最近一次跟进时间
- company TEXT — 公司（通用场景兼容）
- position TEXT — 职位（通用场景兼容）
- notes TEXT — 备注
- tags TEXT — 标签(JSON格式)
- created_at TIMESTAMP — 创建时间
- updated_at TIMESTAMP — 更新时间

## 规则
1. 你必须只返回SQL语句，不需要额外解释
2. 只使用SELECT查询（只读），且只能访问 users / messages / customer_profiles / accounts / sessions / login_logs
3. 用中文别名显示字段名
4. LIMIT 查询结果最多50条
5. SQL语句用 ```sql ... ``` 包裹"""

    def _build_system_prompt_mysql(self) -> str:
        """
        构建系统提示词（MySQL 8.0 方言）
        ================================
        与 SQLite 版的差异：
          · 表结构类型按 MySQL 描述（VARCHAR / TINYINT / DATETIME / MEDIUMTEXT）
          · 日期函数用 DATE_FORMAT / NOW() / DATE_SUB，而非 strftime
          · 显式提醒不要访问 information_schema / mysql 等系统库
        安全仍由 sql_guard 的 MySQL 函数白名单 + 只读账号兜底。
        """
        return """你是一个SQL查询助手。请根据用户的问题，生成 MySQL 8.0 的 SELECT 查询语句。

## 数据库表结构

### users（用户表）
- user_id VARCHAR(64) PRIMARY KEY — 用户ID
- name VARCHAR(128) — 姓名
- phone VARCHAR(32) — 手机号
- created_at DATETIME — 创建时间

### messages（消息表）
- message_id VARCHAR(64) PRIMARY KEY — 消息ID
- session_id VARCHAR(64) — 会话ID
- user_id VARCHAR(64) — 用户ID
- role VARCHAR(16) — 角色(user/assistant)
- content MEDIUMTEXT — 消息内容
- created_at DATETIME — 发送时间

### customer_profiles（客户画像表 · K12 教育 CRM）
- customer_id VARCHAR(64) PRIMARY KEY — 客户ID
- name VARCHAR(128) — 客户姓名
- phone VARCHAR(32) — 电话
- email VARCHAR(255) — 邮箱
- age INT — 年龄（学生客户）
- identity VARCHAR(32) — 身份（学生/家长/成人）
- school_grade VARCHAR(64) — 就读年级（如 初二/高三）
- intent_course VARCHAR(255) — 意向课程
- matched_course VARCHAR(255) — 适配课程
- intent_price VARCHAR(64) — 意向价格/预算
- hourly_fee VARCHAR(64) — 课时费
- active_status VARCHAR(32) — 活跃状态
- advisor_id VARCHAR(64) — 归属课程顾问ID
- source VARCHAR(64) — 客户来源渠道
- last_follow_up DATETIME — 最近一次跟进时间
- company VARCHAR(255) — 公司（通用场景兼容）
- position VARCHAR(128) — 职位（通用场景兼容）
- notes MEDIUMTEXT — 备注
- tags TEXT — 标签（JSON 文本，可用 JSON_EXTRACT/JSON_UNQUOTE）
- created_at DATETIME — 创建时间
- updated_at DATETIME — 更新时间

## 规则
1. 你必须只返回SQL语句，不需要额外解释
2. 只使用 SELECT 查询（只读），且只能访问 users / messages / customer_profiles / accounts / sessions / login_logs
3. 用中文别名显示字段名（如 AS 客户数）
4. LIMIT 查询结果最多50条
5. 日期格式化用 DATE_FORMAT(字段, '%Y-%m')，当前时间用 NOW()，时间差用 DATEDIFF/TIMESTAMPDIFF
6. 禁止访问 information_schema / mysql / performance_schema / sys 等系统库
7. SQL语句用 ```sql ... ``` 包裹"""
    
    # ==================== 3. 自然语言查询 ====================
    def query(self, natural_language: str) -> str:
        """
        自然语言查询数据库
        ====================
        参数：
          natural_language : 自然语言问题，如 "查询所有客户"
        返回：
           格式化的查询结果或错误信息
        """
        # -------- 3.1 自然语言 → SQL --------
        sql = self._nl_to_sql(natural_language)
        if not sql:
            return "[FAIL] 无法生成SQL查询"
        
        logger.info(f"[SEARCH] SQL Agent: '{natural_language}' → {sql}")
        
        # -------- 3.2 执行 SQL 并格式化结果 --------
        result = self._execute_sql(sql)
        return result
    
    def _nl_to_sql(self, natural_language: str) -> Optional[str]:
        """
        调用 LLM 将自然语言转换为 SQL
        ===============================
        参数：
          natural_language : 自然语言问题
        返回：
          SQL 语句字符串 / None（转换失败）
        """
        # 构造消息列表
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": natural_language},
        ]
        
        # 调用 LLM 获取响应
        response = self.llm.invoke(messages)
        content = response.get("content", "")
        
        # 使用正则提取 SQL 语句（从 ```sql ... ``` 中提取）
        sql_match = re.search(
            r"```sql\s*(.*?)\s*```",
            content,
            re.DOTALL  # 跨行匹配
        )
        
        if sql_match:
            sql = sql_match.group(1).strip()
            return sql
        
        # 如果没有 ```sql ``` 包裹，尝试直接使用内容
        if "SELECT" in content.upper() or "select" in content.lower():
            return content.strip()
        
        return None
    
    # ==================== 4. SQL 执行器（AST 白名单 + 只读边界） ====================
    def _execute_sql(self, sql: str) -> str:
        """
        安全执行 SQL 查询（AST 白名单 + 只读边界 + 超时 + 审计）
        ========================================================
        执行链（每一环都有独立的设计理由）：
          1. AST 白名单校验（Statement/Table/Column/Function）
             —— LLM 生成整条 SQL 时，参数化保护不了「结构」，必须解析成语法树校验
          2. 字符串黑名单校验（_validate_sql_safe）—— 历史实现，兼容性双保险
          3. Resource Limit：护栏已把 LIMIT 收敛到 ≤ MAX_ROWS
          4. Read-only Boundary：只读连接（mode=ro）+ PRAGMA query_only
             —— 即使 SQL 校验被绕过，写入也会被数据库层拒绝（不是靠文本判断）
          5. Timeout：busy_timeout + progress_handler 超时中断（防慢查询拖死进程）
          6. Audit：允许 / 拒绝 / 异常全部落审计日志（可追溯谁执行了什么）

        参数：
          sql : SQL 语句
        返回：
            格式化的结果字符串
        """
        # -------- 1. AST 白名单（真正的安全边界）--------
        verdict = _validate_sql_ast(sql)
        if not verdict.ok:
            logger.warning(f"[WARN] SQL AST 校验未通过: {verdict.error}")
            audit_sql("reject", sql, False, verdict.error, stage="ast")
            return f"[FAIL] SQL 安全校验未通过（AST）: {verdict.error}"
        safe_sql = verdict.sql        # 已含合规 LIMIT（补入或收敛）

        # -------- 2. 字符串黑名单（兼容性双保险）--------
        legacy_error = _validate_sql_safe(safe_sql)
        if legacy_error:
            logger.warning(f"[WARN] SQL 黑名单校验未通过: {legacy_error}")
            audit_sql("reject", safe_sql, False, legacy_error, stage="legacy")
            # 埋点：第二层（字符串黑名单）拦截
            incr("sql_check_total", verdict="reject", stage="legacy")
            return legacy_error

        db = get_database()
        conn = None
        ro_mode = ""
        started = time.monotonic()
        try:
            # -------- 3. 只读连接（文件层 + 连接层双重只读）--------
            try:
                conn, ro_mode = open_readonly_connection(
                    getattr(db, "path", None),
                    busy_timeout_ms=getattr(db, "busy_timeout_ms", 5000),
                )
            except Exception as ro_exc:
                # 只读 URI 不可用（例如未来切到非 SQLite 引擎）时回退普通连接，
                # 并记录原因。注意：回退不等于放弃防护 —— AST 白名单仍在生效。
                logger.warning(f"[WARN] 只读连接不可用，回退普通连接: {ro_exc}")
                conn = db.connect()
                ro_mode = "fallback-rw"

            cursor = conn.cursor()
            cursor.execute(safe_sql)      # 单条 SELECT；LIMIT 已收敛
            rows = cursor.fetchall()
            column_names = [desc[0] for desc in cursor.description]
            cursor.close()

            elapsed_ms = int((time.monotonic() - started) * 1000)

            # 埋点：SQL 执行耗时 + 返回行数（资源管控效果的观测数据）
            observe("sql_exec_latency_ms", elapsed_ms)
            observe("sql_exec_rows", len(rows))

            # -------- 4. 审计：放行记录（含耗时 / 模式 / 表 / LIMIT）--------
            audit_sql(
                "allow", safe_sql, True, f"rows={len(rows)}",
                stage="ast", mode=ro_mode, elapsed_ms=elapsed_ms,
                tables=sorted(verdict.tables), limit=verdict.limit,
                limit_injected=verdict.limit_injected,
                limit_clamped=verdict.limit_clamped,
            )

            # -------- 5. 格式化结果 --------
            if not rows:
                return "📭 查询结果为空"

            result_lines = [f"[CHART] 查询结果（共 {len(rows)} 条）", "-" * 60]
            for i, row in enumerate(rows[:50], 1):   # 最多展示 50 条
                # 每行格式：序号 | 字段1=值1 | 字段2=值2
                row_str = " | ".join(
                    f"{col}={val}" for col, val in zip(column_names, row)
                )
                result_lines.append(f"{i}. {row_str}")
            if len(rows) > 50:
                result_lines.append(f"... 还有 {len(rows) - 50} 条未显示")
            return "\n".join(result_lines)

        except Exception as e:
            elapsed_ms = int((time.monotonic() - started) * 1000)
            logger.error(f"[FAIL] SQL 执行失败: {e}")
            audit_sql("error", safe_sql, False, str(e),
                      stage="execute", mode=ro_mode, elapsed_ms=elapsed_ms)
            # 埋点：执行阶段异常（只读边界 / 超时中断 / 引擎错误）
            incr("sql_exec_error_total", stage="execute")
            return f"[FAIL] SQL 执行失败: {str(e)}"
        finally:
            if conn is not None:
                try:
                    conn.set_progress_handler(None, 0)   # 复位超时回调
                except Exception:
                    pass
                try:
                    conn.close()
                except Exception:
                    pass
    
    # ==================== 5. 结构化意图执行（参数化 SQL） ====================
    def execute_intent(self, intent: dict) -> str:
        """
        执行结构化意图（服务端模板 + 参数化 SQL）
        ==========================================
        与 query() 的区别：
          query()          : LLM 自由生成 SQL → AST 白名单 + 黑名单双重校验
          execute_intent() : LLM 只产出结构化意图 → 白名单 → 服务端模板 + 参数化
        后者把安全边界从 Prompt 收回到服务端代码，越权写入被硬拦截。
        
        参数：
          intent : {"action": "...", "customer_id": "...", "field": "...", "value": "..."}
        返回：
            查询/更新结果文本
        """
        error = validate_structured_intent(intent)
        if error:
            logger.warning(f"[WARN] 结构化意图校验未通过: {error}")
            return error
        
        action = intent["action"]
        try:
            if action == "query_customer":
                return self._intent_query(intent, count_only=False)
            if action == "count_customer":
                return self._intent_query(intent, count_only=True)
            if action == "update_customer":
                return self._intent_update(intent)
        except Exception as e:
            logger.error(f"[FAIL] 结构化意图执行失败: {e}")
            return f"[FAIL] 结构化意图执行失败: {str(e)}"
        return f"[FAIL] 未实现的操作: {action}"
    
    @staticmethod
    def _intent_filters(intent: dict):
        """
        由白名单字段构造参数化 WHERE 条件
        ==================================
        字段名已通过白名单校验，可安全作为列名拼接；
        用户值一律走 ? 占位符，杜绝拼接注入。
        """
        conditions, params = [], []
        customer_id = intent.get("customer_id")
        if customer_id:
            conditions.append("customer_id = ?")
            params.append(str(customer_id))
        field = intent.get("field")
        value = intent.get("value")
        if field and value is not None:
            conditions.append(f"{field} = ?")
            params.append(value)
        return conditions, params
    
    def _intent_query(self, intent: dict, count_only: bool = False) -> str:
        """结构化查询（参数化执行 + 表格化输出）"""
        conditions, params = self._intent_filters(intent)
        where = f" WHERE {' AND '.join(conditions)}" if conditions else ""
        if count_only:
            sql = f"SELECT COUNT(*) AS cnt FROM customer_profiles{where}"
        else:
            sql = f"SELECT * FROM customer_profiles{where} LIMIT {MAX_ROWS}"
        
        db = get_database()
        conn = db.connect()
        cursor = conn.cursor()
        cursor.execute(sql, params)   # 参数化执行，杜绝 SQL 注入
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        cursor.close()
        conn.close()
        
        audit_sql(
            "allow", f"SELECT ... FROM customer_profiles{where}", True,
            f"rows={len(rows)}", stage="structured-intent",
            columns=sorted(conditions), count_only=count_only,
        )

        if count_only:
            total = rows[0][0] if rows else 0
            return f"[CHART] 命中客户数：{total}"
        if not rows:
            return "📭 查询结果为空"
        
        lines = [f"[CHART] 查询结果（共 {len(rows)} 条）", "-" * 60]
        for i, row in enumerate(rows[:50], 1):
            lines.append(
                f"{i}. " + " | ".join(f"{c}={v}" for c, v in zip(columns, row))
            )
        return "\n".join(lines)
    
    def _intent_update(self, intent: dict) -> str:
        """
        结构化更新（字段分流：低风险自动写 / 高风险转人工审核）
        """
        customer_id = str(intent.get("customer_id"))
        field = str(intent.get("field"))
        value = intent.get("value")
        
        # 高风险字段：不自动落库，转人工审核（幻觉 / 越权的最后一道闸门）
        if field in MANUAL_REVIEW_FIELDS:
            logger.warning(
                f"[WARN] 高风险字段 '{field}' 命中人工审核，未自动落库 "
                f"(customer_id={customer_id})"
            )
            audit_sql("review", f"UPDATE customer_profiles.{field}", False,
                      "高风险字段转人工审核", stage="structured-intent",
                      customer_id=customer_id)
            return (
                f"[REVIEW] 字段 '{field}' 属高风险字段（金额/状态/归属），"
                f"已写入待人工审核列表，未自动更新（customer_id={customer_id}）。"
            )
        
        if field not in AUTO_UPDATE_FIELDS:
            return f"[FAIL] 字段 '{field}' 不允许自动更新"
        
        from datetime import datetime
        db = get_database()
        affected = db.execute(
            f"UPDATE customer_profiles SET {field} = ?, updated_at = ? "
            f"WHERE customer_id = ?",
            (value, datetime.now().isoformat(timespec="seconds"), customer_id),
        )
        if affected:
            logger.info(f"[OK] 结构化意图更新客户 {customer_id}.{field}")
            audit_sql("allow", f"UPDATE customer_profiles.{field}", True,
                      f"affected={affected}", stage="structured-intent",
                      customer_id=customer_id)
            return f"[OK] 已更新客户 {customer_id} 的 {field}"
        audit_sql("error", f"UPDATE customer_profiles.{field}", False,
                  "未找到客户", stage="structured-intent", customer_id=customer_id)
        return f"[FAIL] 未找到客户 {customer_id}"
