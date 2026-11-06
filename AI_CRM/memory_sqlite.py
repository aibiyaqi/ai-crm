"""
memory_sqlite.py - SQLite 数据库管理模块
===========================================
功能分区：
  1. 数据库初始化和建表（账户表 / 用户表 / 消息表 / 客户画像表 / 登录日志表）
  2. 账户认证 CRUD（注册 / 查询 / 更新）
  3. 用户信息 CRUD（增/查/改）
  4. 对话历史 CRUD（增/查/删）
  5. 客户画像 CRUD（增/查/改/删 - 轻量化 CRM 核心）
  6. 登录日志与密码重置 CRUD
  7. 对话历史长度控制（限最近 N 轮，防 Token 溢出）
"""

# ==================== 1. 导入依赖 ====================
from typing import Optional, List  # 类型提示
from datetime import datetime      # 日期时间处理
import json                        # JSON 序列化（处理标签列表等）
import threading                   # 建表去重锁
import uuid                        # 告警去重键 / 唯一标识生成

from config import (
    MAX_HISTORY_ROUNDS,               # 最大对话历史轮次：5
    MAX_HISTORY_TOKENS,               # 历史最大 Token 数（按 Token 动态裁剪）
    MAX_SINGLE_MESSAGE_CHARS,         # 单条消息最大字符数（入口拦截超长）
    MIN_HISTORY_ROUNDS_ON_OVERFLOW,   # 裁剪兜底：极端场景只保留最近 N 轮
    REDIS_SESSION_TTL,                # 会话令牌 Redis TTL
    REDIS_ALERT_DEDUP_TTL,            # 告警去重 Redis TTL
)
from db import get_database  # DAO 抽象层：统一连接/PRAGMA/引擎切换位
from redis_client import get_redis  # Redis 基础设施：缓存/锁/延迟队列（自动降级）
from models import (  # Pydantic 数据模型
    User, Message, CustomerProfile, Account, PasswordResetToken, LoginLog,
    FollowUp, FOLLOW_UP_CONVERTED_RESULTS,
)
from logger import logger  # 日志记录器


# 进程内建表去重：同一引擎+库只执行一次 DDL（MySQL 下每请求建表代价高）
_SCHEMA_READY: set = set()
_SCHEMA_LOCK = threading.Lock()


# ==================== 2. 数据库管理类 ====================
class SQLiteMemory:
    """
    SQLite 数据库管理类
    ====================
    职责：
      1. 自动建表（第一次使用时初始化）
      2. 账户认证管理（注册、登录、密码重置）
      3. 用户信息管理（注册、查询、更新）
      4. 对话历史管理（保存、加载、清理）
      5. 客户画像管理（增删改查 = CRM 核心功能）
      6. 登录日志审计
    
    使用方式：
      db = SQLiteMemory()
      db.save_account(account)
      account = db.get_account_by_username("admin")
    """
    
    def __init__(self):
        """初始化数据库连接，自动建表"""
        # 通过 DAO 抽象层获取数据库实例（当前 DB_ENGINE=sqlite；迁移只改配置）
        self._db = get_database()
        # 由 DAO 层创建并配置连接（WAL / busy_timeout / foreign_keys / synchronous）
        self.conn = self._db.connect()
        # 创建数据库游标（用于执行 SQL）
        self.cursor = self.conn.cursor()

        # -------- 连接优化：PRAGMA 已统一在 DAO 层设置 --------
        self._optimize_connection()

        # 初始化建表（进程内只做一次：MySQL 下每请求 12 条 DDL 代价太高）
        key = self._schema_key()
        if key not in _SCHEMA_READY:
            with _SCHEMA_LOCK:
                if key not in _SCHEMA_READY:
                    self._create_tables()
                    _SCHEMA_READY.add(key)
        logger.info(f"[OK] 数据库已连接: engine={self._db.engine}")

    def _schema_key(self) -> str:
        """建表缓存的键：引擎 + 库标识（SQLite 用路径，MySQL 用库名）"""
        target = getattr(self._db, "path", None) or getattr(self._db, "database", "")
        return f"{self._db.engine}:{target}"

    def _optimize_connection(self):
        """
        连接优化配置
        ==============
        WAL / busy_timeout / foreign_keys / synchronous 等 PRAGMA 已统一在
        DAO 层 db.SQLiteDatabase.connect() 中设置（有代码证据），此处仅兜底
        重跑一次，防止 DAO 层未生效时遗漏。
        MySQL 无需 PRAGMA（连接参数/时区由连接池 init_command 接管），直接跳过。
        """
        if self._db.engine != "sqlite":
            logger.debug(f"[DB] 引擎 {self._db.engine} 跳过 SQLite PRAGMA 设置")
            return
        pragmas = [
            ("PRAGMA journal_mode=WAL;", "WAL 模式"),
            ("PRAGMA busy_timeout=5000;", "busy_timeout=5000ms"),
            ("PRAGMA foreign_keys=ON;", "外键约束"),
            ("PRAGMA synchronous=NORMAL;", "synchronous=NORMAL"),
        ]
        for pragma, name in pragmas:
            try:
                self.cursor.execute(pragma)
                logger.debug(f"[OK] SQLite PRAGMA 已设置: {name}")
            except Exception as e:
                logger.warning(f"[WARN] SQLite PRAGMA 设置失败 ({name}): {e}")
    
    # ==================== 3. 建表 ====================
    def _create_tables(self):
        """
        创建数据表（如果不存在）
        ========================
        表结构：
          1. accounts：账户认证表（含角色/密码哈希）
          2. password_reset_tokens：密码重置令牌表
          3. login_logs：登录日志表
          4. session_tokens：会话令牌表（跨刷新登录保持）
          5. sessions：对话会话表（支持多对话框）
          6. users：用户信息表
          7. messages：对话历史表（含 session_id）
          8. customer_profiles：客户画像表（CRM 核心）
        
        设计原则：
          [OK] IF NOT EXISTS：幂等操作，重复调用不会重复建表
          [OK] 主键 UNIQUE：保证每条记录可唯一标识
          [OK] 默认时间戳：自动记录创建/更新时间
        """
        # -------- MySQL：走独立方言建表脚本（类型/索引/字符集与 SQLite 不同）--------
        if self._db.engine == "mysql":
            self._create_tables_mysql()
            return
        # -------- 3.1 账户认证表 --------
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS accounts (
                account_id    TEXT PRIMARY KEY,        -- 账户唯一标识（UUID）
                username      TEXT NOT NULL UNIQUE,    -- 用户名（登录用，唯一）
                password_hash TEXT NOT NULL,           -- 密码哈希值
                role          TEXT NOT NULL DEFAULT 'user',  -- 角色：admin/user
                phone         TEXT NOT NULL UNIQUE,    -- 手机号（唯一，可用于登录）
                email         TEXT,                    -- 邮箱（可选）
                is_active     INTEGER NOT NULL DEFAULT 1,  -- 账户是否激活（1=激活，0=禁用）
                last_login    TIMESTAMP,               -- 最后登录时间
                created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,  -- 创建时间
                updated_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP   -- 更新时间
            )
        """)
        
        # -------- 3.2 密码重置令牌表 --------
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS password_reset_tokens (
                token_id    TEXT PRIMARY KEY,          -- 令牌唯一标识（UUID）
                account_id  TEXT NOT NULL,             -- 关联账户 ID
                token       TEXT NOT NULL,             -- 重置令牌
                code        TEXT NOT NULL,             -- 短信验证码（6位数字）
                expires_at  TIMESTAMP NOT NULL,        -- 过期时间
                used        INTEGER NOT NULL DEFAULT 0,-- 是否已使用（0=未使用，1=已使用）
                created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,  -- 创建时间
                FOREIGN KEY (account_id) REFERENCES accounts(account_id)
            )
        """)
        
        # -------- 3.3 登录日志表 --------
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS login_logs (
                log_id      TEXT PRIMARY KEY,          -- 日志唯一标识（UUID）
                account_id  TEXT,                      -- 关联账户 ID（失败时可空）
                username    TEXT NOT NULL,             -- 尝试登录的用户名
                action      TEXT NOT NULL,             -- 操作类型：login/logout/fail
                ip_address  TEXT,                      -- 登录 IP
                detail      TEXT,                      -- 详细信息（失败原因等）
                created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,  -- 操作时间
                FOREIGN KEY (account_id) REFERENCES accounts(account_id)
            )
        """)
        
        # -------- 3.4 会话令牌表（跨刷新登录保持） --------
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS session_tokens (
                token       TEXT PRIMARY KEY,          -- 会话令牌（UUID，存 URL query_params）
                account_id  TEXT NOT NULL,             -- 关联账户 ID
                created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,  -- 创建时间
                expires_at  TIMESTAMP,                 -- 过期时间（默认 7 天）
                FOREIGN KEY (account_id) REFERENCES accounts(account_id)
            )
        """)
        
        # -------- 3.5 对话会话表（支持多对话框） --------
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id  TEXT PRIMARY KEY,          -- 会话唯一标识（UUID）
                account_id  TEXT NOT NULL,             -- 所属账户 ID
                title       TEXT NOT NULL DEFAULT '新对话', -- 会话标题
                created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,  -- 创建时间
                updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,  -- 最后活跃时间
                FOREIGN KEY (account_id) REFERENCES accounts(account_id)
            )
        """)
        
        # -------- 3.6 用户信息表 --------
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id    TEXT PRIMARY KEY,        -- 用户唯一标识（UUID）
                name       TEXT NOT NULL,           -- 用户姓名
                phone      TEXT,                    -- 手机号
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,  -- 创建时间
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP   -- 更新时间
            )
        """)
        
        # -------- 3.7 对话历史表（多会话支持） --------
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                message_id TEXT PRIMARY KEY,        -- 消息唯一标识（UUID）
                session_id TEXT NOT NULL,           -- 所属会话 ID
                user_id    TEXT NOT NULL,           -- 发送用户 ID
                role       TEXT NOT NULL,           -- 角色：user/assistant/system
                content    TEXT NOT NULL,           -- 消息内容
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,  -- 发送时间
                FOREIGN KEY (session_id) REFERENCES sessions(session_id),
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            )
        """)
        
        # -------- 3.8 客户画像表（CRM 核心功能 · K12 教育版） --------
        # 字段分组见 models.CustomerProfile：基础身份 / 业务意向 / 经营状态 / 通用兼容
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS customer_profiles (
                customer_id    TEXT PRIMARY KEY,       -- 客户唯一标识（UUID）
                name           TEXT NOT NULL,           -- 客户姓名
                company        TEXT,                    -- 公司名称（通用企业场景兼容）
                position       TEXT,                    -- 职位（通用企业场景兼容）
                phone          TEXT,                    -- 电话
                email          TEXT,                    -- 邮箱
                notes          TEXT,                    -- 备注
                tags           TEXT DEFAULT '[]',       -- 标签列表（JSON 格式存储）
                created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                -- ===== K12 教育业务字段（二面画像补强）=====
                age            INTEGER,                 -- 年龄（学生客户）
                identity       TEXT,                    -- 身份：学生/家长/成人
                school_grade   TEXT,                    -- 年级：初二 / 高三 / 在职
                intent_course  TEXT,                    -- 意向课程
                matched_course TEXT,                    -- 适配课程（需顾问确认）
                intent_price   TEXT,                    -- 意向价格/预算（高风险字段）
                hourly_fee     TEXT,                    -- 课时费（高风险字段）
                active_status  TEXT,                    -- 活跃状态：活跃/待跟进/沉睡/流失风险
                advisor_id     TEXT,                    -- 归属课程顾问 ID
                source         TEXT,                    -- 客户来源渠道
                last_follow_up TIMESTAMP                -- 最近一次跟进时间
            )
        """)

        # -------- 3.9 跟进记录表（B.4 跟进闭环 · 产品化落地） --------
        # 把「跟进动作」结构化成独立记录：谁 / 何时 / 什么方式 / 什么结果 / 下次何时跟。
        # 有了这张表，跟进记录、跟进提醒（到期巡检）、跟进转化统计才都能算出来。
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS follow_ups (
                follow_up_id      TEXT PRIMARY KEY,     -- 跟进记录唯一标识（UUID）
                customer_id       TEXT NOT NULL,        -- 关联客户 ID
                advisor_id        TEXT,                 -- 跟进人（课程顾问 ID）
                channel           TEXT,                 -- 跟进方式：电话/微信/面谈/短信/邮件
                content           TEXT,                 -- 跟进内容摘要
                result            TEXT,                 -- 跟进结果：已联系/未接/有意向/已成交/已放弃
                next_follow_up_at TIMESTAMP,            -- 计划下次跟进时间（到期提醒依据）
                created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP  -- 本次跟进时间
            )
        """)
        self.cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_follow_ups_customer ON follow_ups(customer_id)"
        )
        self.cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_follow_ups_next ON follow_ups(next_follow_up_at)"
        )

        # 主动告警表（B.1 ⑤ 兜底「常驻定时调度 + 主动告警推送」的落地载体）
        # 背景：异常筛查工具此前只能「用户问了才跑」，属于被动查询；
        #       本表 + scheduler.py 让系统「自己定时跑、把结果推给顾问」，
        #       即把「筛查」从被动查询升级为主动告警。
        # 去重：同一客户 + 同一告警类型 + 同一业务日只落一条（dedup_key 唯一），
        #       避免定时任务每轮重复刷屏。
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                alert_id     TEXT PRIMARY KEY,          -- 告警唯一标识（UUID）
                dedup_key    TEXT UNIQUE,               -- 去重键：客户+类型+业务日
                customer_id  TEXT,                      -- 关联客户 ID
                customer_name TEXT,                     -- 客户姓名（冗余，便于直接展示）
                advisor_id   TEXT,                      -- 归属顾问（推送对象）
                alert_type   TEXT,                      -- 告警类型：高意向未跟进/流失沉睡风险/资料不全/计划到期/超期未跟进
                severity     TEXT,                      -- 严重度：high/medium/low
                reasons      TEXT,                      -- 命中原因（JSON 数组文本）
                source       TEXT,                      -- 来源：scheduler（定时巡检）/ manual（手动）
                status       TEXT DEFAULT 'unread',     -- 状态：unread/read
                created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,  -- 产生时间
                read_at      TIMESTAMP                  -- 已读时间
            )
        """)
        self.cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_alerts_status ON alerts(status)"
        )
        self.cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_alerts_advisor ON alerts(advisor_id)"
        )
        self.cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_alerts_created ON alerts(created_at)"
        )

        # -------- 3.11 今日待办池（分层跑批 · 粗筛/事件驱动产出，精筛输入） --------
        # 背景：需求要求「事件驱动 + 分层跑批」——
        #   ① 事件驱动：录入客户即投递延迟消息，到期命中规则 → 塞入本池；
        #   ② 21:00 粗筛：全量扫表，把可能异常客户提前塞入本池（兜底）；
        #   ③ 次日13:00 精筛：只扫本池（增量），命中写 alerts（去重）。
        # 本表是「粗筛/事件」与「精筛」之间的缓冲，避免精筛全量扫表。
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS pending_todo (
                todo_id      TEXT PRIMARY KEY,          -- 待办唯一标识（UUID）
                customer_id  TEXT NOT NULL,             -- 关联客户 ID
                advisor_id   TEXT,                      -- 归属顾问
                source       TEXT,                      -- 来源：event(事件驱动) / coarse(21点粗筛)
                trigger_at   TIMESTAMP,                 -- 事件驱动：延迟消息触发时间
                biz_date     TEXT,                      -- 业务日 YYYY-MM-DD（精筛按此取当日待办）
                status       TEXT DEFAULT 'pending',    -- 状态：pending/done/expired
                created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_todo_bizdate ON pending_todo(biz_date, status)"
        )
        self.cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_todo_customer ON pending_todo(customer_id)"
        )

        # -------- 3.12 延迟消息表（事件驱动 · 轻量落地，生产可换 Redis ZSET / MQ） --------
        # 背景：销售录入信息 → 投递延迟消息（10 天后触发检查）；
        #       中途客户转化或已跟进 → 撤销延迟消息（status=cancelled），做到零无效扫描。
        # 说明：此处用「表 + 轮询」做轻量落地，接口与 MQ 一致；
        #       生产可用 Redis ZSET（ZADD score=fire_at / ZREM 撤销 / ZRANGEBYSCORE 消费）。
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS delayed_messages (
                msg_id       TEXT PRIMARY KEY,          -- 消息唯一标识（UUID）
                customer_id  TEXT NOT NULL,             -- 关联客户 ID
                event_type   TEXT,                      -- 事件类型：follow_up_check
                fire_at      TIMESTAMP,                 -- 触发时间（录入时间 + delay_days）
                status       TEXT DEFAULT 'scheduled',  -- 状态：scheduled/cancelled/fired
                created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_delay_fire ON delayed_messages(fire_at, status)"
        )
        self.cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_delay_customer ON delayed_messages(customer_id, status)"
        )

        # 提交事务，使建表生效
        self.conn.commit()

        # 幂等迁移：老库（company/position 版）补齐 K12 新列
        # 说明：CREATE TABLE IF NOT EXISTS 不会修改已存在表的列，必须 ALTER TABLE
        self._migrate_customer_columns()

    def _migrate_customer_columns(self):
        """
        客户画像表列迁移（幂等 · 只加列不改列）
        ========================================
        背景：CREATE TABLE IF NOT EXISTS 对已存在的表不会新增列，
              老库需要 ALTER TABLE 补齐 K12 字段。
        做法：PRAGMA table_info 读现有列 → 缺哪补哪 → 提交。
        安全：可重复执行；仅 ADD COLUMN，不影响历史数据。
        MySQL：改用 information_schema.columns 读列，类型用方言类型（见下表）。
        """
        try:
            if self._db.engine == "mysql":
                self.cursor.execute(
                    "SELECT column_name AS col FROM information_schema.columns "
                    "WHERE table_schema = DATABASE() AND table_name = 'customer_profiles'"
                )
                existing = {row["col"] for row in self.cursor.fetchall()}
            else:
                self.cursor.execute("PRAGMA table_info(customer_profiles)")
                existing = {row[1] for row in self.cursor.fetchall()}
            raw_columns = [
                ("age", "INTEGER"),
                ("identity", "TEXT"),
                ("school_grade", "TEXT"),
                ("intent_course", "TEXT"),
                ("matched_course", "TEXT"),
                ("intent_price", "TEXT"),
                ("hourly_fee", "TEXT"),
                ("active_status", "TEXT"),
                ("advisor_id", "TEXT"),
                ("source", "TEXT"),
                ("last_follow_up", "TIMESTAMP"),
                # ===== 45天退场与公海机制（六、SQL逻辑过滤）=====
                ("pool_type", "TEXT DEFAULT '私海'"),   # 池子类型：私海 / 公海
                ("is_sleeping", "INTEGER DEFAULT 0"),   # 沉睡/流失标记：1=沉睡/流失
                ("exit_schedule_at", "TIMESTAMP"),      # 退出调度时间（掉入公海时写入）
            ]
            # MySQL 方言类型映射（长度受控，避免 TEXT 不能带 DEFAULT 的问题）
            mysql_types = {
                "age": "INT",
                "identity": "VARCHAR(32)",
                "school_grade": "VARCHAR(64)",
                "intent_course": "VARCHAR(255)",
                "matched_course": "VARCHAR(255)",
                "intent_price": "VARCHAR(64)",
                "hourly_fee": "VARCHAR(64)",
                "active_status": "VARCHAR(32)",
                "advisor_id": "VARCHAR(64)",
                "source": "VARCHAR(64)",
                "last_follow_up": "DATETIME(6)",
                "pool_type": "VARCHAR(16) DEFAULT '私海'",
                "is_sleeping": "TINYINT(1) DEFAULT 0",
                "exit_schedule_at": "DATETIME(6)",
            }
            k12_columns = (
                [(c, mysql_types.get(c, t)) for c, t in raw_columns]
                if self._db.engine == "mysql"
                else raw_columns
            )
            added = []
            for col, coltype in k12_columns:
                if col not in existing:
                    self.cursor.execute(
                        f"ALTER TABLE customer_profiles ADD COLUMN {col} {coltype}"
                    )
                    added.append(col)
            if added:
                self.conn.commit()
                logger.info(f"[MIGRATE] customer_profiles 已补齐列: {', '.join(added)}")
        except Exception as e:
            logger.error(f"[FAIL] 客户画像列迁移失败: {e}")
    
    def _create_tables_mysql(self):
        """
        MySQL 建表（方言实现）
        ======================
        与 SQLite 的差异全部收敛在 sql/mysql/001_init.sql：
          · TEXT 主键/唯一键 → VARCHAR（InnoDB 索引长度）
          · INTEGER 布尔 → TINYINT(1)
          · TIMESTAMP → DATETIME（无时区换算）
          · 索引内联为 KEY（MySQL 不支持 CREATE INDEX IF NOT EXISTS）
        幂等：全部 CREATE TABLE IF NOT EXISTS，可重复执行。
        """
        from db import load_sql_resource

        script = load_sql_resource("sql/mysql/001_init.sql")
        self._db.executescript(script)
        logger.info("[DB] MySQL 业务表已就绪（sql/mysql/001_init.sql）")
        # 老库补齐列（幂等，仅 ADD COLUMN）
        self._migrate_customer_columns()

    # ==================== 4. 账户认证操作 ====================
    def save_account(self, account: Account) -> bool:
        """
        保存或更新账户信息
        ====================
        参数：
          account : Account 对象
        返回：
          True=成功 / False=失败
        """
        try:
            # MySQL 的 ON DUPLICATE KEY UPDATE 会命中「任意唯一键」（username / phone），
            # 而 SQLite 的 ON CONFLICT(account_id) 只处理主键冲突。
            # 若用户名或手机号已被另一个 account_id 占用，必须显式失败，
            # 否则 MySQL 会静默更新到别人的账户行、返回 True 但内存里的 account_id 是错的。
            self.cursor.execute(
                "SELECT account_id FROM accounts "
                "WHERE (username = ? OR phone = ?) AND account_id <> ? LIMIT 1",
                (account.username, account.phone, account.account_id),
            )
            if self.cursor.fetchone():
                logger.error(
                    f"[FAIL] 保存账户失败：用户名或手机号已被占用 "
                    f"(username={account.username}, phone={account.phone})"
                )
                return False

            self.cursor.execute("""
                INSERT INTO accounts (account_id, username, password_hash, role, phone, email, is_active, last_login, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(account_id) DO UPDATE SET
                    username = excluded.username,
                    password_hash = excluded.password_hash,
                    role = excluded.role,
                    phone = excluded.phone,
                    email = excluded.email,
                    is_active = excluded.is_active,
                    last_login = excluded.last_login,
                    updated_at = CURRENT_TIMESTAMP
            """, (
                account.account_id,
                account.username,
                account.password_hash,
                account.role,
                account.phone,
                account.email,
                1 if account.is_active else 0,
                account.last_login,
                account.created_at,
                account.updated_at,
            ))
            self.conn.commit()
            logger.info(f"[LOCK] 账户已保存: {account.username} ({account.account_id})")
            return True
        except Exception as e:
            logger.error(f"[FAIL] 保存账户失败: {e}")
            return False
    
    def get_account_by_username(self, username: str) -> Optional[Account]:
        """
        根据用户名查询账户
        ====================
        参数：
          username : 用户名
        返回：
          Account 对象 / None
        """
        self.cursor.execute(
            "SELECT * FROM accounts WHERE username = ?",
            (username,)
        )
        return self._row_to_account(self.cursor.fetchone())
    
    def get_account_by_phone(self, phone: str) -> Optional[Account]:
        """
        根据手机号查询账户
        ====================
        参数：
          phone : 手机号
        返回：
          Account 对象 / None
        """
        self.cursor.execute(
            "SELECT * FROM accounts WHERE phone = ?",
            (phone,)
        )
        return self._row_to_account(self.cursor.fetchone())
    
    def get_account_by_id(self, account_id: str) -> Optional[Account]:
        """
        根据 ID 查询账户
        ==================
        参数：
          account_id : 账户唯一标识
        返回：
          Account 对象 / None
        """
        self.cursor.execute(
            "SELECT * FROM accounts WHERE account_id = ?",
            (account_id,)
        )
        return self._row_to_account(self.cursor.fetchone())
    
    def list_all_accounts(self) -> List[Account]:
        """
        列出所有账户
        ==============
        返回：
          Account 对象列表
        """
        self.cursor.execute(
            "SELECT * FROM accounts ORDER BY created_at DESC"
        )
        rows = self.cursor.fetchall()
        return [self._row_to_account(row) for row in rows if row]
    
    def update_account_password(self, account_id: str, new_password_hash: str) -> bool:
        """
        更新账户密码
        ==============
        参数：
          account_id       : 账户 ID
          new_password_hash : 新密码哈希值
        返回：
          True=成功 / False=失败
        """
        try:
            self.cursor.execute("""
                UPDATE accounts
                SET password_hash = ?, updated_at = CURRENT_TIMESTAMP
                WHERE account_id = ?
            """, (new_password_hash, account_id))
            self.conn.commit()
            logger.info(f"[KEY] 账户密码已更新: {account_id}")
            return True
        except Exception as e:
            logger.error(f"[FAIL] 更新密码失败: {e}")
            return False
    
    def update_account_status(self, account_id: str, is_active: bool) -> bool:
        """
        更新账户激活状态
        ==================
        参数：
          account_id : 账户 ID
          is_active  : 是否激活
        返回：
          True=成功 / False=失败
        """
        try:
            self.cursor.execute("""
                UPDATE accounts
                SET is_active = ?, updated_at = CURRENT_TIMESTAMP
                WHERE account_id = ?
            """, (1 if is_active else 0, account_id))
            self.conn.commit()
            logger.info(f"[LOCK] 账户状态已更新: {account_id} -> {'激活' if is_active else '禁用'}")
            return True
        except Exception as e:
            logger.error(f"[FAIL] 更新账户状态失败: {e}")
            return False
    
    def update_last_login(self, account_id: str) -> bool:
        """
        更新最后登录时间
        ==================
        参数：
          account_id : 账户 ID
        返回：
          True=成功 / False=失败
        """
        try:
            self.cursor.execute("""
                UPDATE accounts
                SET last_login = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP
                WHERE account_id = ?
            """, (account_id,))
            self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"[FAIL] 更新登录时间失败: {e}")
            return False
    
    def _row_to_account(self, row) -> Optional[Account]:
        """将数据库行转换为 Account 对象"""
        if row is None:
            return None
        return Account(
            account_id=row[0],
            username=row[1],
            password_hash=row[2],
            role=row[3],
            phone=row[4],
            email=row[5],
            is_active=bool(row[6]),
            last_login=row[7],
            created_at=row[8],
            updated_at=row[9],
        )
    
    # ==================== 5. 密码重置令牌操作 ====================
    def save_reset_token(self, reset_token: PasswordResetToken) -> bool:
        """
        保存密码重置令牌
        ==================
        参数：
          reset_token : PasswordResetToken 对象
        返回：
          True=成功 / False=失败
        """
        try:
            self.cursor.execute("""
                INSERT INTO password_reset_tokens (token_id, account_id, token, code, expires_at, used, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                reset_token.token_id,
                reset_token.account_id,
                reset_token.token,
                reset_token.code,
                reset_token.expires_at,
                1 if reset_token.used else 0,
                reset_token.created_at,
            ))
            self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"[FAIL] 保存重置令牌失败: {e}")
            return False
    
    def get_valid_reset_token(self, account_id: str, code: str) -> Optional[PasswordResetToken]:
        """
        获取有效（未使用且未过期）的重置令牌
        ======================================
        参数：
          account_id : 账户 ID
          code       : 验证码
        返回：
          PasswordResetToken 对象 / None
        """
        now = datetime.now()
        self.cursor.execute("""
            SELECT * FROM password_reset_tokens
            WHERE account_id = ? AND code = ? AND used = 0 AND expires_at > ?
            ORDER BY created_at DESC
            LIMIT 1
        """, (account_id, code, now))
        row = self.cursor.fetchone()
        if row is None:
            return None
        return PasswordResetToken(
            token_id=row[0],
            account_id=row[1],
            token=row[2],
            code=row[3],
            expires_at=row[4],
            used=bool(row[5]),
            created_at=row[6],
        )
    
    def mark_token_used(self, token_id: str) -> bool:
        """
        标记重置令牌为已使用
        ======================
        参数：
          token_id : 令牌 ID
        返回：
          True=成功 / False=失败
        """
        try:
            self.cursor.execute(
                "UPDATE password_reset_tokens SET used = 1 WHERE token_id = ?",
                (token_id,)
            )
            self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"[FAIL] 标记令牌已使用失败: {e}")
            return False
    
    # ==================== 6. 登录日志操作 ====================
    def save_login_log(self, log: LoginLog) -> bool:
        """
        保存登录日志
        ==============
        参数：
          log : LoginLog 对象
        返回：
          True=成功 / False=失败
        """
        try:
            self.cursor.execute("""
                INSERT INTO login_logs (log_id, account_id, username, action, ip_address, detail, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                log.log_id,
                log.account_id,
                log.username,
                log.action,
                log.ip_address,
                log.detail,
                log.created_at,
            ))
            self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"[FAIL] 保存登录日志失败: {e}")
            return False
    
    def count_recent_fails(self, username: str, minutes: int = 15) -> int:
        """
        统计最近 N 分钟内的登录失败次数
        ================================
        参数：
          username : 用户名
          minutes  : 时间范围（分钟）
        返回：
          失败次数
        """
        from datetime import timedelta
        since = datetime.now() - timedelta(minutes=minutes)
        self.cursor.execute("""
            SELECT COUNT(*) FROM login_logs
            WHERE username = ? AND action = 'fail' AND created_at > ?
        """, (username, since))
        row = self.cursor.fetchone()
        return row[0] if row else 0
    
    def get_login_logs(self, limit: int = 50) -> List[LoginLog]:
        """
        获取最近的登录日志
        ====================
        参数：
          limit : 返回条数
        返回：
          LoginLog 对象列表
        """
        self.cursor.execute("""
            SELECT * FROM login_logs ORDER BY created_at DESC LIMIT ?
        """, (limit,))
        rows = self.cursor.fetchall()
        results = []
        for row in rows:
            results.append(LoginLog(
                log_id=row[0],
                account_id=row[1],
                username=row[2],
                action=row[3],
                ip_address=row[4],
                detail=row[5],
                created_at=row[6],
            ))
        return results
    
    # ==================== 7. 会话令牌操作（跨刷新登录保持） ====================
    def save_session_token(self, token: str, account_id: str, expires_days: int = 7) -> bool:
        """
        保存会话令牌（登录成功后生成）
        ===============================
        参数：
          token        : 会话令牌（UUID，存 URL query_params）
          account_id   : 关联账户 ID
          expires_days : 过期天数（默认 7 天）
        返回：
          True=成功 / False=失败
        """
        try:
            from datetime import timedelta
            expires_at = datetime.now() + timedelta(days=expires_days)
            self.cursor.execute("""
                INSERT INTO session_tokens (token, account_id, created_at, expires_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(token) DO UPDATE SET
                    account_id = excluded.account_id,
                    expires_at = excluded.expires_at
            """, (token, account_id, datetime.now(), expires_at))
            self.conn.commit()
            # Redis 写穿：鉴权走缓存，miss 回源数据库（Redis 是加速层，不是真相源）
            ttl = min(int(expires_days * 86400), REDIS_SESSION_TTL)
            get_redis().set(get_redis().key("session", token), account_id, ttl=ttl)
            return True
        except Exception as e:
            logger.error(f"[FAIL] 保存会话令牌失败: {e}")
            return False
    
    def get_account_id_by_token(self, token: str) -> Optional[str]:
        """
        根据会话令牌查询账户 ID（验证令牌有效性）
        ==========================================
        参数：
          token : 会话令牌
        返回：
          有效 → 账户 ID / 无效/过期 → None
        """
        try:
            # ① 先查 Redis（高频鉴权路径，避免每请求打库）
            rd = get_redis()
            cache_key = rd.key("session", token)
            cached = rd.get(cache_key)
            if cached:
                return cached

            # ② 回源数据库
            self.cursor.execute("""
                SELECT account_id FROM session_tokens
                WHERE token = ? AND (expires_at IS NULL OR expires_at > ?)
            """, (token, datetime.now()))
            row = self.cursor.fetchone()
            if not row:
                # 库里没有（已删/过期）→ 顺手清掉可能残留的缓存，避免脏读
                rd.delete(cache_key)
                return None

            account_id = row[0]
            rd.set(cache_key, account_id, ttl=REDIS_SESSION_TTL)
            return account_id
        except Exception as e:
            logger.error(f"[FAIL] 查询会话令牌失败: {e}")
            return None
    
    def delete_session_token(self, token: str) -> bool:
        """
        删除会话令牌（退出登录时清理）
        ===============================
        参数：
          token : 会话令牌
        返回：
          True=成功 / False=失败
        """
        try:
            self.cursor.execute("DELETE FROM session_tokens WHERE token = ?", (token,))
            self.conn.commit()
            rd = get_redis()
            rd.delete(rd.key("session", token))
            return True
        except Exception as e:
            logger.error(f"[FAIL] 删除会话令牌失败: {e}")
            return False
    
    def cleanup_expired_tokens(self) -> int:
        """
        清理过期会话令牌（定期维护）
        ============================
        返回：
          清理的记录数
        """
        try:
            self.cursor.execute(
                "DELETE FROM session_tokens WHERE expires_at < ?",
                (datetime.now(),)
            )
            self.conn.commit()
            deleted = self.cursor.rowcount
            if deleted:
                logger.info(f"[CLEAN] 已清理 {deleted} 条过期会话令牌")
            return deleted
        except Exception as e:
            logger.error(f"[FAIL] 清理过期令牌失败: {e}")
            return 0
    
    # ==================== 8. 对话会话操作（多对话框支持） ====================
    def save_session(self, session: 'ChatSession') -> bool:
        """
        保存或更新对话会话
        ===================
        参数：
          session : ChatSession 对象
        返回：
          True=成功 / False=失败
        """
        try:
            self.cursor.execute("""
                INSERT INTO sessions (session_id, account_id, title, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    title = excluded.title,
                    updated_at = excluded.updated_at
            """, (
                session.session_id,
                session.account_id,
                session.title,
                session.created_at,
                session.updated_at,
            ))
            self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"[FAIL] 保存会话失败: {e}")
            return False
    
    def get_session(self, session_id: str) -> Optional['ChatSession']:
        """
        根据 session_id 查询会话
        =========================
        参数：
          session_id : 会话 ID
        返回：
          ChatSession 对象 / None
        """
        self.cursor.execute(
            "SELECT * FROM sessions WHERE session_id = ?",
            (session_id,)
        )
        row = self.cursor.fetchone()
        if row is None:
            return None
        from models import ChatSession
        return ChatSession(
            session_id=row[0],
            account_id=row[1],
            title=row[2],
        )
    
    def list_sessions_by_account(self, account_id: str) -> List['ChatSession']:
        """
        获取指定账户的所有会话（按更新时间倒序）
        ==========================================
        参数：
          account_id : 账户 ID
        返回：
          ChatSession 对象列表
        """
        self.cursor.execute("""
            SELECT * FROM sessions
            WHERE account_id = ?
            ORDER BY updated_at DESC
        """, (account_id,))
        rows = self.cursor.fetchall()
        from models import ChatSession
        return [
            ChatSession(
                session_id=row[0],
                account_id=row[1],
                title=row[2],
            )
            for row in rows
        ]
    
    def update_session_title(self, session_id: str, title: str) -> bool:
        """
        更新会话标题（根据首条消息自动生成）
        =====================================
        参数：
          session_id : 会话 ID
          title      : 新标题
        返回：
          True=成功 / False=失败
        """
        try:
            self.cursor.execute("""
                UPDATE sessions SET title = ?, updated_at = CURRENT_TIMESTAMP
                WHERE session_id = ?
            """, (title, session_id))
            self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"[FAIL] 更新会话标题失败: {e}")
            return False
    
    def update_session_time(self, session_id: str) -> bool:
        """
        更新会话最后活跃时间
        =====================
        参数：
          session_id : 会话 ID
        返回：
          True=成功 / False=失败
        """
        try:
            self.cursor.execute("""
                UPDATE sessions SET updated_at = CURRENT_TIMESTAMP
                WHERE session_id = ?
            """, (session_id,))
            self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"[FAIL] 更新会话时间失败: {e}")
            return False
    
    def delete_session(self, session_id: str) -> bool:
        """
        删除会话及其所有消息
        =====================
        参数：
          session_id : 会话 ID
        返回：
          True=成功 / False=失败
        """
        try:
            # 先删除该会话的所有消息
            self.cursor.execute(
                "DELETE FROM messages WHERE session_id = ?",
                (session_id,)
            )
            # 再删除会话本身
            self.cursor.execute(
                "DELETE FROM sessions WHERE session_id = ?",
                (session_id,)
            )
            self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"[FAIL] 删除会话失败: {e}")
            return False
    
    def count_sessions_by_account(self, account_id: str) -> int:
        """
        统计指定账户的会话数量
        =======================
        参数：
          account_id : 账户 ID
        返回：
          会话数量
        """
        self.cursor.execute(
            "SELECT COUNT(*) FROM sessions WHERE account_id = ?",
            (account_id,)
        )
        return self.cursor.fetchone()[0]
    
    # ==================== 9. 用户信息操作 ====================
    def save_user(self, user: User) -> bool:
        """
        保存或更新用户信息
        ====================
        参数：
          user : User 对象（含 user_id / name / phone）
        返回：
          True=成功 / False=失败
        """
        try:
            self.cursor.execute("""
                INSERT INTO users (user_id, name, phone, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                -- ON CONFLICT 处理重复 user_id 的情况
                ON CONFLICT(user_id) DO UPDATE SET
                    name = excluded.name,            -- 更新姓名
                    phone = excluded.phone,          -- 更新电话
                    updated_at = CURRENT_TIMESTAMP   -- 更新修改时间
            """, (
                user.user_id,
                user.name,
                user.phone,
                user.created_at,
                user.updated_at,
            ))
            self.conn.commit()
            logger.info(f"[USER] 用户已保存: {user.name} ({user.user_id})")
            return True
        except Exception as e:
            logger.error(f"[FAIL] 保存用户失败: {e}")
            return False
    
    def get_user(self, user_id: str) -> Optional[User]:
        """
        根据 user_id 查询用户信息
        ===========================
        参数：
          user_id : 用户唯一标识
        返回：
          找到 → User 对象 / 未找到 → None
        """
        self.cursor.execute(
            "SELECT * FROM users WHERE user_id = ?",
            (user_id,)
        )
        row = self.cursor.fetchone()
        if row is None:
            return None
        
        # 将数据库行数据转换为 User 对象
        return User(
            user_id=row[0],
            name=row[1],
            phone=row[2],
        )
    
    # ==================== 10. 对话历史操作 ====================
    def save_message(self, message: Message) -> bool:
        """
        保存对话消息到数据库（支持多会话）
        ===================================
        参数：
          message : Message 对象（需含 session_id）
        返回：
          True=成功 / False=失败

        入口拦截（防"单条超长"打爆上下文）：
          · 单条消息超过 MAX_SINGLE_MESSAGE_CHARS 时，落库前截断并加提示尾标；
          · 只截断"入库内容"，不改调用方对象，避免影响当次回答；
          · 截断是"止血"手段——保证后续 get_history 不会因单条超长而爆窗口。
        """
        content = message.content or ""
        if len(content) > MAX_SINGLE_MESSAGE_CHARS:
            original_len = len(content)
            content = (
                content[:MAX_SINGLE_MESSAGE_CHARS]
                + f"\n…[内容过长已截断，原始 {original_len} 字]"
            )
            logger.warning(
                f"[TRIM] 单条消息超长（{original_len} > {MAX_SINGLE_MESSAGE_CHARS} 字），"
                f"已截断入库 session={message.session_id}"
            )

        try:
            self.cursor.execute("""
                INSERT INTO messages (message_id, session_id, user_id, role, content, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                message.message_id,
                message.session_id,
                message.user_id,
                message.role,
                content,
                message.created_at,
            ))
            self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"[FAIL] 保存消息失败: {e}")
            return False
    
    # -------- Token 估算（不引入 tiktoken 依赖，用轻量近似） --------
    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """
        估算一段文本的 Token 数（轻量近似，不引入额外依赖）
        ==================================================
        口径（与通义千问中文场景的近似）：
          · 中文：约 1 字 ≈ 1 Token
          · 英文 / 数字 / 符号：约 4 字符 ≈ 1 Token
        说明：这是"够用就好"的估算，用于裁剪决策，不追求与官方分词器完全一致；
              宁可略微高估（多裁一点），也不低估（避免真的超限）。
        """
        if not text:
            return 0
        # 统计中文字符数（CJK 统一表意文字区间）
        cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
        other = len(text) - cjk
        return cjk + (other + 3) // 4  # 非中文按 4 字符 1 Token 向上取整

    def get_history(self, session_id: str, limit: int = None) -> List[Message]:
        """
        获取指定会话的最近 N 条消息（轮数 + Token 双闸门裁剪）
        ======================================================
        参数：
          session_id : 会话 ID
          limit      : 返回的最大消息数（默认用 MAX_HISTORY_ROUNDS * 2）
                       *2 是因为每轮对话包含 user + assistant 两条消息
        返回：
          Message 对象列表（按时间正序）

        裁剪策略（两道闸门，先粗后细）：
          ① 轮数闸门：先按 MAX_HISTORY_ROUNDS 取候选（SQL LIMIT，粗筛）；
          ② Token 闸门：从最新往旧累加，超过 MAX_HISTORY_TOKENS 即停（细裁）；
          ③ 兜底：若 Token 裁剪后一条都不剩（极端超长单条），
             退化为只保留最近 MIN_HISTORY_ROUNDS_ON_OVERFLOW 轮，保证主链路不空转。
        """
        if limit is None:
            limit = MAX_HISTORY_ROUNDS * 2

        self.cursor.execute("""
            SELECT * FROM messages
            WHERE session_id = ?
            ORDER BY created_at DESC
            LIMIT ?
        """, (session_id, limit))

        rows = self.cursor.fetchall()
        rows.reverse()  # 转为时间正序（旧 → 新）

        # -------- ② Token 闸门：从最新往旧累加，超限即停 --------
        kept: List = []
        used_tokens = 0
        for row in reversed(rows):  # 从最新开始
            msg_tokens = self._estimate_tokens(row[4] or "")
            if kept and used_tokens + msg_tokens > MAX_HISTORY_TOKENS:
                break  # 再加就超限，停止（保留已累加的较新消息）
            kept.append(row)
            used_tokens += msg_tokens
        kept.reverse()  # 还原为时间正序

        # -------- ③ 兜底：极端场景（单条就超限）至少保留最近 N 轮 --------
        if not kept and rows:
            fallback_n = max(MIN_HISTORY_ROUNDS_ON_OVERFLOW * 2, 1)
            kept = rows[-fallback_n:]
            logger.warning(
                f"[TRIM] 会话 {session_id} 单条消息超 Token 上限，"
                f"兜底保留最近 {len(kept)} 条"
            )

        return [
            Message(
                message_id=row[0],
                session_id=row[1],
                user_id=row[2],
                role=row[3],
                content=row[4],
            )
            for row in kept
        ]
    
    def get_history_dicts(self, session_id: str) -> List[dict]:
        """
        获取对话历史（字典格式，直接传给 LLM）
        =========================================
        参数：
          session_id : 会话 ID
        返回：
          [{"role": "user", "content": "你好"}, {"role": "assistant", "content": "你好！"}]
        """
        messages = self.get_history(session_id)
        return [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]
    
    def get_all_session_messages(self, session_id: str) -> List[Message]:
        """
        获取指定会话的全部消息（恢复对话用）
        =====================================
        参数：
          session_id : 会话 ID
        返回：
          Message 对象列表（按时间正序）
        """
        self.cursor.execute("""
            SELECT * FROM messages
            WHERE session_id = ?
            ORDER BY created_at ASC
        """, (session_id,))
        
        rows = self.cursor.fetchall()
        return [
            Message(
                message_id=row[0],
                session_id=row[1],
                user_id=row[2],
                role=row[3],
                content=row[4],
            )
            for row in rows
        ]
    
    def clear_history(self, session_id: str) -> bool:
        """
        清除指定会话的所有消息
        =======================
        参数：
          session_id : 会话 ID
        返回：
          True=成功 / False=失败
        """
        try:
            self.cursor.execute(
                "DELETE FROM messages WHERE session_id = ?",
                (session_id,)
            )
            self.conn.commit()
            logger.info(f"🗑️ 已清除会话 {session_id} 的消息")
            return True
        except Exception as e:
            logger.error(f"[FAIL] 清除历史失败: {e}")
            return False
    
    
    # ==================== 11. 客户画像操作（CRM 核心） ====================
    # 显式列清单：保证 _row_to_customer 的列顺序稳定
    # （物理列序 = 建表语句列序；老库经 _migrate_customer_columns 追加后亦一致）
    _CUSTOMER_COLUMNS = (
        "customer_id", "name", "company", "position", "phone", "email",
        "notes", "tags", "created_at", "updated_at",
        "age", "identity", "school_grade", "intent_course", "matched_course",
        "intent_price", "hourly_fee", "active_status", "advisor_id",
        "source", "last_follow_up",
    )

    @staticmethod
    def _parse_dt(value):
        """把 SQLite 文本时间戳解析为 datetime（兼容多种格式）"""
        if not value:
            return None
        if isinstance(value, datetime):
            return value
        text = str(value)
        for fmt in (
            "%Y-%m-%d %H:%M:%S.%f",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%dT%H:%M:%S",
        ):
            try:
                return datetime.strptime(text, fmt)
            except ValueError:
                continue
        return None

    def _row_to_customer(self, row, columns=None) -> Optional[CustomerProfile]:
        """数据库行 → CustomerProfile（对列缺失/顺序变化具备一定容错）"""
        if row is None:
            return None
        cols = list(columns) if columns else list(self._CUSTOMER_COLUMNS)
        data = dict(zip(cols, row))

        tags = []
        raw_tags = data.get("tags")
        if raw_tags:
            try:
                tags = json.loads(raw_tags)
            except (json.JSONDecodeError, TypeError):
                tags = []

        return CustomerProfile(
            customer_id=data.get("customer_id"),
            name=data.get("name") or "",
            phone=data.get("phone"),
            email=data.get("email"),
            age=data.get("age"),
            identity=data.get("identity"),
            school_grade=data.get("school_grade"),
            intent_course=data.get("intent_course"),
            matched_course=data.get("matched_course"),
            intent_price=data.get("intent_price"),
            hourly_fee=data.get("hourly_fee"),
            active_status=data.get("active_status"),
            advisor_id=data.get("advisor_id"),
            source=data.get("source"),
            last_follow_up=data.get("last_follow_up"),
            company=data.get("company"),
            position=data.get("position"),
            notes=data.get("notes"),
            tags=tags,
        )

    def save_customer(self, profile: CustomerProfile) -> bool:
        """
        保存或更新客户画像（K12 教育版 · ON CONFLICT 全字段更新）
        ========================================================
        参数：
          profile : CustomerProfile 对象
        返回：
          True=成功 / False=失败
        """
        try:
            tags_json = json.dumps(profile.tags or [], ensure_ascii=False)
            self.cursor.execute("""
                INSERT INTO customer_profiles (
                    customer_id, name, company, position, phone, email, notes, tags,
                    created_at, updated_at,
                    age, identity, school_grade, intent_course, matched_course,
                    intent_price, hourly_fee, active_status, advisor_id, source, last_follow_up
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(customer_id) DO UPDATE SET
                    name = excluded.name,
                    company = excluded.company,
                    position = excluded.position,
                    phone = excluded.phone,
                    email = excluded.email,
                    notes = excluded.notes,
                    tags = excluded.tags,
                    age = excluded.age,
                    identity = excluded.identity,
                    school_grade = excluded.school_grade,
                    intent_course = excluded.intent_course,
                    matched_course = excluded.matched_course,
                    intent_price = excluded.intent_price,
                    hourly_fee = excluded.hourly_fee,
                    active_status = excluded.active_status,
                    advisor_id = excluded.advisor_id,
                    source = excluded.source,
                    last_follow_up = excluded.last_follow_up,
                    updated_at = CURRENT_TIMESTAMP
            """, (
                profile.customer_id, profile.name, profile.company, profile.position,
                profile.phone, profile.email, profile.notes, tags_json,
                profile.created_at, profile.updated_at,
                profile.age, profile.identity, profile.school_grade,
                profile.intent_course, profile.matched_course,
                profile.intent_price, profile.hourly_fee,
                profile.active_status, profile.advisor_id,
                profile.source, profile.last_follow_up,
            ))
            self.conn.commit()
            logger.info(f"[USER] 客户画像已保存: {profile.name} ({profile.customer_id})")
            return True
        except Exception as e:
            logger.error(f"[FAIL] 保存客户画像失败: {e}")
            return False

    def get_customer(self, customer_id: str) -> Optional[CustomerProfile]:
        """
        根据 customer_id 查询客户画像
        ===============================
        参数：
          customer_id : 客户唯一标识
        返回：
          CustomerProfile 对象 / None
        """
        self.cursor.execute(
            "SELECT * FROM customer_profiles WHERE customer_id = ?",
            (customer_id,),
        )
        return self._row_to_customer(
            self.cursor.fetchone(), [d[0] for d in self.cursor.description]
        )

    def search_customers(self, keyword: str) -> List[CustomerProfile]:
        """
        按关键词搜索客户（模糊匹配姓名/公司/备注/意向课程/年级）
        ==========================================================
        参数：
          keyword : 搜索关键词
        返回：
          匹配的 CustomerProfile 对象列表
        """
        pattern = f"%{keyword}%"
        self.cursor.execute("""
            SELECT * FROM customer_profiles
            WHERE name LIKE ?
               OR company LIKE ?
               OR notes LIKE ?
               OR intent_course LIKE ?
               OR school_grade LIKE ?
            ORDER BY updated_at DESC
        """, (pattern, pattern, pattern, pattern, pattern))
        cols = [d[0] for d in self.cursor.description]
        return [self._row_to_customer(row, cols) for row in self.cursor.fetchall()]

    def list_all_customers(self) -> List[CustomerProfile]:
        """
        列出所有客户画像（按更新时间倒序）
        ====================================
        返回：
          所有 CustomerProfile 对象列表
        """
        self.cursor.execute(
            "SELECT * FROM customer_profiles ORDER BY updated_at DESC"
        )
        cols = [d[0] for d in self.cursor.description]
        return [self._row_to_customer(row, cols) for row in self.cursor.fetchall()]

    def list_active_customers(self) -> List[CustomerProfile]:
        """
        列出「私海」客户（退出调度过滤 · 六、SQL逻辑过滤）
        ==================================================
        背景：45 天退场后客户掉入公海，直接退出定时调度，不再占用销售待办。
        做法：精筛 / 粗筛 / 事件消费全部走本方法，SQL 永远带
              WHERE pool_type = '私海'（兼容历史数据 pool_type 为 NULL 的情况），
              公海客户在**底层直接跳过**，不进入任何扫描。
        返回：
          私海（含历史 NULL）CustomerProfile 列表
        """
        self.cursor.execute("""
            SELECT * FROM customer_profiles
            WHERE pool_type = '私海' OR pool_type IS NULL
            ORDER BY updated_at DESC
        """)
        cols = [d[0] for d in self.cursor.description]
        return [self._row_to_customer(row, cols) for row in self.cursor.fetchall()]

    def list_public_pool_customers(self) -> List[CustomerProfile]:
        """
        列出「公海」客户（长尾培育对象）
        ==================================
        返回：
          公海 CustomerProfile 列表（交给 AI 做长尾培育 / 定期推送资料）
        """
        self.cursor.execute("""
            SELECT * FROM customer_profiles
            WHERE pool_type = '公海'
            ORDER BY updated_at DESC
        """)
        cols = [d[0] for d in self.cursor.description]
        return [self._row_to_customer(row, cols) for row in self.cursor.fetchall()]

    # ==================== 11c. 45天退场与公海机制（六） ====================
    def move_to_public_pool(self, customer_id: str) -> bool:
        """
        掉入公海（45天周期走完仍未转化，销售手动标记流失）
        ====================================================
        动作：
          1. 【归属顾问】置空（advisor_id=NULL）
          2. 【池子类型】改为公海（pool_type='公海'）
          3. 标记「沉睡/流失」（is_sleeping=1）
          4. 记录退出调度时间（exit_schedule_at）
          5. 撤销该客户所有未触发的延迟消息（退出调度）
          6. 关闭该客户未读告警（不再占用销售待办）
        返回：True=成功 / False=失败
        """
        try:
            self.cursor.execute("""
                UPDATE customer_profiles
                SET advisor_id = NULL,
                    pool_type = '公海',
                    is_sleeping = 1,
                    exit_schedule_at = CURRENT_TIMESTAMP,
                    updated_at = CURRENT_TIMESTAMP
                WHERE customer_id = ?
            """, (customer_id,))
            self.conn.commit()
            # 退出调度：撤销延迟消息 + 关闭未读告警
            self.cancel_follow_up_check(customer_id)
            self.resolve_alerts(customer_id)
            logger.info(f"[POOL] 客户已掉入公海（退出调度）: {customer_id}")
            return True
        except Exception as e:
            logger.error(f"[FAIL] 掉入公海失败: {e}")
            return False

    def wake_up_customer(self, customer_id: str, advisor_id: str) -> bool:
        """
        唤醒回流（顾问手动打标记 → 重新分配 → 重回 R1/R2/R3 巡检周期）
        =================================================================
        动作：
          1. 【池子类型】改回私海（pool_type='私海'）
          2. 清除沉睡标记（is_sleeping=0）
          3. 重新分配归属顾问（advisor_id）
          4. 清除退出调度时间（exit_schedule_at=NULL）
          5. 重新投递延迟消息（10 天后检查），再次进入巡检周期
        返回：True=成功 / False=失败
        """
        try:
            self.cursor.execute("""
                UPDATE customer_profiles
                SET pool_type = '私海',
                    is_sleeping = 0,
                    advisor_id = ?,
                    exit_schedule_at = NULL,
                    updated_at = CURRENT_TIMESTAMP
                WHERE customer_id = ?
            """, (advisor_id, customer_id))
            self.conn.commit()
            # 重新进入巡检周期：投递延迟消息
            self.schedule_follow_up_check(customer_id, delay_days=10)
            logger.info(f"[POOL] 客户已唤醒回流: {customer_id} → 顾问 {advisor_id}")
            return True
        except Exception as e:
            logger.error(f"[FAIL] 唤醒回流失败: {e}")
            return False

    # ==================== 11d. 延迟消息（事件驱动 · 精准触发 + 撤销） ====================
    def schedule_follow_up_check(self, customer_id: str, delay_days: int = 10) -> str:
        """
        投递延迟消息（销售录入信息 → 10 天后触发检查）
        ================================================
        参数：
          customer_id : 客户 ID
          delay_days  : 延迟天数（默认 10 天）
        返回：
          msg_id（消息唯一标识）
        """
        from datetime import timedelta
        msg_id = str(uuid.uuid4())
        fire_at = datetime.now() + timedelta(days=delay_days)
        try:
            self.cursor.execute("""
                INSERT INTO delayed_messages
                    (msg_id, customer_id, event_type, fire_at, status)
                VALUES (?, ?, 'follow_up_check', ?, 'scheduled')
            """, (msg_id, customer_id, fire_at))
            self.conn.commit()
            # Redis ZSET 加速：score=触发时间戳；数据库仍是真相源（可对账重建）
            rd = get_redis()
            rd.zadd(rd.key("delay", "follow_up"), {msg_id: fire_at.timestamp()})
            logger.info(f"[EVENT] 已投递延迟消息: {customer_id} → {fire_at:%Y-%m-%d %H:%M}")
        except Exception as e:
            logger.error(f"[FAIL] 投递延迟消息失败: {e}")
        return msg_id

    def cancel_follow_up_check(self, customer_id: str) -> int:
        """
        撤销延迟消息（客户转化或已跟进 → 零无效扫描）
        ================================================
        参数：
          customer_id : 客户 ID
        返回：
          被撤销的消息条数
        """
        try:
            # 先取待撤销的 msg_id（用于同步清理 Redis ZSET）
            self.cursor.execute("""
                SELECT msg_id FROM delayed_messages
                WHERE customer_id = ? AND status = 'scheduled'
            """, (customer_id,))
            msg_ids = [r[0] for r in self.cursor.fetchall()]

            self.cursor.execute("""
                UPDATE delayed_messages SET status = 'cancelled'
                WHERE customer_id = ? AND status = 'scheduled'
            """, (customer_id,))
            self.conn.commit()
            n = self.cursor.rowcount
            if msg_ids:
                rd = get_redis()
                rd.zrem(rd.key("delay", "follow_up"), *msg_ids)
            if n:
                logger.info(f"[EVENT] 已撤销延迟消息 {n} 条: {customer_id}")
            return n
        except Exception as e:
            logger.error(f"[FAIL] 撤销延迟消息失败: {e}")
            return 0

    def fetch_due_delayed_messages(self, limit: int = 500) -> List[dict]:
        """
        取出到期的延迟消息（fire_at <= now 且 status='scheduled'）
        ============================================================
        返回：
          [{"msg_id", "customer_id", "event_type", "fire_at"}, ...]
        """
        try:
            # ① Redis ZSET 优先：只取到期成员，避免全表按时间扫
            rd = get_redis()
            delay_key = rd.key("delay", "follow_up")
            due_ids = rd.zrange_by_score(
                delay_key, datetime.now().timestamp(), limit=limit
            )
            if due_ids:
                placeholders = ",".join("?" for _ in due_ids)
                self.cursor.execute(f"""
                    SELECT msg_id, customer_id, event_type, fire_at
                    FROM delayed_messages
                    WHERE msg_id IN ({placeholders})
                      AND status = 'scheduled' AND fire_at <= ?
                    ORDER BY fire_at ASC
                """, tuple(due_ids) + (datetime.now(),))
                cols = [d[0] for d in self.cursor.description]
                rows = [dict(zip(cols, row)) for row in self.cursor.fetchall()]
                # ZSET 里有、但库里已不是 scheduled 的成员（已取消/已触发）→ 清理
                found = {r["msg_id"] for r in rows}
                stale = [m for m in due_ids if m not in found]
                if stale:
                    rd.zrem(delay_key, *stale)
                if rows:
                    return rows

            # ② 回退数据库查询（Redis 不可用 / ZSET 为空时兜底）
            self.cursor.execute("""
                SELECT msg_id, customer_id, event_type, fire_at
                FROM delayed_messages
                WHERE status = 'scheduled' AND fire_at <= ?
                ORDER BY fire_at ASC
                LIMIT ?
            """, (datetime.now(), limit))
            cols = [d[0] for d in self.cursor.description]
            return [dict(zip(cols, row)) for row in self.cursor.fetchall()]
        except Exception as e:
            logger.error(f"[FAIL] 取到期延迟消息失败: {e}")
            return []

    def mark_delayed_fired(self, msg_id: str) -> bool:
        """把延迟消息标记为已触发（fired）"""
        try:
            self.cursor.execute(
                "UPDATE delayed_messages SET status = 'fired' WHERE msg_id = ?",
                (msg_id,),
            )
            self.conn.commit()
            rd = get_redis()
            rd.zrem(rd.key("delay", "follow_up"), msg_id)
            return True
        except Exception as e:
            logger.error(f"[FAIL] 标记延迟消息已触发失败: {e}")
            return False

    # ==================== 11e. 今日待办池（分层跑批 · 粗筛/事件 → 精筛） ====================
    def add_pending_todo(
        self,
        customer_id: str,
        advisor_id: Optional[str] = None,
        source: str = "coarse",
        trigger_at=None,
        biz_date: Optional[str] = None,
    ) -> bool:
        """
        把客户塞入「今日待办池」（粗筛 / 事件驱动产出）
        ==================================================
        去重：同一客户 + 同一业务日只保留一条 pending（避免重复塞入）。
        参数：
          customer_id : 客户 ID
          advisor_id  : 归属顾问
          source      : 来源 event(事件驱动) / coarse(21点粗筛)
          trigger_at  : 事件驱动的触发时间
          biz_date    : 业务日 YYYY-MM-DD（默认今天）
        返回：True=新增 / False=已存在或失败
        """
        biz_date = biz_date or datetime.now().strftime("%Y-%m-%d")
        try:
            # 先查是否已存在（同客户 + 同业务日 + pending）
            self.cursor.execute("""
                SELECT todo_id FROM pending_todo
                WHERE customer_id = ? AND biz_date = ? AND status = 'pending'
                LIMIT 1
            """, (customer_id, biz_date))
            if self.cursor.fetchone():
                return False
            self.cursor.execute("""
                INSERT INTO pending_todo
                    (todo_id, customer_id, advisor_id, source, trigger_at, biz_date, status)
                VALUES (?, ?, ?, ?, ?, ?, 'pending')
            """, (str(uuid.uuid4()), customer_id, advisor_id, source, trigger_at, biz_date))
            self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"[FAIL] 塞入待办池失败: {e}")
            return False

    def list_pending_todos(self, biz_date: Optional[str] = None, limit: int = 1000) -> List[dict]:
        """
        列出「今日待办池」中待处理的客户（精筛输入）
        ================================================
        参数：
          biz_date : 业务日 YYYY-MM-DD（默认今天）
          limit    : 最多返回条数
        返回：
          [{"todo_id", "customer_id", "advisor_id", "source", "trigger_at"}, ...]
        """
        biz_date = biz_date or datetime.now().strftime("%Y-%m-%d")
        try:
            self.cursor.execute("""
                SELECT todo_id, customer_id, advisor_id, source, trigger_at
                FROM pending_todo
                WHERE biz_date = ? AND status = 'pending'
                ORDER BY created_at ASC
                LIMIT ?
            """, (biz_date, limit))
            cols = [d[0] for d in self.cursor.description]
            return [dict(zip(cols, row)) for row in self.cursor.fetchall()]
        except Exception as e:
            logger.error(f"[FAIL] 列出待办池失败: {e}")
            return []

    def mark_todo_done(self, todo_id: str, status: str = "done") -> bool:
        """把待办标记为已处理（done / expired）"""
        try:
            self.cursor.execute(
                "UPDATE pending_todo SET status = ? WHERE todo_id = ?",
                (status, todo_id),
            )
            self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"[FAIL] 标记待办完成失败: {e}")
            return False

    def eval_anomaly_rules(self, customer_id: str) -> List[str]:
        """
        对单个客户重新校验异常规则（字段驱动闭环的核心）
        ==================================================
        背景：精筛时不直接信任待办池，而是**重新校验**——
              销售跟进后把【最近跟进时间】或【转化状态】改成非命中值，
              下次精筛 reasons 为空 → 不写告警，旧告警自动消除。
        返回：
          命中原因列表（空列表 = 已不命中，告警应消除）
        """
        from datetime import timedelta
        c = self.get_customer(customer_id)
        if not c:
            return []
        # 公海客户直接跳过（退出调度）
        pool_type = getattr(c, "pool_type", None)
        if pool_type == "公海":
            return []

        now = datetime.now()
        stale_before = now - timedelta(days=10)
        churn_before = now - timedelta(days=30)
        high_risk_before = now - timedelta(days=45)

        reasons: List[str] = []
        last = self._parse_dt(c.last_follow_up)
        status = c.active_status or ""
        has_intent = bool(c.intent_course or c.intent_price)
        converted = any(kw in status for kw in ("成交", "已报名", "已转化", "在读"))

        if has_intent and (last is None or last < stale_before):
            reasons.append("有意向未跟进(>10天或无跟进)")
        if ("流失" in status) or ("沉睡" in status):
            reasons.append("流失/沉睡风险")
        elif (not converted) and (last is None or last < churn_before):
            reasons.append("正价课未转化(>30天)")
        if (not converted) and (last is None or last < high_risk_before):
            reasons.append("高风险低意向(>45天未激活/未转化)")
        return reasons

    def resolve_alerts(self, customer_id: str) -> int:
        """
        字段驱动闭环：客户不再命中规则 → 未读告警自动消除
        ====================================================
        参数：
          customer_id : 客户 ID
        返回：
          被消除（标记 resolved）的告警条数
        """
        try:
            self.cursor.execute("""
                UPDATE alerts SET status = 'resolved'
                WHERE customer_id = ? AND status = 'unread'
            """, (customer_id,))
            self.conn.commit()
            return self.cursor.rowcount
        except Exception as e:
            logger.error(f"[FAIL] 消除告警失败: {e}")
            return 0

    def delete_customer(self, customer_id: str) -> bool:
        """
        删除客户画像
        ==============
        参数：
          customer_id : 客户唯一标识
        返回：
          True=成功 / False=失败
        """
        try:
            self.cursor.execute(
                "DELETE FROM customer_profiles WHERE customer_id = ?",
                (customer_id,),
            )
            self.conn.commit()
            logger.info(f"已删除客户: {customer_id}")
            return True
        except Exception as e:
            logger.error(f"[FAIL] 删除客户失败: {e}")
            return False

    def screen_anomaly_customers(
        self,
        stale_days: int = 10,
        churn_days: int = 30,
        high_risk_days: int = 45,
    ) -> List[dict]:
        """
        异常客户筛查（规则版 · 结果可解释）
        ====================================
        设计目标：主动挑出「需要人工介入的客户」，避免高价值线索被遗漏。
        三条规则（按业务优先级排序，可解释、可复现）：
          R1 有意向但没跟进：已填意向课程/预算（有意向），但无跟进记录或超
             stale_days 天没跟（默认 10 天）——高价值线索正在冷却，最该先捞。
          R2 快流失或已沉睡：active_status 命中「流失/沉睡」，或超 churn_days 天
             （默认 30 天）正价课仍未转化成功——再不跟就彻底丢了。
          R3 高风险低意向：超 high_risk_days 天（默认 45 天）仍未激活/转化成功的
             低意向客户——长期沉默、转化概率低，属高风险名单，做最后触达或归档。
        参数：
          stale_days     : R1「有意向未跟进」阈值天数（默认 10 天）
          churn_days     : R2「正价课未转化」阈值天数（默认 30 天）
          high_risk_days : R3「未激活/未转化」阈值天数（默认 45 天）
        返回：
          [{"customer_id", "name", "advisor_id", "reasons": [...]}, ...]
        """
        from datetime import timedelta
        now = datetime.now()
        stale_before = now - timedelta(days=stale_days)
        churn_before = now - timedelta(days=churn_days)
        high_risk_before = now - timedelta(days=high_risk_days)

        results = []
        # 退出调度过滤：只扫私海客户（公海客户底层直接跳过，不占待办）
        for c in self.list_active_customers():
            reasons = []
            last = self._parse_dt(c.last_follow_up)
            status = c.active_status or ""
            # 「有意向」= 填了意向课程或预算；「已转化」= 状态里带成交/已报名等正向词
            has_intent = bool(c.intent_course or c.intent_price)
            converted = any(
                kw in status for kw in ("成交", "已报名", "已转化", "在读")
            )

            # -------- R1 有意向但没跟进（>stale_days 天或无跟进） --------
            if has_intent and (last is None or last < stale_before):
                reasons.append(f"有意向未跟进(>{stale_days}天或无跟进)")

            # -------- R2 快流失或已沉睡 --------
            # ① 状态直接命中流失/沉睡；② 超 churn_days 天正价课仍未转化成功
            if ("流失" in status) or ("沉睡" in status):
                reasons.append("流失/沉睡风险")
            elif (not converted) and (last is None or last < churn_before):
                reasons.append(f"正价课未转化(>{churn_days}天)")

            # -------- R3 高风险低意向（>high_risk_days 天未激活/未转化） --------
            if (not converted) and (last is None or last < high_risk_before):
                reasons.append(f"高风险低意向(>{high_risk_days}天未激活/未转化)")

            if reasons:
                results.append({
                    "customer_id": c.customer_id,
                    "name": c.name,
                    "advisor_id": c.advisor_id,
                    "reasons": reasons,
                })
        return results

    # ==================== 11b. 跟进记录 CRUD（B.4 跟进闭环 · 产品化） ====================
    _FOLLOW_UP_COLUMNS = (
        "follow_up_id", "customer_id", "advisor_id", "channel",
        "content", "result", "next_follow_up_at", "created_at",
    )

    def _row_to_follow_up(self, row, columns=None) -> Optional[FollowUp]:
        """数据库行 → FollowUp（对列缺失 / 顺序变化具备一定容错）"""
        if row is None:
            return None
        cols = list(columns) if columns else list(self._FOLLOW_UP_COLUMNS)
        data = dict(zip(cols, row))
        return FollowUp(
            follow_up_id=data.get("follow_up_id") or "",
            customer_id=data.get("customer_id") or "",
            advisor_id=data.get("advisor_id"),
            channel=data.get("channel") or "电话",
            content=data.get("content"),
            result=data.get("result") or "已联系",
            next_follow_up_at=data.get("next_follow_up_at"),
            created_at=self._parse_dt(data.get("created_at")) or datetime.now(),
        )

    def save_follow_up(self, record: FollowUp) -> bool:
        """
        保存一条跟进记录，并回写客户画像的「最近跟进时间」
        ==================================================
        参数：
          record : FollowUp 对象（客户 / 跟进人 / 方式 / 内容 / 结果 / 下次跟进时间）
        返回：
          True=成功 / False=失败

        设计（写入权限纪律）：
          只回写 last_follow_up 这个「时间戳」低风险字段；
          active_status 等高风险字段不在本链路自动改写（仍走人工审核）。
          这样「跟进 → 回写」闭环复位，下一轮异常筛查 R1/R2 立刻读到最新跟进时间。
        """
        try:
            self.cursor.execute("""
                INSERT INTO follow_ups (
                    follow_up_id, customer_id, advisor_id, channel,
                    content, result, next_follow_up_at, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(follow_up_id) DO UPDATE SET
                    customer_id = excluded.customer_id,
                    advisor_id = excluded.advisor_id,
                    channel = excluded.channel,
                    content = excluded.content,
                    result = excluded.result,
                    next_follow_up_at = excluded.next_follow_up_at,
                    created_at = excluded.created_at
            """, (
                record.follow_up_id, record.customer_id, record.advisor_id,
                record.channel, record.content, record.result,
                record.next_follow_up_at, record.created_at,
            ))
            self.conn.commit()

            # -------- 回写画像「最近跟进时间」（低风险字段） --------
            try:
                self.cursor.execute("""
                    UPDATE customer_profiles
                    SET last_follow_up = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE customer_id = ?
                """, (record.created_at, record.customer_id))
                self.conn.commit()
            except Exception as inner:
                logger.warning(f"[WARN] 回写 last_follow_up 失败: {inner}")

            # -------- 事件驱动闭环：已跟进 → 撤销延迟消息（零无效扫描） --------
            # 销售跟进后，该客户不再需要「10 天后触发检查」的延迟消息，
            # 直接撤销，避免到期后做无效扫描。
            try:
                self.cancel_follow_up_check(record.customer_id)
            except Exception as inner:
                logger.warning(f"[WARN] 撤销延迟消息失败: {inner}")

            # -------- 可观测性埋点（失败不影响主流程） --------
            try:
                from metrics import incr
                incr("follow_up_total", channel=record.channel, result=record.result)
                if record.result in FOLLOW_UP_CONVERTED_RESULTS:
                    incr("follow_up_converted_total", result=record.result)
            except Exception:
                pass

            logger.info(
                f"[USER] 跟进记录已保存: {record.customer_id} "
                f"({record.channel} / {record.result})"
            )
            return True
        except Exception as e:
            logger.error(f"[FAIL] 保存跟进记录失败: {e}")
            return False

    def list_follow_ups(
        self, customer_id: Optional[str] = None, limit: int = 200
    ) -> List[FollowUp]:
        """
        查询跟进记录（按跟进时间倒序）
        ================================
        参数：
          customer_id : 客户 ID；为空则返回全部客户的跟进记录
          limit       : 最多返回条数（默认 200）
        返回：
          FollowUp 对象列表
        """
        try:
            if customer_id:
                self.cursor.execute("""
                    SELECT * FROM follow_ups
                    WHERE customer_id = ?
                    ORDER BY created_at DESC LIMIT ?
                """, (customer_id, int(limit)))
            else:
                self.cursor.execute("""
                    SELECT * FROM follow_ups
                    ORDER BY created_at DESC LIMIT ?
                """, (int(limit),))
            cols = [d[0] for d in self.cursor.description]
            return [self._row_to_follow_up(row, cols) for row in self.cursor.fetchall()]
        except Exception as e:
            logger.error(f"[FAIL] 查询跟进记录失败: {e}")
            return []

    def list_due_follow_ups(self, due_days: int = 14, limit: int = 200) -> List[dict]:
        """
        待跟进提醒清单（到期巡检 · 两类口径）
        ======================================
        ① 计划到期：每位客户「最新一条」跟进记录里填的 next_follow_up_at 已到点。
           取最新一条即可天然实现「跟过了就不再提醒」——后续跟进会取代它，
           若后续跟进没填 next_follow_up_at，则视为该客户不再需要提醒。
        ② 超期未跟进：画像 last_follow_up 超过 due_days 天，或从来没有跟进记录。
        参数：
          due_days : 「超期未跟进」阈值天数（默认 14 天）
          limit    : 最多返回条数
        返回：
          [{"customer_id", "name", "due_type", "days", "reasons": [...]}, ...]
        """
        from datetime import timedelta
        now = datetime.now()
        stale_before = now - timedelta(days=due_days)
        items: List[dict] = []
        seen = set()

        # -------- ① 计划到期（只看每位客户最新一条跟进记录） --------
        try:
            self.cursor.execute("""
                SELECT f.customer_id, f.next_follow_up_at
                FROM follow_ups f
                JOIN (
                    SELECT customer_id, MAX(created_at) AS latest_at
                    FROM follow_ups
                    GROUP BY customer_id
                ) m
                  ON m.customer_id = f.customer_id
                 AND m.latest_at = f.created_at
                WHERE f.next_follow_up_at IS NOT NULL
                  AND f.next_follow_up_at <= ?
                ORDER BY f.next_follow_up_at ASC
            """, (now,))
            planned = self.cursor.fetchall()
        except Exception as e:
            logger.warning(f"[WARN] 查询计划到期失败: {e}")
            planned = []

        for customer_id, next_at in planned:
            if customer_id in seen:
                continue
            self.cursor.execute(
                "SELECT name, advisor_id FROM customer_profiles WHERE customer_id = ?",
                (customer_id,),
            )
            row = self.cursor.fetchone()
            seen.add(customer_id)
            items.append({
                "customer_id": customer_id,
                "name": (row[0] if row else None) or "（客户已删除）",
                "advisor_id": row[1] if row else None,
                "due_type": "计划到期",
                "days": None,
                "reasons": [f"计划跟进时间已到（{next_at}）"],
            })
            if len(items) >= limit:
                return items

        # -------- ② 超期未跟进 --------
        # 退出调度过滤：只扫私海客户（公海客户底层直接跳过）
        for c in self.list_active_customers():
            if c.customer_id in seen:
                continue
            last = self._parse_dt(c.last_follow_up)
            if last is None or last < stale_before:
                days = None if last is None else (now - last).days
                items.append({
                    "customer_id": c.customer_id,
                    "name": c.name,
                    "advisor_id": c.advisor_id,
                    "due_type": "超期未跟进",
                    "days": days,
                    "reasons": [
                        f"超 {due_days} 天未跟进" if last else "从未跟进"
                    ],
                })
                if len(items) >= limit:
                    break
        return items

    def follow_up_conversion_stats(self, days: int = 30) -> dict:
        """
        跟进转化统计（B.4 追问「跟进转化率多少」的口径落地）
        ==================================================
        口径：
          分母 = 统计窗口内的跟进记录条数
          分子 = 结果为「已成交 / 有意向」的跟进条数（见 FOLLOW_UP_CONVERTED_RESULTS）
          转化率 = 分子 ÷ 分母（无数据时返回 None，绝不返回 0 冒充）
        参数：
          days : 统计窗口天数（默认近 30 天）
        返回：
          {"window_days", "total", "converted", "conversion_rate",
           "covered_customers", "by_result"}
        """
        from datetime import timedelta
        since = datetime.now() - timedelta(days=days)
        by_result: dict = {}
        total = 0
        try:
            self.cursor.execute("""
                SELECT result, COUNT(1) FROM follow_ups
                WHERE created_at >= ?
                GROUP BY result
            """, (since,))
            for result, count in self.cursor.fetchall():
                by_result[result or "未知"] = int(count or 0)
            total = sum(by_result.values())

            self.cursor.execute("""
                SELECT COUNT(DISTINCT customer_id) FROM follow_ups
                WHERE created_at >= ?
            """, (since,))
            covered = int((self.cursor.fetchone() or [0])[0] or 0)
        except Exception as e:
            logger.error(f"[FAIL] 跟进转化统计失败: {e}")
            return {
                "window_days": days, "total": 0, "converted": 0,
                "conversion_rate": None, "covered_customers": 0, "by_result": {},
            }

        converted = sum(
            n for r, n in by_result.items() if r in FOLLOW_UP_CONVERTED_RESULTS
        )
        return {
            "window_days": days,
            "total": total,
            "converted": converted,
            "conversion_rate": (round(converted / total, 4) if total else None),
            "covered_customers": covered,
            "by_result": by_result,
        }

    def delete_follow_up(self, follow_up_id: str) -> bool:
        """
        删除跟进记录
        ==============
        参数：
          follow_up_id : 跟进记录唯一标识
        返回：
          True=成功 / False=失败
        """
        try:
            self.cursor.execute(
                "DELETE FROM follow_ups WHERE follow_up_id = ?",
                (follow_up_id,),
            )
            self.conn.commit()
            logger.info(f"已删除跟进记录: {follow_up_id}")
            return True
        except Exception as e:
            logger.error(f"[FAIL] 删除跟进记录失败: {e}")
            return False


    # ==================== 11c. 主动告警 CRUD（B.1 ⑤ 兜底 · 定时调度产物） ====================
    _ALERT_COLUMNS = (
        "alert_id", "dedup_key", "customer_id", "customer_name", "advisor_id",
        "alert_type", "severity", "reasons", "source", "status",
        "created_at", "read_at",
    )

    def _row_to_alert(self, row, columns=None) -> dict:
        """数据库行 → 告警字典（reasons 反序列化为列表）"""
        if row is None:
            return {}
        cols = list(columns) if columns else list(self._ALERT_COLUMNS)
        data = dict(zip(cols, row))
        raw = data.get("reasons")
        try:
            data["reasons"] = json.loads(raw) if raw else []
        except (ValueError, TypeError):
            data["reasons"] = [raw] if raw else []
        return data

    def upsert_alert(
        self,
        customer_id: str,
        alert_type: str,
        reasons: List[str],
        customer_name: Optional[str] = None,
        advisor_id: Optional[str] = None,
        severity: str = "medium",
        source: str = "scheduler",
        biz_date: Optional[str] = None,
    ) -> bool:
        """
        写入 / 更新一条主动告警（按 dedup_key 幂等去重）
        ==================================================
        去重键 = customer_id + alert_type + 业务日（默认今天）
        同一客户同一类型同一天只保留一条：定时任务每轮重跑不会刷屏，
        但「已读」状态会被保留（重复命中不重置为未读）。

        参数：
          customer_id   : 客户 ID
          alert_type    : 告警类型（高意向未跟进 / 流失沉睡风险 / 资料不全 / 计划到期 / 超期未跟进）
          reasons       : 命中原因列表
          customer_name : 客户姓名（冗余展示）
          advisor_id    : 归属顾问（推送对象）
          severity      : 严重度 high / medium / low
          source        : 来源 scheduler / manual
          biz_date      : 业务日（默认取今天，格式 YYYY-MM-DD）
        返回：
          True=成功 / False=失败
        """
        try:
            day = biz_date or datetime.now().strftime("%Y-%m-%d")
            dedup_key = f"{customer_id}|{alert_type}|{day}"

            # Redis 前置去重：同一客户+类型+业务日只处理一次，避免定时任务每轮重复写库。
            # 三态语义：True=首次 / False=已存在（跳过）/ None=Redis 不可用（走数据库唯一键兜底）。
            dedup_redis_key = get_redis().key("alert", "dedup", dedup_key)
            acquired = get_redis().try_set_nx(
                dedup_redis_key, "1", ttl=REDIS_ALERT_DEDUP_TTL
            )
            if acquired is False:
                logger.debug(f"[CACHE] 告警已去重（Redis），跳过写库: {dedup_key}")
                return True

            self.cursor.execute("""
                INSERT INTO alerts (
                    alert_id, dedup_key, customer_id, customer_name, advisor_id,
                    alert_type, severity, reasons, source, status, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'unread', ?)
                ON CONFLICT(dedup_key) DO UPDATE SET
                    reasons = excluded.reasons,
                    severity = excluded.severity,
                    customer_name = excluded.customer_name,
                    advisor_id = excluded.advisor_id
            """, (
                str(uuid.uuid4()), dedup_key, customer_id, customer_name, advisor_id,
                alert_type, severity, json.dumps(reasons, ensure_ascii=False),
                source, datetime.now(),
            ))
            self.conn.commit()
            return True
        except Exception as e:
            # 写库失败 → 释放 Redis 去重标记，让下一轮还能重试
            try:
                if dedup_redis_key:
                    get_redis().delete(dedup_redis_key)
            except Exception:
                pass
            logger.error(f"[FAIL] 写入告警失败: {e}")
            return False

    def list_alerts(
        self,
        status: Optional[str] = None,
        advisor_id: Optional[str] = None,
        limit: int = 200,
    ) -> List[dict]:
        """
        查询告警列表（按产生时间倒序）
        ================================
        参数：
          status     : 过滤状态 unread / read；为空返回全部
          advisor_id : 过滤归属顾问；为空返回全部
          limit      : 最多返回条数
        返回：
          告警字典列表
        """
        try:
            sql = "SELECT * FROM alerts WHERE 1=1"
            params: List = []
            if status:
                sql += " AND status = ?"
                params.append(status)
            if advisor_id:
                sql += " AND advisor_id = ?"
                params.append(advisor_id)
            sql += " ORDER BY created_at DESC LIMIT ?"
            params.append(int(limit))
            self.cursor.execute(sql, tuple(params))
            cols = [d[0] for d in self.cursor.description]
            return [self._row_to_alert(r, cols) for r in self.cursor.fetchall()]
        except Exception as e:
            logger.error(f"[FAIL] 查询告警失败: {e}")
            return []

    def alert_exists(self, dedup_key: str) -> bool:
        """判断某去重键的告警是否已存在（用于区分“新增”与“更新”）"""
        try:
            self.cursor.execute(
                "SELECT 1 FROM alerts WHERE dedup_key = ? LIMIT 1", (dedup_key,)
            )
            return self.cursor.fetchone() is not None
        except Exception as e:
            logger.error(f"[FAIL] 查询告警去重键失败: {e}")
            return False

    def count_unread_alerts(self, advisor_id: Optional[str] = None) -> int:
        """统计未读告警数（前端红点 / 推送角标用）"""
        try:
            if advisor_id:
                self.cursor.execute(
                    "SELECT COUNT(1) FROM alerts WHERE status = 'unread' AND advisor_id = ?",
                    (advisor_id,),
                )
            else:
                self.cursor.execute(
                    "SELECT COUNT(1) FROM alerts WHERE status = 'unread'"
                )
            return int((self.cursor.fetchone() or [0])[0] or 0)
        except Exception as e:
            logger.error(f"[FAIL] 统计未读告警失败: {e}")
            return 0

    def mark_alert_read(self, alert_id: str) -> bool:
        """把一条告警标记为已读（幂等）"""
        try:
            self.cursor.execute("""
                UPDATE alerts
                SET status = 'read', read_at = CURRENT_TIMESTAMP
                WHERE alert_id = ? AND status != 'read'
            """, (alert_id,))
            self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"[FAIL] 标记告警已读失败: {e}")
            return False

    def mark_all_alerts_read(self, advisor_id: Optional[str] = None) -> int:
        """把未读告警全部标记为已读，返回受影响条数"""
        try:
            if advisor_id:
                self.cursor.execute("""
                    UPDATE alerts
                    SET status = 'read', read_at = CURRENT_TIMESTAMP
                    WHERE status = 'unread' AND advisor_id = ?
                """, (advisor_id,))
            else:
                self.cursor.execute("""
                    UPDATE alerts
                    SET status = 'read', read_at = CURRENT_TIMESTAMP
                    WHERE status = 'unread'
                """)
            affected = self.cursor.rowcount or 0
            self.conn.commit()
            return int(affected)
        except Exception as e:
            logger.error(f"[FAIL] 批量标记告警已读失败: {e}")
            return 0

    # ==================== 12. 清理资源 ====================
    def close(self):
        """关闭数据库连接（程序退出时调用）"""
        if self.conn:
            self.conn.close()
            logger.info("[PLUG] SQLite 数据库连接已关闭")
