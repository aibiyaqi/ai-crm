-- ============================================================
-- 001_init.sql — AI CRM 业务库（MySQL 8.0+）
-- ============================================================
-- 对应原 SQLite 业务库 crm.db 的 12 张表。
-- 类型映射：TEXT 主键/唯一键 → VARCHAR；INTEGER 布尔 → TINYINT(1)；
--           TIMESTAMP → DATETIME（与应用侧 datetime.now() 一致，无时区换算）。
-- 索引一律内联在建表语句里：MySQL 不支持给索引加 IF NOT EXISTS 守卫。
-- ============================================================

-- 1. 账户认证表
CREATE TABLE IF NOT EXISTS accounts (
    account_id    VARCHAR(64)  NOT NULL COMMENT '账户唯一标识（UUID）',
    username      VARCHAR(64)  NOT NULL COMMENT '用户名（登录用，唯一）',
    password_hash VARCHAR(255) NOT NULL COMMENT '密码哈希值',
    role          VARCHAR(16)  NOT NULL DEFAULT 'user' COMMENT '角色：admin/user',
    phone         VARCHAR(32)  NOT NULL COMMENT '手机号（唯一）',
    email         VARCHAR(255) DEFAULT NULL COMMENT '邮箱',
    is_active     TINYINT(1)   NOT NULL DEFAULT 1 COMMENT '是否激活：1=激活 0=禁用',
    last_login    DATETIME(6)     DEFAULT NULL COMMENT '最后登录时间',
    created_at    DATETIME(6)     NOT NULL DEFAULT CURRENT_TIMESTAMP(6) COMMENT '创建时间',
    updated_at    DATETIME(6)     NOT NULL DEFAULT CURRENT_TIMESTAMP(6) COMMENT '更新时间',
    PRIMARY KEY (account_id),
    UNIQUE KEY uk_accounts_username (username),
    UNIQUE KEY uk_accounts_phone (phone)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='账户认证表';

-- 2. 密码重置令牌表
CREATE TABLE IF NOT EXISTS password_reset_tokens (
    token_id    VARCHAR(64)  NOT NULL COMMENT '令牌唯一标识',
    account_id  VARCHAR(64)  NOT NULL COMMENT '关联账户ID',
    token       VARCHAR(128) NOT NULL COMMENT '重置令牌',
    code        VARCHAR(16)  NOT NULL COMMENT '短信验证码',
    expires_at  DATETIME(6)     NOT NULL COMMENT '过期时间',
    used        TINYINT(1)   NOT NULL DEFAULT 0 COMMENT '是否已使用',
    created_at  DATETIME(6)     NOT NULL DEFAULT CURRENT_TIMESTAMP(6) COMMENT '创建时间',
    PRIMARY KEY (token_id),
    KEY idx_prt_account (account_id),
    CONSTRAINT fk_prt_account FOREIGN KEY (account_id) REFERENCES accounts (account_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='密码重置令牌表';

-- 3. 登录日志表
CREATE TABLE IF NOT EXISTS login_logs (
    log_id      VARCHAR(64)  NOT NULL COMMENT '日志唯一标识',
    account_id  VARCHAR(64)  DEFAULT NULL COMMENT '关联账户ID（失败时可空）',
    username    VARCHAR(64)  NOT NULL COMMENT '尝试登录的用户名',
    action      VARCHAR(16)  NOT NULL COMMENT 'login/logout/fail',
    ip_address  VARCHAR(64)  DEFAULT NULL COMMENT '登录IP',
    detail      TEXT         DEFAULT NULL COMMENT '详细信息',
    created_at  DATETIME(6)     NOT NULL DEFAULT CURRENT_TIMESTAMP(6) COMMENT '操作时间',
    PRIMARY KEY (log_id),
    KEY idx_login_logs_user_time (username, created_at),
    KEY idx_login_logs_account (account_id),
    CONSTRAINT fk_login_logs_account FOREIGN KEY (account_id) REFERENCES accounts (account_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='登录日志表';

-- 4. 会话令牌表（跨刷新登录保持）
CREATE TABLE IF NOT EXISTS session_tokens (
    token       VARCHAR(128) NOT NULL COMMENT '会话令牌',
    account_id  VARCHAR(64)  NOT NULL COMMENT '关联账户ID',
    created_at  DATETIME(6)     NOT NULL DEFAULT CURRENT_TIMESTAMP(6) COMMENT '创建时间',
    expires_at  DATETIME(6)     DEFAULT NULL COMMENT '过期时间',
    PRIMARY KEY (token),
    KEY idx_session_tokens_account (account_id),
    CONSTRAINT fk_session_tokens_account FOREIGN KEY (account_id) REFERENCES accounts (account_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='会话令牌表';

-- 5. 对话会话表（多对话框）
CREATE TABLE IF NOT EXISTS sessions (
    session_id  VARCHAR(64) NOT NULL COMMENT '会话唯一标识',
    account_id  VARCHAR(64) NOT NULL COMMENT '所属账户ID',
    title       VARCHAR(255) NOT NULL DEFAULT '新对话' COMMENT '会话标题',
    created_at  DATETIME(6)    NOT NULL DEFAULT CURRENT_TIMESTAMP(6) COMMENT '创建时间',
    updated_at  DATETIME(6)    NOT NULL DEFAULT CURRENT_TIMESTAMP(6) COMMENT '最后活跃时间',
    PRIMARY KEY (session_id),
    KEY idx_sessions_account (account_id, updated_at),
    CONSTRAINT fk_sessions_account FOREIGN KEY (account_id) REFERENCES accounts (account_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='对话会话表';

-- 6. 用户信息表
CREATE TABLE IF NOT EXISTS users (
    user_id     VARCHAR(64) NOT NULL COMMENT '用户唯一标识',
    name        VARCHAR(128) NOT NULL COMMENT '用户姓名',
    phone       VARCHAR(32) DEFAULT NULL COMMENT '手机号',
    created_at  DATETIME(6)    NOT NULL DEFAULT CURRENT_TIMESTAMP(6) COMMENT '创建时间',
    updated_at  DATETIME(6)    NOT NULL DEFAULT CURRENT_TIMESTAMP(6) COMMENT '更新时间',
    PRIMARY KEY (user_id),
    KEY idx_users_phone (phone)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='用户信息表';

-- 7. 对话历史表
CREATE TABLE IF NOT EXISTS messages (
    message_id  VARCHAR(64) NOT NULL COMMENT '消息唯一标识',
    session_id  VARCHAR(64) NOT NULL COMMENT '所属会话ID',
    user_id     VARCHAR(64) NOT NULL COMMENT '发送用户ID',
    role        VARCHAR(16) NOT NULL COMMENT 'user/assistant/system',
    content     MEDIUMTEXT  NOT NULL COMMENT '消息内容',
    created_at  DATETIME(6)    NOT NULL DEFAULT CURRENT_TIMESTAMP(6) COMMENT '发送时间',
    PRIMARY KEY (message_id),
    KEY idx_messages_session_time (session_id, created_at),
    CONSTRAINT fk_messages_session FOREIGN KEY (session_id) REFERENCES sessions (session_id),
    CONSTRAINT fk_messages_user FOREIGN KEY (user_id) REFERENCES users (user_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='对话历史表';

-- 8. 客户画像表（CRM 核心 · K12 教育版）
CREATE TABLE IF NOT EXISTS customer_profiles (
    customer_id     VARCHAR(64)  NOT NULL COMMENT '客户唯一标识',
    name            VARCHAR(128) NOT NULL COMMENT '客户姓名',
    company         VARCHAR(255) DEFAULT NULL COMMENT '公司（通用场景兼容）',
    position        VARCHAR(128) DEFAULT NULL COMMENT '职位（通用场景兼容）',
    phone           VARCHAR(32)  DEFAULT NULL COMMENT '电话',
    email           VARCHAR(255) DEFAULT NULL COMMENT '邮箱',
    notes           MEDIUMTEXT   DEFAULT NULL COMMENT '备注',
    tags            TEXT         DEFAULT NULL COMMENT '标签列表（JSON）',
    created_at      DATETIME(6)     NOT NULL DEFAULT CURRENT_TIMESTAMP(6) COMMENT '创建时间',
    updated_at      DATETIME(6)     NOT NULL DEFAULT CURRENT_TIMESTAMP(6) COMMENT '更新时间',
    age             INT          DEFAULT NULL COMMENT '年龄',
    identity        VARCHAR(32)  DEFAULT NULL COMMENT '身份：学生/家长/成人',
    school_grade    VARCHAR(64)  DEFAULT NULL COMMENT '年级：初二/高三/在职',
    intent_course   VARCHAR(255) DEFAULT NULL COMMENT '意向课程',
    matched_course  VARCHAR(255) DEFAULT NULL COMMENT '适配课程',
    intent_price    VARCHAR(64)  DEFAULT NULL COMMENT '意向价格/预算（高风险）',
    hourly_fee      VARCHAR(64)  DEFAULT NULL COMMENT '课时费（高风险）',
    active_status   VARCHAR(32)  DEFAULT NULL COMMENT '活跃状态',
    advisor_id      VARCHAR(64)  DEFAULT NULL COMMENT '归属课程顾问ID',
    source          VARCHAR(64)  DEFAULT NULL COMMENT '客户来源渠道',
    last_follow_up  DATETIME(6)     DEFAULT NULL COMMENT '最近跟进时间',
    pool_type       VARCHAR(16)  DEFAULT '私海' COMMENT '池子类型：私海/公海',
    is_sleeping     TINYINT(1)   DEFAULT 0 COMMENT '沉睡/流失标记',
    exit_schedule_at DATETIME(6)    DEFAULT NULL COMMENT '退出调度时间',
    PRIMARY KEY (customer_id),
    KEY idx_customer_advisor (advisor_id),
    KEY idx_customer_phone (phone),
    KEY idx_customer_pool (pool_type, last_follow_up)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='客户画像表';

-- 9. 跟进记录表（B.4 跟进闭环）
CREATE TABLE IF NOT EXISTS follow_ups (
    follow_up_id      VARCHAR(64) NOT NULL COMMENT '跟进记录唯一标识',
    customer_id       VARCHAR(64) NOT NULL COMMENT '关联客户ID',
    advisor_id        VARCHAR(64) DEFAULT NULL COMMENT '跟进人',
    channel           VARCHAR(16) DEFAULT NULL COMMENT '电话/微信/面谈/短信/邮件',
    content           TEXT        DEFAULT NULL COMMENT '跟进内容摘要',
    result            VARCHAR(16) DEFAULT NULL COMMENT '已联系/未接/有意向/已成交/已放弃',
    next_follow_up_at DATETIME(6)    DEFAULT NULL COMMENT '计划下次跟进时间',
    created_at        DATETIME(6)    NOT NULL DEFAULT CURRENT_TIMESTAMP(6) COMMENT '本次跟进时间',
    PRIMARY KEY (follow_up_id),
    KEY idx_follow_ups_customer (customer_id),
    KEY idx_follow_ups_next (next_follow_up_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='跟进记录表';

-- 10. 主动告警表
CREATE TABLE IF NOT EXISTS alerts (
    alert_id      VARCHAR(64)  NOT NULL COMMENT '告警唯一标识',
    dedup_key     VARCHAR(255) DEFAULT NULL COMMENT '去重键：客户+类型+业务日',
    customer_id   VARCHAR(64)  DEFAULT NULL COMMENT '关联客户ID',
    customer_name VARCHAR(128) DEFAULT NULL COMMENT '客户姓名（冗余）',
    advisor_id    VARCHAR(64)  DEFAULT NULL COMMENT '归属顾问',
    alert_type    VARCHAR(32)  DEFAULT NULL COMMENT '告警类型',
    severity      VARCHAR(16)  DEFAULT NULL COMMENT 'high/medium/low',
    reasons       TEXT         DEFAULT NULL COMMENT '命中原因（JSON 数组）',
    source        VARCHAR(16)  DEFAULT NULL COMMENT 'scheduler/manual',
    status        VARCHAR(16)  DEFAULT 'unread' COMMENT 'unread/read',
    created_at    DATETIME(6)     NOT NULL DEFAULT CURRENT_TIMESTAMP(6) COMMENT '产生时间',
    read_at       DATETIME(6)     DEFAULT NULL COMMENT '已读时间',
    PRIMARY KEY (alert_id),
    UNIQUE KEY uk_alerts_dedup (dedup_key),
    KEY idx_alerts_status (status),
    KEY idx_alerts_advisor (advisor_id),
    KEY idx_alerts_created (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='主动告警表';

-- 11. 今日待办池（分层跑批）
CREATE TABLE IF NOT EXISTS pending_todo (
    todo_id     VARCHAR(64) NOT NULL COMMENT '待办唯一标识',
    customer_id VARCHAR(64) NOT NULL COMMENT '关联客户ID',
    advisor_id  VARCHAR(64) DEFAULT NULL COMMENT '归属顾问',
    source      VARCHAR(16) DEFAULT NULL COMMENT 'event/coarse',
    trigger_at  DATETIME(6)    DEFAULT NULL COMMENT '事件触发时间',
    biz_date    VARCHAR(10) DEFAULT NULL COMMENT '业务日 YYYY-MM-DD',
    status      VARCHAR(16) DEFAULT 'pending' COMMENT 'pending/done/expired',
    created_at  DATETIME(6)    NOT NULL DEFAULT CURRENT_TIMESTAMP(6) COMMENT '创建时间',
    PRIMARY KEY (todo_id),
    KEY idx_todo_bizdate (biz_date, status),
    KEY idx_todo_customer (customer_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='今日待办池';

-- 12. 延迟消息表（事件驱动）
CREATE TABLE IF NOT EXISTS delayed_messages (
    msg_id      VARCHAR(64) NOT NULL COMMENT '消息唯一标识',
    customer_id VARCHAR(64) NOT NULL COMMENT '关联客户ID',
    event_type  VARCHAR(32) DEFAULT NULL COMMENT '事件类型',
    fire_at     DATETIME(6)    DEFAULT NULL COMMENT '触发时间',
    status      VARCHAR(16) DEFAULT 'scheduled' COMMENT 'scheduled/cancelled/fired',
    created_at  DATETIME(6)    NOT NULL DEFAULT CURRENT_TIMESTAMP(6) COMMENT '创建时间',
    PRIMARY KEY (msg_id),
    KEY idx_delay_fire (fire_at, status),
    KEY idx_delay_customer (customer_id, status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='延迟消息表';
