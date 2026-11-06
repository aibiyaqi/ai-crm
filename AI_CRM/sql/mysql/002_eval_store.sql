-- ============================================================
-- 002_eval_store.sql — 评测库（MySQL 8.0+）
-- ============================================================
-- 对应原 eval_store.db 的 4 张表。
-- created_at 沿用 ISO 字符串（VARCHAR），与 eval_store._now() 保持一致，
-- 避免 "YYYY-MM-DDTHH:MM:SS" 与 DATETIME(6) 解析差异。
-- ============================================================

CREATE TABLE IF NOT EXISTS eval_dataset (
    id          INT          NOT NULL AUTO_INCREMENT COMMENT '自增主键',
    name        VARCHAR(128) NOT NULL COMMENT '数据集名',
    version     VARCHAR(32)  NOT NULL COMMENT '版本号',
    kind        VARCHAR(32)  NOT NULL COMMENT 'intent/retrieval/safety/e2e',
    note        TEXT         DEFAULT NULL COMMENT '备注',
    is_active   TINYINT(1)   NOT NULL DEFAULT 1 COMMENT '是否当前启用版本',
    created_at  VARCHAR(32)  NOT NULL COMMENT '创建时间（ISO 字符串）',
    PRIMARY KEY (id),
    UNIQUE KEY uk_eval_dataset_name_version (name, version)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='评测数据集版本';

CREATE TABLE IF NOT EXISTS eval_case (
    id          INT          NOT NULL AUTO_INCREMENT COMMENT '自增主键',
    dataset_id  INT          NOT NULL COMMENT '所属数据集',
    `query`     TEXT         NOT NULL COMMENT '用例问题',
    expected    MEDIUMTEXT   DEFAULT NULL COMMENT '期望值（JSON）',
    source      VARCHAR(16)  DEFAULT 'manual' COMMENT 'manual/online/synthetic',
    meta        MEDIUMTEXT   DEFAULT NULL COMMENT '其它标注（JSON）',
    created_at  VARCHAR(32)  NOT NULL COMMENT '创建时间',
    PRIMARY KEY (id),
    KEY idx_eval_case_dataset (dataset_id),
    CONSTRAINT fk_eval_case_dataset FOREIGN KEY (dataset_id)
        REFERENCES eval_dataset (id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='评测用例';

CREATE TABLE IF NOT EXISTS eval_badcase (
    id          INT          NOT NULL AUTO_INCREMENT COMMENT '自增主键',
    kind        VARCHAR(32)  NOT NULL COMMENT 'intent_miss/rag_empty/sql_reject/...',
    `query`     TEXT         NOT NULL COMMENT 'badcase 问题',
    expected    TEXT         DEFAULT NULL COMMENT '期望',
    actual      TEXT         DEFAULT NULL COMMENT '实际',
    source      VARCHAR(16)  DEFAULT 'online' COMMENT 'online/manual',
    status      VARCHAR(16)  NOT NULL DEFAULT 'open' COMMENT 'open/triaged/fixed/wontfix',
    note        TEXT         DEFAULT NULL COMMENT '备注',
    created_at  VARCHAR(32)  NOT NULL COMMENT '创建时间',
    updated_at  VARCHAR(32)  NOT NULL COMMENT '更新时间',
    PRIMARY KEY (id),
    KEY idx_eval_badcase_status (status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='badcase 库';

CREATE TABLE IF NOT EXISTS eval_run (
    id          INT          NOT NULL AUTO_INCREMENT COMMENT '自增主键',
    dataset_id  INT          DEFAULT NULL COMMENT '数据集ID',
    dataset     VARCHAR(255) DEFAULT '' COMMENT '冗余 name@version',
    metric      VARCHAR(64)  NOT NULL COMMENT '指标名',
    score       DOUBLE       NOT NULL COMMENT '得分',
    detail      MEDIUMTEXT   DEFAULT NULL COMMENT '分项细节（JSON）',
    created_at  VARCHAR(32)  NOT NULL COMMENT '创建时间',
    PRIMARY KEY (id),
    KEY idx_eval_run_dataset (dataset_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='评测运行记录';
