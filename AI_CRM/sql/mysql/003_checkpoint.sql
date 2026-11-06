-- ============================================================
-- 003_checkpoint.sql — LangGraph 任务检查点（MySQL 8.0+）
-- ============================================================
-- 对应原 checkpoints.db 的 checkpoints 表。
-- 复合主键 (thread_id, checkpoint_id)：同一任务的多轮快照按 id 递增排序。
-- ============================================================

CREATE TABLE IF NOT EXISTS checkpoints (
    thread_id     VARCHAR(128) NOT NULL COMMENT '任务线程ID',
    checkpoint_id BIGINT       NOT NULL COMMENT '检查点ID（单调递增）',
    node          VARCHAR(64)  DEFAULT NULL COMMENT '当前节点',
    state_json    LONGTEXT     DEFAULT NULL COMMENT '状态快照（JSON）',
    created_at    DATETIME(6)     NOT NULL DEFAULT CURRENT_TIMESTAMP(6) COMMENT '写入时间',
    PRIMARY KEY (thread_id, checkpoint_id),
    KEY idx_cp_thread (thread_id, checkpoint_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='LangGraph 检查点';
