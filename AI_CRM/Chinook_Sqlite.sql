-- ============================================================
-- Chinook_Sqlite.sql - SQLite 数据库初始化脚本
-- ============================================================
-- 功能：
--   1. 创建基础表结构（users / messages / customer_profiles）
--   2. 插入演示数据（方便测试）
--   3. 创建索引（提升查询性能）
--
-- 使用方式：
--   sqlite3 crm.db < Chinook_Sqlite.sql
--   或在 Python 中由 memory_sqlite.py 自动建表（推荐）
-- ============================================================

-- ==================== 1. 用户表 ====================
CREATE TABLE IF NOT EXISTS users (
    user_id    TEXT PRIMARY KEY,           -- 用户唯一标识
    name       TEXT NOT NULL,              -- 用户姓名
    phone      TEXT,                       -- 手机号
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,  -- 创建时间
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP   -- 更新时间
);

-- ==================== 2. 消息表 ====================
CREATE TABLE IF NOT EXISTS messages (
    message_id TEXT PRIMARY KEY,           -- 消息唯一标识
    user_id    TEXT NOT NULL,              -- 用户ID（外键关联）
    role       TEXT NOT NULL,              -- 角色：user/assistant/system
    content    TEXT NOT NULL,              -- 消息内容
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,  -- 发送时间
    FOREIGN KEY (user_id) REFERENCES users(user_id)  -- 外键约束
);

-- ==================== 3. 客户画像表 ====================
-- 轻量化 CRM 核心：仅保留必要字段
-- 待优化：线索评分 / 360°视图 / 审计日志
CREATE TABLE IF NOT EXISTS customer_profiles (
    customer_id TEXT PRIMARY KEY,          -- 客户唯一标识
    name        TEXT NOT NULL,             -- 客户姓名
    company     TEXT,                      -- 公司
    position    TEXT,                      -- 职位
    phone       TEXT,                      -- 电话
    email       TEXT,                      -- 邮箱
    notes       TEXT,                      -- 备注
    tags        TEXT DEFAULT '[]',         -- 标签（JSON格式）
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ==================== 4. 演示数据 ====================
-- 插入测试用户
INSERT OR IGNORE INTO users (user_id, name, phone) VALUES
    ('u001', '张三', '13800138001'),
    ('u002', '李四', '13800138002'),
    ('u003', '王五', '13800138003');

-- 插入测试客户画像（方便演示CRM功能）
INSERT OR IGNORE INTO customer_profiles (customer_id, name, company, position, phone, email, notes, tags) VALUES
    ('c001', '阿里巴巴', '阿里巴巴集团', '技术总监', '13900139001', 'cto@alibaba.com', '高价值客户，有长期合作意向', '["高价值", "意向客户", "科技"]'),
    ('c002', '腾讯科技', '腾讯计算机系统有限公司', '产品经理', '13900139002', 'pm@tencent.com', '咨询 AI 客服解决方案', '["意向客户", "互联网"]'),
    ('c003', '华为技术', '华为技术有限公司', '副总裁', '13900139003', 'vp@huawei.com', '已签约客户，正在实施阶段', '["已签约", "高价值", "通信"]'),
    ('c004', '字节跳动', '字节跳动有限公司', '运营总监', '13900139004', 'ops@bytedance.com', '初步接触，待跟进', '["潜在客户", "互联网"]'),
    ('c005', '小米科技', '小米科技有限公司', '市场经理', '13900139005', 'mkt@xiaomi.com', '对智能客服感兴趣，已发方案', '["意向客户", "科技"]');

-- ==================== 5. 索引（提升性能） ====================
CREATE INDEX IF NOT EXISTS idx_messages_user_id ON messages(user_id);
CREATE INDEX IF NOT EXISTS idx_messages_created_at ON messages(created_at);
CREATE INDEX IF NOT EXISTS idx_customers_updated_at ON customer_profiles(updated_at);

-- ==================== 6. 验证 ====================
-- 运行后检查数据：
-- SELECT COUNT(*) FROM users;
-- SELECT COUNT(*) FROM customer_profiles;
-- SELECT name, company, tags FROM customer_profiles;
