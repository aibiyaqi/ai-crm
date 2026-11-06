"""
logger.py - 日志工具模块
====================================
功能分区：
  1. 日志记录器初始化（控制台 + 文件双输出）
  2. 日志级别控制（开发用 DEBUG，生产用 INFO）
  3. 日志格式定义（时间/级别/模块/消息）
  4. 日志文件轮转（按大小自动分割）
"""

# ==================== 1. 导入依赖 ====================
import logging           # Python 标准日志库
import os                # 操作系统接口，用于创建日志目录
from logging.handlers import (
    RotatingFileHandler  # 日志文件轮转处理器，自动按大小分割
)
from config import (
    LOG_FILE_PATH,       # 日志文件路径：logfile/app.log
    LOG_LEVEL,           # 日志级别：DEBUG 或 INFO
)


# ==================== 2. 日志记录器初始化函数 ====================
def setup_logger(name: str = "CRM") -> logging.Logger:
    """
    初始化并返回一个日志记录器
    ==========================
    参数：
      name : 日志记录器名称，默认为 "CRM"
    
    返回：
      配置好的 logging.Logger 实例
    
    日志输出目标：
      1. 控制台（终端）：显示日志消息
      2. 文件（logfile/app.log）：持久化保存
    
    日志格式：
      [2026-05-18 16:30:00] [INFO] [CRM] 这是一条日志消息
    """
    
    # ==================== 3. 创建日志记录器 ====================
    # getLogger 获取或创建指定名称的记录器
    logger = logging.getLogger(name)
    
    # 设置日志级别（低于此级别的日志将被忽略）
    # LOG_LEVEL 从 config.py 读取，开发=DEBUG，生产=INFO
    logger.setLevel(getattr(logging, LOG_LEVEL))
    
    # 防止重复添加处理器（多次调用 setup_logger 时）
    if logger.handlers:
        return logger
    
    # ==================== 4. 定义日志格式 ====================
    # 格式字符串说明：
    #   %(asctime)s   → 时间戳，如 2026-05-18 16:30:00
    #   %(levelname)s → 日志级别，如 INFO / ERROR
    #   %(name)s      → 日志记录器名称
    #   %(message)s   → 日志消息内容
    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"  # 时间格式：年-月-日 时:分:秒
    )
    
    # ==================== 5. 控制台处理器 ====================
    # StreamHandler 输出到终端（stderr）
    # 强制 stderr 使用 UTF-8 编码 + errors='replace' 兜底，
    # 避免 Windows GBK 环境下 emoji 等字符导致 UnicodeEncodeError 崩溃
    import io as _io
    try:
        # Python 3.7+ 支持 reconfigure，errors='replace' 确保无法编码的字符用 ? 代替而非抛异常
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)     # 终端显示所有级别
    console_handler.setFormatter(formatter)     # 应用格式
    
    # ==================== 6. 文件处理器 ====================
    # 先确保日志目录存在（如 logfile/）
    log_dir = os.path.dirname(LOG_FILE_PATH)    # 提取目录部分
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)     # 创建目录（不存在则创建）
    
    # RotatingFileHandler：日志文件轮转处理器
    #   参数说明：
    #     filename   → 日志文件路径
    #     maxBytes   → 单个文件最大字节数（1MB）
    #     backupCount → 保留的备份文件数（超过则删除最旧的）
    #
    # 多进程（uvicorn workers>1）注意：
    #   多个进程同时持有同一个 RotatingFileHandler 时，轮转（rename）会互相冲突，
    #   在 Windows 上表现为 PermissionError: [WinError 32] 文件被占用。
    #   解决方案：每个进程写入独立的日志文件（按 PID 区分），避免轮转竞争。
    #   单进程场景（workers=1）仍写原路径，保持向后兼容。
    _pid = os.getpid()
    _multi_worker = os.getenv("API_WORKERS", "1").strip() not in ("", "0", "1")
    if _multi_worker:
        _root, _ext = os.path.splitext(LOG_FILE_PATH)
        _file_path = f"{_root}.{_pid}{_ext or '.log'}"
    else:
        _file_path = LOG_FILE_PATH
    file_handler = RotatingFileHandler(
        filename=_file_path,        # 日志文件路径（多进程时按 PID 分文件）
        maxBytes=1 * 1024 * 1024,   # 每个日志文件最大 1MB
        backupCount=3,              # 保留最近 3 个备份文件
        encoding="utf-8"            # 编码格式（支持中文日志）
    )
    file_handler.setLevel(logging.DEBUG)        # 文件记录所有级别
    file_handler.setFormatter(formatter)        # 应用格式
    
    # ==================== 7. 注册处理器 ====================
    # 将两个处理器添加到日志记录器
    logger.addHandler(console_handler)    # 添加控制台处理器
    logger.addHandler(file_handler)       # 添加文件处理器
    
    return logger


# ==================== 8. 全局单例日志记录器 ====================
# 模块加载时自动初始化，其他模块直接导入使用
# 使用方式：from logger import logger
#           logger.info("消息")
#           logger.error("错误")
logger = setup_logger()
