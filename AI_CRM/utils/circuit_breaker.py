"""
circuit_breaker.py - 熔断器 + 重试机制
========================================
功能：
  1. CircuitBreaker：熔断器模式，防止级联故障
  2. retry_with_backoff：指数退避重试，应对临时故障

使用方式：
  cb = CircuitBreaker(name="llm", failure_threshold=3, recovery_timeout=30)
  result = cb.call(my_function, arg1, arg2)

  # 或使用重试装饰器
  result = retry_with_backoff(my_function, arg1, max_retries=3, arg2=...)
"""

import time
import random
import logging
from enum import Enum
from typing import Callable, Any, Optional

logger = logging.getLogger("circuit_breaker")


class CircuitBreakerState(Enum):
    """熔断器状态"""
    CLOSED = "closed"        # 正常状态，请求放行
    OPEN = "open"            # 熔断状态，直接拒绝请求
    HALF_OPEN = "half_open"  # 半开状态，试探性放行一个请求


class CircuitBreakerOpenError(Exception):
    """熔断器打开异常 — 服务暂时不可用"""
    def __init__(self, name: str):
        self.service_name = name
        super().__init__(f"[STOP] 服务 '{name}' 暂时不可用（熔断器已打开）")


class CircuitBreaker:
    """
    熔断器
    =======
    防止对故障服务的重复调用，让服务有时间恢复。
    
    状态转换：
      CLOSED → 连续失败达到阈值 → OPEN
      OPEN → 超时恢复时间 → HALF_OPEN
      HALF_OPEN → 试探请求成功 → CLOSED
      HALF_OPEN → 试探请求失败 → OPEN
    
    参数：
      name              : 服务名称（日志用）
      failure_threshold : 连续失败次数阈值（达到后熔断）
      recovery_timeout  : 熔断后等待恢复的秒数
      half_open_max_retries : 半开状态最大试探次数（默认 1）
    """
    
    def __init__(
        self,
        name: str = "unknown",
        failure_threshold: int = 5,
        recovery_timeout: int = 30,
        half_open_max_retries: int = 1,
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_retries = half_open_max_retries
        
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.half_open_attempts = 0
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        通过熔断器调用函数
        ====================
        参数：
          func : 要调用的函数
          *args, **kwargs : 传递给函数的参数
        返回：
          函数执行结果
        抛出：
          CircuitBreakerOpenError — 熔断器打开时
          原始异常 — 函数执行失败时
        """
        # -------- 状态检查 --------
        if self.state == CircuitBreakerState.OPEN:
            # 检查是否到达恢复时间
            if self.last_failure_time and (
                time.time() - self.last_failure_time >= self.recovery_timeout
            ):
                logger.info(f"[UNLOCK] 熔断器 '{self.name}' 超时恢复，进入半开状态")
                self.state = CircuitBreakerState.HALF_OPEN
                self.half_open_attempts = 0
            else:
                raise CircuitBreakerOpenError(self.name)
        
        try:
            # -------- 执行调用 --------
            result = func(*args, **kwargs)
            
            # -------- 成功处理 --------
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.half_open_attempts += 1
                if self.half_open_attempts >= self.half_open_max_retries:
                    logger.info(f"[LOCK] 熔断器 '{self.name}' 恢复关闭状态")
                    self.state = CircuitBreakerState.CLOSED
                    self.failure_count = 0
            
            # CLOSED 状态下成功 → 重置计数
            if self.state == CircuitBreakerState.CLOSED:
                self.failure_count = 0
            
            return result
        
        except Exception as e:
            # -------- 失败处理 --------
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if (self.state == CircuitBreakerState.CLOSED and 
                self.failure_count >= self.failure_threshold):
                logger.warning(
                    f"[STOP] 熔断器 '{self.name}' 触发熔断！"
                    f"连续失败 {self.failure_count} 次，"
                    f"恢复等待 {self.recovery_timeout}s"
                )
                self.state = CircuitBreakerState.OPEN
            
            elif self.state == CircuitBreakerState.HALF_OPEN:
                logger.warning(
                    f"[STOP] 熔断器 '{self.name}' 半开状态试探失败，回到打开状态"
                )
                self.state = CircuitBreakerState.OPEN
            
            raise
    
    def reset(self):
        """手动重置熔断器"""
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        self.half_open_attempts = 0
        logger.info(f"[REFRESH] 熔断器 '{self.name}' 已手动重置")
    
    def __str__(self) -> str:
        return (
            f"CircuitBreaker('{self.name}', "
            f"state={self.state.value}, "
            f"failures={self.failure_count}/{self.failure_threshold})"
        )


def retry_with_backoff(
    func: Callable,
    *args,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 10.0,
    backoff_factor: float = 2.0,
    jitter: bool = True,
    **kwargs,
) -> Any:
    """
    指数退避重试
    ==============
    对临时故障（网络超时、服务暂时不可用）进行重试。
    使用指数退避 + 随机抖动，避免惊群效应。
    
    参数：
      func           : 要重试的函数
      max_retries    : 最大重试次数（默认 3）
      base_delay     : 初始等待秒数（默认 1s）
      max_delay      : 最大等待秒数（默认 10s）
      backoff_factor : 退避倍数（默认 2x）
      jitter         : 是否添加随机抖动（默认 True）
      *args, **kwargs: 传递给函数的参数
    
    返回：
      函数执行结果
    
    抛出：
      最后一次尝试的异常（所有重试均失败时）
    """
    last_exception = None
    
    for attempt in range(1 + max_retries):
        try:
            return func(*args, **kwargs)
        except CircuitBreakerOpenError:
            # 熔断器打开不重试（等待熔断恢复）
            raise
        except Exception as e:
            last_exception = e
            
            if attempt < max_retries:
                # 计算退避时间
                delay = min(base_delay * (backoff_factor ** attempt), max_delay)
                if jitter:
                    # 添加 ±25% 的随机抖动
                    delay *= 0.75 + random.random() * 0.5
                
                logger.warning(
                    f"[REFRESH] 重试 #{attempt + 1}/{max_retries} "
                    f"等待 {delay:.1f}s... "
                    f"错误: {e}"
                )
                time.sleep(delay)
            else:
                logger.error(
                    f"[FAIL] 重试 {max_retries} 次均失败: {e}"
                )
    
    raise last_exception  # type: ignore


# ==================== 预定义熔断器实例 ====================

# LLM 调用熔断器（连续 3 次失败熔断 30 秒）
llm_circuit_breaker = CircuitBreaker(
    name="LLM",
    failure_threshold=3,
    recovery_timeout=30,
)

# 搜索工具熔断器（连续 3 次失败熔断 20 秒）
search_circuit_breaker = CircuitBreaker(
    name="WebSearch",
    failure_threshold=3,
    recovery_timeout=20,
)

# RAG 检索熔断器（连续 5 次失败熔断 60 秒）
rag_circuit_breaker = CircuitBreaker(
    name="RAG",
    failure_threshold=5,
    recovery_timeout=60,
)
