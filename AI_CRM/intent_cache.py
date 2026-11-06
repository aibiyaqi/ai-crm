# -*- coding: utf-8 -*-
"""
intent_cache.py — 意图识别缓存（Redis 优先，LRU 兜底）
======================================================
背景（为什么要从 LRU 改成 Redis）
--------------------------------
原 agent.py 用进程内 OrderedDict 做 LRU 缓存（容量 200），存在三个问题：
  1. **多进程不共享**：uvicorn --workers N 或多进程部署时，每个进程各持一份缓存，
     命中率被稀释（N 份缓存各自冷启动），高并发下等于「缓存形同虚设」。
  2. **容量受限**：固定 200 条，高并发下频繁淘汰，热点 query 反复回源调 LLM。
  3. **重启即失效**：进程重启缓存全丢，冷启动期 LLM 调用量陡增。

改造方案
--------
抽象出 IntentCache 门面，**Redis 作为共享缓存**（多进程 / 多实例共享同一份），
Redis 不可用时**自动降级为进程内 LRU**，绝不阻塞主链路（高可用设计）。

P4 之后的调整：
  连接/熔断/超时/观测统一收敛到 redis_client.RedisManager，
  本模块只保留「业务语义（键 = md5(query)，值 = 意图标签）+ LRU 兜底」。

Redis 键设计
------------
  key   = crm:intent:{md5(query)}      —— 用 MD5 压缩键长，避免超长 query 撑爆键
  value = 意图标签（web_search / rag_retrieve / direct_llm）
  TTL   = INTENT_CACHE_TTL（默认 7 天）—— 意图分类结果稳定，可长缓存

降级策略（关键）
----------------
  · Redis 不可用（未安装/连不上/熔断中）→ 走进程内 LRU；
  · 运行中读/写 Redis 失败 → RedisManager 吞掉异常并计数，本模块回落 LRU；
  · 任何情况下都不向上抛异常，保证意图识别主链路不中断。

用法
----
  from intent_cache import IntentCache
  cache = IntentCache()
  hit = cache.get(query)          # 命中返回意图标签，否则 None
  if hit is None:
      intent = call_llm(query)
      cache.set(query, intent)

自检：
  python intent_cache.py
"""

from __future__ import annotations

import hashlib
import os
import threading
from collections import OrderedDict
from typing import Optional

from logger import logger
from redis_client import RedisManager, get_redis


# ==================== 1. 配置（环境变量可覆盖） ====================
# Redis 连接串（默认本机 6379；docker-compose 已提供 redis 服务）
REDIS_URL: str = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")
# 缓存 TTL（秒）：默认 7 天。意图分类结果稳定，长 TTL 可显著降低 LLM 调用
INTENT_CACHE_TTL: int = int(os.getenv("INTENT_CACHE_TTL", str(7 * 24 * 3600)))
# LRU 兜底容量（Redis 不可用时生效）
INTENT_CACHE_MAXSIZE: int = int(os.getenv("INTENT_CACHE_MAXSIZE", "200"))
# 键前缀（便于在 Redis 里按业务域隔离 / 批量清理）
KEY_PREFIX: str = os.getenv("INTENT_CACHE_PREFIX", "crm:intent:")


def _make_key(query: str) -> str:
    """query → 缓存键（MD5 压缩，避免超长键）"""
    digest = hashlib.md5(query.encode("utf-8")).hexdigest()
    return KEY_PREFIX + digest


# ==================== 2. 缓存门面 ====================
class IntentCache:
    """
    意图缓存门面：Redis 优先，失败自动降级进程内 LRU
    ==================================================
    线程安全：
      · Redis 客户端本身线程安全（连接池）；
      · LRU 兜底用 threading.Lock 保护 OrderedDict 的读写与淘汰。
    """

    def __init__(self, redis_url: Optional[str] = None):
        # 默认复用全局 Redis 单例；显式传 url 时独立建一个（便于自测/多租户）
        if redis_url and redis_url != REDIS_URL:
            self._redis: Optional[RedisManager] = RedisManager(url=redis_url)
        else:
            self._redis = get_redis()
        self._lru: "OrderedDict[str, str]" = OrderedDict()
        self._lock = threading.Lock()
        # 统计（近似值，仅作展示）
        self.hits = 0
        self.misses = 0
        self.redis_errors = 0

    @property
    def backend(self) -> str:
        """redis / lru（供 /api 观测与日志）"""
        return "redis" if (self._redis and self._redis.available) else "lru"

    # -------- 2.1 读 --------
    def get(self, query: str) -> Optional[str]:
        """
        读缓存：命中返回意图标签，未命中返回 None
        ==========================================
        Redis 不可用 / 读失败 → 本次降级 LRU（不抛异常）
        """
        key = _make_key(query)

        if self._redis is not None and self._redis.available:
            val = self._redis.get(key)
            if val:
                self.hits += 1
                logger.info(f"📍 [CACHE] Redis 命中 → {val}")
                return val
            self.misses += 1
            self.redis_errors = self._redis.errors
            return None

        # ---- LRU 兜底 ----
        with self._lock:
            if key in self._lru:
                self._lru.move_to_end(key)  # LRU：访问后移到末尾
                self.hits += 1
                logger.info(f"📍 [CACHE] LRU 命中 → {self._lru[key]}")
                return self._lru[key]
        self.misses += 1
        return None

    # -------- 2.2 写 --------
    def set(self, query: str, intent: str) -> None:
        """
        写缓存（Redis 优先，失败降级 LRU）
        ====================================
        """
        key = _make_key(query)

        if self._redis is not None and self._redis.available:
            if self._redis.set(key, intent, ttl=INTENT_CACHE_TTL):
                return
            self.redis_errors = self._redis.errors

        # ---- LRU 兜底（带容量淘汰） ----
        with self._lock:
            self._lru[key] = intent
            if len(self._lru) > INTENT_CACHE_MAXSIZE:
                self._lru.popitem(last=False)  # 淘汰最旧项

    # -------- 2.3 观测 --------
    def stats(self) -> dict:
        """返回缓存运行状态快照（供接口 / 日志观察）"""
        total = self.hits + self.misses
        info = {
            "backend": self.backend,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 4) if total else 0.0,
            "redis_errors": self.redis_errors,
            "ttl_seconds": INTENT_CACHE_TTL,
        }
        if self._redis is not None:
            info["redis"] = self._redis.stats()
            if self._redis.available:
                info["redis_keys"] = self._redis.run("dbsize", lambda c: c.dbsize())
        if self.backend != "redis":
            info["lru_size"] = len(self._lru)
            info["lru_maxsize"] = INTENT_CACHE_MAXSIZE
        return info


# ==================== 3. 自检 ====================
if __name__ == "__main__":
    print("=" * 60)
    print("intent_cache.py 自检")
    print("=" * 60)
    print(f"REDIS_URL          : {REDIS_URL}")
    print(f"INTENT_CACHE_TTL   : {INTENT_CACHE_TTL}s")
    print(f"INTENT_CACHE_MAXSIZE: {INTENT_CACHE_MAXSIZE}")
    print("-" * 60)

    cache = IntentCache()
    print(f"实际后端           : {cache.backend}")

    q = "初三数学一对一课时费大概多少？"
    print(f"\n[1] 首次读（应 miss）: {cache.get(q)}")
    cache.set(q, "rag_retrieve")
    print(f"[2] 写入后再读（应 hit）: {cache.get(q)}")
    print(f"[3] 统计: {cache.stats()}")
    print("=" * 60)
    print("[OK] intent_cache.py 自检完成")
