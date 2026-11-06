"""
redis_client.py - Redis 统一基础设施（连接池 / 熔断 / 观测 / 键规范）
=====================================================================
定位（贯穿全项目的一条原则）：
    MySQL 是事实源，Redis 是加速层与协同层。
    Redis 不可用 → 功能降级、不阻断主链路、不丢数据。

提供的能力：
  1. 单例连接池（进程内共享，多线程安全）
  2. 统一键前缀 crm:{域}:{业务}:{标识}
  3. 统一降级入口 run()：Redis 异常只记数并返回 default，绝不向上抛
  4. 熔断：连续失败 N 次后冷却 M 秒直接走降级，避免每次请求都等超时
  5. 观测：ops / errors / fallbacks / 健康检查（供 /api/status、/api/metrics）
  6. 分布式锁：SET NX PX + Lua 比对释放（防误删他人锁）

用法：
    from redis_client import get_redis
    rd = get_redis()
    rd.set("session:abc", account_id, ttl=3600)
    val = rd.run("get", lambda c: c.get("k"), default=None)
"""

from __future__ import annotations

import threading
import time
import uuid
from typing import Any, Callable, Optional

from config import (
    REDIS_ENABLED,
    REDIS_URL,
    REDIS_PASSWORD,
    REDIS_DB,
    REDIS_MAX_CONNECTIONS,
    REDIS_TIMEOUT_S,
    REDIS_BREAKER_THRESHOLD,
    REDIS_BREAKER_COOLDOWN_S,
    REDIS_KEY_PREFIX,
)
from logger import logger

try:  # 指标埋点（可选）
    from metrics import incr as _metric_incr
except Exception:  # pragma: no cover
    def _metric_incr(*args, **kwargs):  # type: ignore[misc]
        return None


_RELEASE_LUA = (
    "if redis.call('get', KEYS[1]) == ARGV[1] then "
    "return redis.call('del', KEYS[1]) else return 0 end"
)


class RedisManager:
    """
    Redis 门面（单例）
    ==================
    线程安全：连接池本身线程安全；熔断计数用锁保护。
    """

    def __init__(
        self,
        url: str = REDIS_URL,
        password: str = REDIS_PASSWORD,
        db: int = REDIS_DB,
        enabled: bool = REDIS_ENABLED,
        auto_connect: bool = True,
    ):
        self.url = url
        self.password = password
        self.db = db
        self.enabled = enabled
        self.backend = "disabled" if not enabled else "down"
        self._client = None
        self._lock = threading.Lock()
        self._fails = 0
        self._open_until = 0.0
        self.ops = 0
        self.errors = 0
        self.fallbacks = 0
        self._last_error = ""
        self._next_reconnect = 0.0
        if enabled and auto_connect:
            self._connect()

    # -------- 1. 连接 --------
    def _connect(self) -> None:
        try:
            import redis  # 延迟导入：未安装 redis 包时整体降级
        except Exception as exc:  # pragma: no cover
            self.backend = "no-package"
            logger.warning(f"[CACHE] 未安装 redis 包，Redis 能力整体降级：{exc}")
            return
        try:
            pool = redis.ConnectionPool.from_url(
                self.url,
                password=self.password or None,
                db=self.db,
                decode_responses=True,
                max_connections=REDIS_MAX_CONNECTIONS,
                socket_timeout=REDIS_TIMEOUT_S,
                socket_connect_timeout=REDIS_TIMEOUT_S,
                health_check_interval=30,
            )
            client = redis.Redis(connection_pool=pool)
            client.ping()
            self._client = client
            self.backend = "redis"
            logger.info(f"[CACHE] Redis 已连接: {self._sanitize(self.url)} db={self.db}")
        except Exception as exc:
            self._client = None
            self.backend = "down"
            logger.warning(f"[CACHE] Redis 不可用，全部能力降级：{exc}")

    @staticmethod
    def _sanitize(url: str) -> str:
        """日志里隐藏密码（redis://:pass@host）"""
        if "@" in url and "://" in url:
            head, tail = url.split("://", 1)
            return f"{head}://***@{tail.split('@', 1)[1]}"
        return url

    # -------- 2. 可用性与熔断 --------
    @property
    def available(self) -> bool:
        return self._client is not None and time.monotonic() >= self._open_until

    def _on_ok(self) -> None:
        with self._lock:
            if self._fails or self._open_until:
                logger.info("[CACHE] Redis 已恢复，熔断关闭")
            self._fails = 0
            self._open_until = 0.0

    def _on_err(self, op: str, exc: Exception) -> None:
        with self._lock:
            self.errors += 1
            self._last_error = str(exc)
            self._fails += 1
            if self._fails >= REDIS_BREAKER_THRESHOLD and not self._open_until:
                self._open_until = time.monotonic() + REDIS_BREAKER_COOLDOWN_S
                logger.warning(
                    f"[CACHE] Redis 连续失败 {self._fails} 次，熔断 "
                    f"{REDIS_BREAKER_COOLDOWN_S}s（期间直接走降级）: {exc}"
                )
        _metric_incr("redis_op_total", op=op, status="error")

    # -------- 3. 统一调用入口 --------
    def run(self, op: str, fn: Callable[[Any], Any], default: Any = None) -> Any:
        """
        执行一次 Redis 操作（失败返回 default，不抛异常）
        ===============================================
        op : 操作名（用于日志/指标标签，如 get/set/zadd/lock）
        fn : 接收 client 的闭包，返回操作结果
        """
        self.ops += 1
        if not self.enabled:
            return default
        # 启动时没连上：每 30s 尝试重连一次（Redis 后起也能自动接上）
        if self._client is None and time.monotonic() >= self._next_reconnect:
            self._next_reconnect = time.monotonic() + 30
            self._connect()
        if not self.available:
            self.fallbacks += 1
            _metric_incr("redis_op_total", op=op, status="fallback")
            return default
        try:
            value = fn(self._client)
            self._on_ok()
            _metric_incr("redis_op_total", op=op, status="ok")
            return value
        except Exception as exc:  # noqa: BLE001 - 降级是设计目标
            self._on_err(op, exc)
            return default

    # -------- 4. 常用命令（薄封装，保持调用点简洁） --------
    def key(self, *parts: str) -> str:
        """拼键：crm:{域}:{业务}:{标识}"""
        return REDIS_KEY_PREFIX + ":".join(str(p) for p in parts)

    def get(self, key: str) -> Optional[str]:
        return self.run("get", lambda c: c.get(key))

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        if ttl:
            return bool(self.run("setex", lambda c: c.setex(key, int(ttl), value), False))
        return bool(self.run("set", lambda c: c.set(key, value), False))

    def set_nx(self, key: str, value: Any = "1", ttl: Optional[int] = None) -> bool:
        """SET NX（用于去重/抢锁）；成功返回 True"""
        if ttl:
            ok = self.run(
                "set_nx_ex",
                lambda c: c.set(key, value, nx=True, ex=int(ttl)),
                None,
            )
        else:
            ok = self.run("set_nx", lambda c: c.set(key, value, nx=True), None)
        return bool(ok)

    def try_set_nx(self, key: str, value: Any = "1", ttl: Optional[int] = None) -> Optional[bool]:
        """
        三态 SET NX（去重专用）
        ======================
        返回 True=本次抢到（首次出现）/ False=键已存在（重复）/
             None=Redis 不可用（调用方必须回退到数据库唯一键兜底）
        """
        if not self.enabled or not self.available:
            return None

        def _do(c):
            if ttl:
                return c.set(key, value, nx=True, ex=int(ttl))
            return c.set(key, value, nx=True)

        result = self.run("set_nx_tri", _do, "unavailable")
        if result == "unavailable":
            return None
        return bool(result)

    def delete(self, *keys: str) -> int:
        if not keys:
            return 0
        return int(self.run("del", lambda c: c.delete(*keys), 0) or 0)

    def exists(self, key: str) -> bool:
        return bool(self.run("exists", lambda c: c.exists(key), 0))

    def incr(self, key: str, ttl: Optional[int] = None) -> Optional[int]:
        def _do(c):
            pipe = c.pipeline()
            pipe.incr(key)
            if ttl:
                pipe.expire(key, int(ttl))
            return pipe.execute()[0]

        val = self.run("incr", _do)
        return int(val) if val is not None else None

    def expire(self, key: str, ttl: int) -> bool:
        return bool(self.run("expire", lambda c: c.expire(key, int(ttl)), False))

    # -------- 5. 分布式锁 --------
    def acquire_lock(self, name: str, ttl_seconds: int, token: Optional[str] = None) -> Optional[str]:
        """
        抢锁（SET NX EX）。返回 token 表示抢到；None 表示没抢到或 Redis 不可用。
        注意：Redis 不可用时返回 None，调用方应回退为"单实例模式"（照常执行）。
        """
        token = token or uuid.uuid4().hex
        ok = self.run(
            "lock_acquire",
            lambda c: c.set(self.key("lock", name), token, nx=True, ex=int(ttl_seconds)),
            None,
        )
        return token if ok else None

    def release_lock(self, name: str, token: str) -> bool:
        """
        释放锁（Lua 比对 token，避免误删别人的锁）
        ==========================================
        兼容性：极少数环境（如 fakeredis 未装 lupa）不支持 EVAL，
                此时退化为「GET 比对 + DEL」的非原子兜底（误删窗口极小）。
        """
        if not token:
            return False
        key = self.key("lock", name)
        result = self.run(
            "lock_release",
            lambda c: c.eval(_RELEASE_LUA, 1, key, token),
            None,
        )
        if result is None:
            if self.get(key) == token:
                self.delete(key)
                return True
            return False
        return bool(result)

    # -------- 6. 有序集合（延迟队列） --------
    def zadd(self, key: str, mapping: dict) -> int:
        return int(self.run("zadd", lambda c: c.zadd(key, mapping), 0) or 0)

    def zrem(self, key: str, *members: str) -> int:
        if not members:
            return 0
        return int(self.run("zrem", lambda c: c.zrem(key, *members), 0) or 0)

    def zrange_by_score(self, key: str, max_score: float, limit: int = 500) -> list:
        return list(
            self.run(
                "zrangebyscore",
                lambda c: c.zrangebyscore(key, "-inf", max_score, start=0, num=int(limit)),
                [],
            )
            or []
        )

    # -------- 7. 观测 --------
    def health(self) -> dict:
        started = time.monotonic()
        ok = False
        if self.enabled and self._client is not None:
            ok = bool(self.run("ping", lambda c: c.ping(), False))
        return {
            "enabled": self.enabled,
            "ok": ok,
            "backend": self.backend,
            "latency_ms": round((time.monotonic() - started) * 1000, 2) if ok else None,
            "breaker_open": time.monotonic() < self._open_until,
            "errors": self.errors,
            "fallbacks": self.fallbacks,
            "last_error": self._last_error[:200],
        }

    def stats(self) -> dict:
        total = self.ops or 1
        return {
            "backend": self.backend,
            "enabled": self.enabled,
            "ops": self.ops,
            "errors": self.errors,
            "fallbacks": self.fallbacks,
            "error_rate": round(self.errors / total, 4),
            "breaker_open": time.monotonic() < self._open_until,
        }


# ==================== 单例 ====================
_singleton: Optional[RedisManager] = None
_singleton_lock = threading.Lock()


def get_redis() -> RedisManager:
    """获取 Redis 单例（首次调用时连接）"""
    global _singleton
    if _singleton is None:
        with _singleton_lock:
            if _singleton is None:
                _singleton = RedisManager()
    return _singleton


def reset_redis() -> None:
    """重置单例（测试 / 配置热更新用）"""
    global _singleton
    with _singleton_lock:
        _singleton = None


def configure_client(client: Any) -> RedisManager:
    """
    注入自定义 client（测试用 fakeredis）
    =====================================
    注入后 backend 标记为 fake，available 为 True。
    """
    global _singleton
    with _singleton_lock:
        mgr = RedisManager(enabled=True, auto_connect=False)
        mgr._client = client
        mgr.backend = "fake"
        mgr._fails = 0
        mgr._open_until = 0.0
        _singleton = mgr
    return mgr
