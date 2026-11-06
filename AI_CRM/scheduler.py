"""
scheduler.py - 常驻定时调度 + 分层跑批 + 主动告警推送
================================================================================
背景（对应三面「定时调度与异常筛查」的整改）：
  此前 scheduler 是「单一 30 分钟轮询全量扫表」，存在两个问题：
    ① 全量扫表：客户量大时每轮都扫全表，无效扫描多、成本高；
    ② 无事件驱动：销售录入客户后不会「精准触发」，只能等下一轮轮询。

本模块重构为「事件驱动 + 分层跑批」四层架构：
  ① 事件驱动（精准触发）：销售录入信息 → 投递延迟消息（10 天后触发检查）；
     中途客户转化或已跟进 → 撤销延迟消息，做到**零无效扫描**。
  ② 晚上 21:00 粗筛（全量兜底）：销售收尾阶段，跑一次全量扫表，
     把可能触发异常的客户提前塞进「今日待办池」临时表。
  ③ 第二天下午 13:00 精筛（增量扫描与去重）：销售刚开机，只扫「今日待办池」，
     命中规则写告警表、亮红点；按「客户+类型+业务日」去重，一天只推送一次。
  ④ 字段驱动闭环（告警消除）：销售跟进完，把【最近跟进时间】或【转化状态】
     改为非命中规则，下次精筛无法命中，告警自动消失。

设计原则（与项目其它模块一致）：
  1. 纯标准库（threading + time），零外部依赖，不引入 APScheduler；
  2. 守护线程（daemon=True）：主进程退出即随之结束，不会挂住进程；
  3. 异常自愈：单轮任务抛异常只记日志，线程继续下一轮，绝不整体崩掉；
  4. 可开关：环境变量 SCHEDULER_ENABLED=0 可关闭（测试 / 本地调试用）；
  5. 可观测：每轮任务结果写 metrics 埋点 + 日志，便于监控大盘观察。

用法：
  from scheduler import get_scheduler
  sched = get_scheduler()          # 单例
  sched.start()                    # 启动（幂等，重复调用不会起两个线程）
  sched.run_event_consumer()       # 手动跑事件消费
  sched.run_coarse_scan()          # 手动跑 21 点粗筛
  sched.run_fine_scan()            # 手动跑次日 13 点精筛
  sched.stop()                     # 停止

自检：
  python scheduler.py
"""

# ==================== 1. 导入依赖 ====================
from __future__ import annotations

import os
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional

from logger import logger


# ==================== 2. 配置（环境变量可覆盖） ====================
def _env_int(name: str, default: int) -> int:
    """读取整型环境变量（非法值回退默认）"""
    try:
        return int(str(os.getenv(name, default)).strip())
    except (ValueError, TypeError):
        return default


# 事件消费轮询间隔（秒）：默认 60 秒（每分钟消费一次到期延迟消息）
EVENT_INTERVAL_SECONDS: int = _env_int("SCHEDULER_EVENT_INTERVAL_SECONDS", 60)
# 粗筛时间（小时）：默认 21 点
COARSE_HOUR: int = _env_int("SCHEDULER_COARSE_HOUR", 21)
# 精筛时间（小时）：默认次日 13 点
FINE_HOUR: int = _env_int("SCHEDULER_FINE_HOUR", 13)
# 时间窗口（分钟）：在该小时内前 N 分钟执行一次（避免同小时重复跑）
RUN_WINDOW_MINUTES: int = _env_int("SCHEDULER_RUN_WINDOW_MINUTES", 5)
# 延迟消息默认延迟天数（事件驱动：录入后 N 天触发检查）
DELAY_DAYS: int = _env_int("SCHEDULER_DELAY_DAYS", 10)
# R1「有意向未跟进」阈值天数（默认 10 天）
STALE_DAYS: int = _env_int("SCHEDULER_STALE_DAYS", 10)
# R2「正价课未转化」阈值天数（默认 30 天）
CHURN_DAYS: int = _env_int("SCHEDULER_CHURN_DAYS", 30)
# R3「高风险低意向未激活/未转化」阈值天数（默认 45 天）
HIGH_RISK_DAYS: int = _env_int("SCHEDULER_HIGH_RISK_DAYS", 45)
# 「超期未跟进」阈值天数（到期巡检用）
DUE_DAYS: int = _env_int("SCHEDULER_DUE_DAYS", 14)
# 是否启用（0 / false / no 关闭）
SCHEDULER_ENABLED: bool = str(
    os.getenv("SCHEDULER_ENABLED", "1")
).strip().lower() not in ("0", "false", "no", "off")


# ==================== 3. 告警类型 ↔ 严重度映射 ====================
# 严重度决定前端展示优先级：high 红 / medium 橙 / low 灰
# 三条规则对应三类告警（R1 最急 → R3 最缓）
SEVERITY_BY_TYPE: Dict[str, str] = {
    "有意向未跟进": "high",     # R1 高价值线索正在冷却，最该先捞
    "流失沉睡风险": "high",     # R2 再不跟就彻底丢了
    "计划到期": "medium",
    "超期未跟进": "medium",
    "高风险低意向": "low",      # R3 长期沉默、转化概率低，做最后触达或归档
}


def _severity_of(alert_type: str) -> str:
    """按告警类型取严重度（未知类型默认 medium）"""
    return SEVERITY_BY_TYPE.get(alert_type, "medium")


def _classify_reason(reason: str) -> str:
    """
    把筛查返回的「原因文本」归类成告警类型
    ==========================================
    输入示例：
      "有意向未跟进(>10天或无跟进)"          → 有意向未跟进
      "流失/沉睡风险" / "正价课未转化(>30天)" → 流失沉睡风险
      "高风险低意向(>45天未激活/未转化)"      → 高风险低意向
    """
    if "有意向未跟进" in reason:
        return "有意向未跟进"
    # 注意：R3 的原因文本里也含「未转化」，必须先判 R3 再判 R2，
    # 否则 "高风险低意向(>45天未激活/未转化)" 会被误归到「流失沉睡风险」。
    if "高风险低意向" in reason:
        return "高风险低意向"
    if ("流失" in reason) or ("沉睡" in reason) or ("未转化" in reason):
        return "流失沉睡风险"
    return "其它"


def _group_reasons_by_type(reasons: List[str]) -> Dict[str, List[str]]:
    """把原因列表按告警类型分组（一个客户可能同时命中多条规则）"""
    grouped: Dict[str, List[str]] = {}
    for reason in reasons:
        grouped.setdefault(_classify_reason(reason), []).append(reason)
    return grouped


# ==================== 4. 调度器主体 ====================
class AlertScheduler:
    """
    常驻定时调度器（后台守护线程 · 分层跑批）
    ==========================================
    职责：
      ① 事件消费（每分钟）：消费到期延迟消息 → 命中规则塞入今日待办池
      ② 21:00 粗筛（全量兜底）：全量扫表 → 塞入今日待办池
      ③ 次日 13:00 精筛（增量去重）：只扫今日待办池 → 写告警（去重）
    线程模型：单后台线程串行执行，避免并发写库；
              每轮之间 sleep(interval)，可被 stop() 立即唤醒退出。
    """

    def __init__(self, interval_seconds: Optional[int] = None):
        # 事件消费轮询间隔（兼容旧参数名 interval_seconds）
        self.interval = int(interval_seconds or EVENT_INTERVAL_SECONDS)
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        # 运行统计（供 /api/alerts/status 观察调度器健康度）
        self.started_at: Optional[datetime] = None
        self.last_run_at: Optional[datetime] = None
        self.last_run_seconds: float = 0.0
        self.run_count: int = 0
        self.error_count: int = 0
        self.last_error: str = ""
        self.last_created: int = 0
        # 分层任务「当日已跑」标记（避免同一时间窗口重复跑）
        self._coarse_done_date: str = ""
        self._fine_done_date: str = ""

    # -------- 4.1 生命周期 --------
    @property
    def running(self) -> bool:
        """线程是否在跑"""
        return bool(self._thread and self._thread.is_alive())

    def start(self) -> bool:
        """
        启动常驻调度（幂等）
        =====================
        返回：True=本次真正启动 / False=已在运行或已禁用
        """
        if not SCHEDULER_ENABLED:
            logger.info("[SKIP] 定时调度已禁用（SCHEDULER_ENABLED=0）")
            return False
        with self._lock:
            if self.running:
                logger.info("[SKIP] 定时调度已在运行，忽略重复启动")
                return False
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._loop, name="alert-scheduler", daemon=True
            )
            self._thread.start()
            self.started_at = datetime.now()
            logger.info(
                f"[START] 分层调度已启动：事件消费每 {self.interval}s / "
                f"粗筛 {COARSE_HOUR}:00 / 精筛 {FINE_HOUR}:00"
                f"（stale_days={STALE_DAYS} / due_days={DUE_DAYS}）"
            )
            return True

    def stop(self, timeout: float = 3.0) -> bool:
        """
        停止调度（唤醒 sleep 并等待线程退出）
        ======================================
        返回：True=已停止 / False=本来就没在跑
        """
        with self._lock:
            if not self.running:
                return False
            self._stop_event.set()
            thread = self._thread
        if thread:
            thread.join(timeout=timeout)
        logger.info("[STOP] 分层调度已停止")
        return True

    # -------- 4.2 主循环（分层触发） --------
    def _loop(self) -> None:
        """
        后台线程主循环：
          · 每轮（默认 60s）跑一次事件消费；
          · 命中 21:00 窗口 → 跑粗筛（当日仅一次）；
          · 命中 13:00 窗口 → 跑精筛（当日仅一次）。
        """
        # 启动后稍等 3 秒再跑首轮，避免与进程启动初始化抢资源
        if self._stop_event.wait(3.0):
            return
        while not self._stop_event.is_set():
            round_started = time.time()
            now = datetime.now()
            today = now.strftime("%Y-%m-%d")

            # ① 事件消费（每分钟）
            try:
                self.run_event_consumer()
            except Exception as e:
                self.error_count += 1
                self.last_error = str(e)
                logger.error(f"[FAIL] 事件消费异常（线程继续）: {e}")

            # ② 21:00 粗筛（当日仅一次）
            if (
                now.hour == COARSE_HOUR
                and now.minute < RUN_WINDOW_MINUTES
                and self._coarse_done_date != today
            ):
                try:
                    self.run_coarse_scan()
                    self._coarse_done_date = today
                except Exception as e:
                    self.error_count += 1
                    self.last_error = str(e)
                    logger.error(f"[FAIL] 粗筛异常（线程继续）: {e}")

            # ③ 次日 13:00 精筛（当日仅一次）
            if (
                now.hour == FINE_HOUR
                and now.minute < RUN_WINDOW_MINUTES
                and self._fine_done_date != today
            ):
                try:
                    self.run_fine_scan()
                    self._fine_done_date = today
                except Exception as e:
                    self.error_count += 1
                    self.last_error = str(e)
                    logger.error(f"[FAIL] 精筛异常（线程继续）: {e}")

            # 一轮 = 一次 run_count（事件消费 + 窗口内可能的粗筛/精筛）
            self.run_count += 1
            self.last_run_at = datetime.now()
            self.last_run_seconds = round(time.time() - round_started, 3)

            # 用 wait 代替 sleep：stop() 时能立即唤醒退出
            if self._stop_event.wait(self.interval):
                break

    # -------- 4.3b 分布式跑批锁（多副本部署时只允许一个实例执行） --------
    def _acquire_scan_lock(self, job: str, ttl: int = 0):
        """
        抢跑批锁
        ========
        返回 (lock_name, token, distributed)：
          distributed=False  → Redis 不可用，按单实例模式照常执行（不阻断）
          token=None         → 锁被其它实例持有，本次应跳过
        """
        try:
            from config import REDIS_SCAN_LOCK_TTL
            from redis_client import get_redis

            rd = get_redis()
            if not rd.available:
                return None, None, False
            name = f"scan:{job}:{datetime.now():%Y-%m-%d-%H:%M}"
            token = rd.acquire_lock(name, ttl or REDIS_SCAN_LOCK_TTL)
            return name, token, True
        except Exception as exc:
            logger.warning(f"[WARN] 分布式锁不可用，按单实例执行: {exc}")
            return None, None, False

    def _release_scan_lock(self, name: Optional[str], token: Optional[str]) -> None:
        if name and token:
            try:
                from redis_client import get_redis
                get_redis().release_lock(name, token)
            except Exception:
                pass

    # -------- 4.3 任务①：事件消费（精准触发） --------
    def run_event_consumer(self) -> Dict[str, int]:
        """
        消费到期的延迟消息 → 命中规则则塞入今日待办池
        ================================================
        返回：{"fired", "added"}
        """
        # 每分钟一轮：锁名带「分钟」，多副本下同一分钟只有一台消费（TTL 120s 兜住单轮耗时）
        lock_name, lock_token, distributed = self._acquire_scan_lock("event", ttl=120)
        if distributed and lock_token is None:
            logger.debug("[SKIP] 本轮事件消费已由其他实例执行（Redis 锁）")
            return {"fired": 0, "added": 0}
        from memory_sqlite import SQLiteMemory

        db = SQLiteMemory()
        fired = 0
        added = 0
        try:
            due = db.fetch_due_delayed_messages()
            for m in due:
                db.mark_delayed_fired(m["msg_id"])
                fired += 1
                cid = m.get("customer_id")
                if not cid:
                    continue
                # 重新校验规则（字段驱动闭环：已跟进/已转化则不命中）
                if db.eval_anomaly_rules(cid):
                    if db.add_pending_todo(cid, source="event"):
                        added += 1
            if fired:
                logger.info(f"[EVENT] 事件消费完成：触发 {fired} 条，塞入待办 {added} 条")
        finally:
            try:
                db.close()
            except Exception:
                pass
            self._release_scan_lock(lock_name, lock_token)
        return {"fired": fired, "added": added}

    # -------- 4.4 任务②：21:00 粗筛（全量兜底） --------
    def run_coarse_scan(self) -> Dict[str, int]:
        """
        全量扫表，把可能触发异常的客户提前塞进「今日待办池」
        ======================================================
        返回：{"scanned", "coarse_added"}
        """
        t0 = time.time()
        lock_name, lock_token, distributed = self._acquire_scan_lock("coarse")
        if distributed and lock_token is None:
            logger.info("[SKIP] 21点粗筛已由其他实例执行（Redis 锁）")
            return {"scanned": 0, "coarse_added": 0}
        from memory_sqlite import SQLiteMemory

        db = SQLiteMemory()
        scanned = 0
        added = 0
        try:
            # screen_anomaly_customers 内部已用 list_active_customers（公海过滤）
            anomalies = db.screen_anomaly_customers(
                stale_days=STALE_DAYS,
                churn_days=CHURN_DAYS,
                high_risk_days=HIGH_RISK_DAYS,
            )
            scanned = len(anomalies)
            for item in anomalies:
                if db.add_pending_todo(
                    item.get("customer_id", ""),
                    advisor_id=item.get("advisor_id"),
                    source="coarse",
                ):
                    added += 1
            logger.info(
                f"[COARSE] 21点粗筛完成：扫描 {scanned} 条，塞入待办 {added} 条，"
                f"耗时 {time.time() - t0:.2f}s"
            )
            self._observe("coarse", t0, added)
        finally:
            try:
                db.close()
            except Exception:
                pass
            self._release_scan_lock(lock_name, lock_token)
        return {"scanned": scanned, "coarse_added": added}

    # -------- 4.5 任务③：次日 13:00 精筛（增量 + 去重） --------
    def run_fine_scan(self) -> Dict[str, int]:
        """
        只扫「今日待办池」，命中规则写告警表（按客户+类型+业务日去重）
        ================================================================
        返回：{"todos", "created", "resolved"}
        """
        t0 = time.time()
        lock_name, lock_token, distributed = self._acquire_scan_lock("fine")
        if distributed and lock_token is None:
            logger.info("[SKIP] 13点精筛已由其他实例执行（Redis 锁）")
            return {"todos": 0, "created": 0, "resolved": 0}
        from memory_sqlite import SQLiteMemory

        db = SQLiteMemory()
        todos = []
        created = 0
        resolved = 0
        biz_date = datetime.now().strftime("%Y-%m-%d")
        try:
            todos = db.list_pending_todos(biz_date=biz_date)
            for t in todos:
                cid = t.get("customer_id")
                if not cid:
                    db.mark_todo_done(t["todo_id"])
                    continue
                # 字段驱动闭环：精筛时重新校验规则（不直接信任待办池）
                reasons = db.eval_anomaly_rules(cid)
                if not reasons:
                    # 已跟进 / 已转化 → 不命中，告警自动消除
                    resolved += db.resolve_alerts(cid)
                    db.mark_todo_done(t["todo_id"])
                    continue
                # 命中 → 按「客户+类型+业务日」去重写告警
                # 补上客户姓名/顾问：告警列表直接可读，不用再回表拼 ID
                customer = db.get_customer(cid)
                customer_name = customer.name if customer else None
                advisor_id = t.get("advisor_id") or (customer.advisor_id if customer else None)
                for alert_type, rs in _group_reasons_by_type(reasons).items():
                    ok = db.upsert_alert(
                        customer_id=cid,
                        alert_type=alert_type,
                        reasons=rs,
                        customer_name=customer_name,
                        advisor_id=advisor_id,
                        severity=_severity_of(alert_type),
                        source="scheduler",
                    )
                    if ok:
                        created += 1
                db.mark_todo_done(t["todo_id"])
            logger.info(
                f"[FINE] 次日13点精筛完成：待办 {len(todos)} 条，"
                f"落告警 {created} 条，消除 {resolved} 条，耗时 {time.time() - t0:.2f}s"
            )
            self._observe("fine", t0, created)
        finally:
            try:
                db.close()
            except Exception:
                pass
            self._release_scan_lock(lock_name, lock_token)
        return {"todos": len(todos), "created": created, "resolved": resolved}

    # -------- 4.6 埋点（失败不影响主流程） --------
    def _observe(self, stage: str, t0: float, created: int) -> None:
        try:
            from metrics import incr, observe
            incr("scheduler_run_total", status="success", stage=stage)
            incr("alert_created_total", source="scheduler", value=created)
            observe("scheduler_run_ms", (time.time() - t0) * 1000)
        except Exception:
            pass
        self.last_created = created

    # -------- 4.7 兼容旧接口：run_once（= 事件消费 + 粗筛 + 精筛） --------
    def run_once(self) -> Dict[str, int]:
        """
        兼容旧调用：手动跑一轮完整分层流程（事件消费 + 粗筛 + 精筛）
        ==============================================================
        返回合并统计，便于测试 / 立即巡检。
        同时更新 run_count / last_run_at，保持与常驻调度一致的观测口径。
        """
        t0 = time.time()
        ev = self.run_event_consumer()
        coarse = self.run_coarse_scan()
        fine = self.run_fine_scan()
        self.run_count += 1
        self.last_run_at = datetime.now()
        self.last_run_seconds = round(time.time() - t0, 3)
        self.last_created = fine["created"]
        return {
            # 旧字段（保留兼容）
            "fired": ev["fired"],
            "coarse_added": coarse["coarse_added"],
            "fine_created": fine["created"],
            "fine_resolved": fine["resolved"],
            # 统一统计口径（与 status() 对齐）
            "scanned": coarse["scanned"],
            "created": fine["created"],
            "skipped": 0,
            "errors": self.error_count,
        }

    # -------- 4.8 健康状态（供接口观察） --------
    def status(self) -> Dict:
        """返回调度器运行状态快照"""
        return {
            "enabled": SCHEDULER_ENABLED,
            "running": self.running,
            "event_interval_seconds": self.interval,
            "coarse_hour": COARSE_HOUR,
            "fine_hour": FINE_HOUR,
            "delay_days": DELAY_DAYS,
            "stale_days": STALE_DAYS,
            "churn_days": CHURN_DAYS,
            "high_risk_days": HIGH_RISK_DAYS,
            "due_days": DUE_DAYS,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "last_run_at": self.last_run_at.isoformat() if self.last_run_at else None,
            "last_run_seconds": self.last_run_seconds,
            "run_count": self.run_count,
            "error_count": self.error_count,
            "last_error": self.last_error,
            "last_created": self.last_created,
            "coarse_done_date": self._coarse_done_date,
            "fine_done_date": self._fine_done_date,
        }


# ==================== 5. 单例访问 ====================
_scheduler_singleton: Optional[AlertScheduler] = None
_singleton_lock = threading.Lock()


def get_scheduler() -> AlertScheduler:
    """获取全局调度器单例（线程安全）"""
    global _scheduler_singleton
    if _scheduler_singleton is None:
        with _singleton_lock:
            if _scheduler_singleton is None:
                _scheduler_singleton = AlertScheduler()
    return _scheduler_singleton


# ==================== 6. 自检 ====================
if __name__ == "__main__":
    print("=" * 60)
    print("scheduler.py 自检（分层跑批）")
    print("=" * 60)
    print(f"启用状态        : {SCHEDULER_ENABLED}")
    print(f"事件消费间隔(秒): {EVENT_INTERVAL_SECONDS}")
    print(f"粗筛时间        : {COARSE_HOUR}:00")
    print(f"精筛时间        : {FINE_HOUR}:00")
    print(f"延迟天数        : {DELAY_DAYS}")
    print(f"未跟进阈值(天)  : {STALE_DAYS}")
    print(f"超期阈值(天)    : {DUE_DAYS}")
    print("-" * 60)

    sched = get_scheduler()
    print(f"单例获取成功    : {sched is get_scheduler()}")
    print(f"初始运行状态    : {sched.running}")

    print("-" * 60)
    print("手动跑一轮完整分层流程（run_once）...")
    result = sched.run_once()
    print(f"分层结果        : {result}")
    print(f"运行状态        : {sched.status()}")

    print("-" * 60)
    print("启动常驻线程 → 立即停止（验证生命周期）...")
    started = sched.start()
    print(f"start() 返回    : {started}")
    print(f"启动后 running  : {sched.running}")
    stopped = sched.stop()
    print(f"stop() 返回     : {stopped}")
    print(f"停止后 running  : {sched.running}")
    print("=" * 60)
    print("[OK] scheduler.py 自检完成")
