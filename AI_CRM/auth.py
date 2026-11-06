"""
auth.py - 用户认证核心模块
============================
功能分区：
  1. 密码加密与验证（PBKDF2 + SHA256 + 随机盐值）
  2. 账户注册（用户名/手机号/密码）
  3. 用户登录（用户名/手机号 + 密码，含失败锁定保护）
  4. 手机验证码生成与验证（模拟短信，可对接阿里云 API）
  5. 密码重置流程（验证码验证 → 设置新密码）
  6. 管理员账户初始化（首次运行自动创建）

安全设计：
  [OK] 密码不可逆存储（PBKDF2 + SHA256 + 盐值）
  [OK] 登录失败锁定（连续5次失败，锁定15分钟）
  [OK] 验证码有效期（5分钟过期）
  [OK] 验证码单次使用后失效
  [OK] 密码强度校验（至少8位，含大小写+数字）
  [OK] 角色权限隔离（admin/user 路由保护）
"""

# ==================== 1. 导入依赖 ====================
import hashlib                      # 密码哈希算法
import hmac                         # 恒定时间比较（防时序攻击）
import os                           # 随机盐值生成
import uuid                         # 唯一令牌生成
import random                       # 验证码随机数
from datetime import datetime, timedelta  # 时间处理
import secrets                     # 安全随机数生成
from typing import Optional, Tuple  # 类型提示

from config import (
    PASSWORD_SALT_LENGTH,           # 盐值长度：32 字节
    PASSWORD_HASH_ITERATIONS,       # PBKDF2 迭代次数：100000
    PASSWORD_MIN_LENGTH,            # 密码最小长度：8
    VERIFICATION_CODE_EXPIRE_MINUTES,  # 验证码有效期：5 分钟
    LOGIN_FAIL_LOCK_THRESHOLD,      # 锁定阈值：5 次
    LOGIN_LOCK_MINUTES,             # 锁定时间：15 分钟
    ADMIN_DEFAULT_USERNAME,         # 默认管理员用户名
    ADMIN_DEFAULT_PASSWORD,         # 默认管理员密码（未设置时为 None）
    ADMIN_DEFAULT_PHONE,            # 默认管理员手机号
)
from models import Account, PasswordResetToken, LoginLog
from memory_sqlite import SQLiteMemory
from logger import logger


# ==================== 2. 密码加密与验证 ====================
class PasswordManager:
    """
    密码管理器
    ===========
    使用 PBKDF2-HMAC-SHA256 算法进行密码哈希
    每次加密自动生成随机盐值，保证相同密码不同哈希
    
    安全等级：企业级（同 Django/Bitwarden 等使用的算法）
    """
    
    @staticmethod
    def hash_password(password: str) -> str:
        """
        对密码进行哈希加密
        ====================
        格式：algorithm$iterations$salt$hash
        示例：pbkdf2$100000$a1b2c3d4...$e5f6g7h8...
        
        参数：
          password : 明文密码
        返回：
          带元信息的哈希字符串（可逆解析用于验证）
        """
        # 生成随机盐值
        salt = os.urandom(PASSWORD_SALT_LENGTH)
        
        # PBKDF2 密码哈希
        pwd_hash = hashlib.pbkdf2_hmac(
            'sha256',                   # 哈希算法
            password.encode('utf-8'),   # 密码转字节
            salt,                       # 盐值
            PASSWORD_HASH_ITERATIONS,   # 迭代次数
        )
        
        # 编码为十六进制字符串
        salt_hex = salt.hex()
        hash_hex = pwd_hash.hex()
        
        # 返回带格式的哈希字符串
        return f"pbkdf2${PASSWORD_HASH_ITERATIONS}${salt_hex}${hash_hex}"
    
    @staticmethod
    def verify_password(password: str, hashed: str) -> bool:
        """
        验证密码是否匹配哈希
        ======================
        参数：
          password : 待验证的明文密码
          hashed   : 存储的哈希字符串
        返回：
          True=匹配 / False=不匹配
        """
        try:
            # 解析哈希字符串
            parts = hashed.split('$')
            if len(parts) != 4:
                logger.error(f"[FAIL] 密码哈希格式异常: {len(parts)} 段")
                return False
            
            algorithm = parts[0]        # 算法标识（当前仅支持 pbkdf2）
            iterations = int(parts[1])  # 迭代次数
            salt_hex = parts[2]         # 盐值（十六进制）
            stored_hash = parts[3]      # 存储的哈希值
            
            if algorithm != 'pbkdf2':
                logger.error(f"[FAIL] 不支持的哈希算法: {algorithm}")
                return False
            
            # 将盐值从十六进制转回字节
            salt = bytes.fromhex(salt_hex)
            
            # 用相同参数计算密码哈希
            pwd_hash = hashlib.pbkdf2_hmac(
                'sha256',
                password.encode('utf-8'),
                salt,
                iterations,
            )
            
            # 比对哈希值（恒定时间比较，防时序攻击）
            # hmac.compare_digest 是恒定时间比较函数
            result = hmac.compare_digest(pwd_hash.hex(), stored_hash)
            return result
            
        except Exception as e:
            logger.error(f"[FAIL] 密码验证异常: {e}")
            return False


# ==================== 3. 验证码生成器 ====================
class VerificationCodeManager:
    """
    验证码管理器
    =============
    生成6位数字验证码
    当前为本地模拟模式（验证码输出到控制台）
    后续可对接阿里云短信 API / 腾讯云 SMS
    """
    
    @staticmethod
    def generate_code() -> str:
        """
        生成6位数字验证码
        ==================
        返回：
          6位数字字符串（如 "385291"）
        """
        return f"{random.randint(100000, 999999)}"
    
    @staticmethod
    def simulate_send_sms(phone: str, code: str) -> bool:
        """
        模拟发送短信验证码
        ====================
        当前为本地模拟模式：
          - 将验证码输出到控制台日志
          - 后续可替换为真实短信 API
        
        参数：
          phone : 手机号
          code  : 验证码
        返回：
          True=发送成功 / False=失败
        """
        logger.info(f"[PHONE] [模拟短信] 发送至 {phone}: 验证码 {code}，有效期 {VERIFICATION_CODE_EXPIRE_MINUTES} 分钟")
        print(f"\n{'='*50}")
        print(f"  [PHONE] 【模拟短信】发送至 {phone}")
        print(f"  [KEY] 验证码：{code}")
        print(f"  ⏰ 有效期：{VERIFICATION_CODE_EXPIRE_MINUTES} 分钟")
        print(f"{'='*50}\n")
        return True


# ==================== 4. 密码校验器 ====================
class PasswordValidator:
    """
    密码强度校验器
    ===============
    规则：
      - 至少 PASSWORD_MIN_LENGTH 位（默认 8 位）
      - 至少包含 1 个大写字母
      - 至少包含 1 个小写字母
      - 至少包含 1 个数字
      - 可选：至少包含 1 个特殊字符
    """
    
    @staticmethod
    def validate(password: str) -> Tuple[bool, str]:
        """
        校验密码强度
        =============
        参数：
          password : 待校验的密码
        返回：
          (是否通过, 错误信息)
        """
        # 长度检查
        if len(password) < PASSWORD_MIN_LENGTH:
            return False, f"密码长度不能少于 {PASSWORD_MIN_LENGTH} 位"
        
        # 大写字母检查
        if not any(c.isupper() for c in password):
            return False, "密码必须包含至少 1 个大写字母"
        
        # 小写字母检查
        if not any(c.islower() for c in password):
            return False, "密码必须包含至少 1 个小写字母"
        
        # 数字检查
        if not any(c.isdigit() for c in password):
            return False, "密码必须包含至少 1 个数字"
        
        return True, ""


# ==================== 5. 认证服务（核心） ====================
class AuthService:
    """
    认证服务（核心业务逻辑）
    =========================
    职责：
      1. 账户注册（用户名/手机号/密码）
      2. 用户登录（含失败锁定保护）
      3. 手机验证码登录
      4. 发送验证码（忘记密码）
      5. 密码重置
      6. 管理员账户自动初始化
    
    使用方式：
      auth = AuthService(db)
      result = auth.login("admin", "admin123")
      result = auth.register("newuser", "12345678901", "Pass1234")
    """
    
    def __init__(self, db: SQLiteMemory):
        """
        初始化认证服务
        ==============
        参数：
          db : SQLiteMemory 数据库实例
        
        首次初始化时自动创建管理员账户（如不存在）
        """
        self.db = db
        self._init_admin_account()
    
    def _init_admin_account(self):
        """
        初始化管理员账户
        ==================
        首次运行时，如果系统中没有管理员账户，自动创建管理员。
        
        安全策略：
          - 如果设置了 ADMIN_PASSWORD 环境变量，则使用该密码
          - 如果未设置 ADMIN_PASSWORD，则自动生成 16 位随机密码
          - 随机密码包含大小写字母 + 数字 + 特殊字符
          - 密码仅在此处打印一次，不会持久化存储
        """
        admin = self.db.get_account_by_username(ADMIN_DEFAULT_USERNAME)
        if admin is None:
            logger.info("[LOCK] 正在初始化管理员账户...")

            # 确定管理员密码
            if ADMIN_DEFAULT_PASSWORD:
                # 环境变量设置了密码，直接使用
                admin_password = ADMIN_DEFAULT_PASSWORD
                is_random = False
            else:
                # 未设置密码，生成 16 位安全随机密码
                # 使用 secrets 模块保证密码不可预测
                alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*"
                admin_password = "".join(secrets.choice(alphabet) for _ in range(16))
                is_random = True
            
            password_hash = PasswordManager.hash_password(admin_password)
            
            account = Account(
                username=ADMIN_DEFAULT_USERNAME,
                password_hash=password_hash,
                role="admin",
                phone=ADMIN_DEFAULT_PHONE,
            )
            
            if self.db.save_account(account):
                logger.info(f"管理员账户已创建: {ADMIN_DEFAULT_USERNAME}")
                print(f"\n{'='*50}")
                print(f"  [管理员账户已自动创建]")
                print(f"  用户名：{ADMIN_DEFAULT_USERNAME}")
                print(f"  密  码：{admin_password}")
                print(f"  手机号：{ADMIN_DEFAULT_PHONE}")
                if is_random:
                    print(f"  [注意] 请立即保存此密码！这是唯一一次显示！")
                else:
                    print(f"  [注意] 登录后请立即修改密码！")
                print(f"{'='*50}\n")
            else:
                logger.error("[FAIL] 管理员账户创建失败")
    
    # ==================== 5.1 用户注册 ====================
    def register(
        self,
        username: str,
        phone: str,
        password: str,
        email: Optional[str] = None,
    ) -> Tuple[bool, str, Optional[Account]]:
        """
        用户注册
        =========
        参数：
          username : 用户名（唯一）
          phone    : 手机号（唯一）
          password : 密码
          email    : 邮箱（可选）
        返回：
          (是否成功, 消息, Account对象)
        """
        # -------- 检查用户名是否已存在 --------
        existing = self.db.get_account_by_username(username)
        if existing:
            return False, "用户名已被注册", None
        
        # -------- 检查手机号是否已存在 --------
        existing_phone = self.db.get_account_by_phone(phone)
        if existing_phone:
            return False, "手机号已被注册", None
        
        # -------- 校验密码强度 --------
        valid, error_msg = PasswordValidator.validate(password)
        if not valid:
            return False, error_msg, None
        
        # -------- 创建账户 --------
        password_hash = PasswordManager.hash_password(password)
        
        account = Account(
            username=username,
            password_hash=password_hash,
            role="user",
            phone=phone,
            email=email,
        )
        
        if self.db.save_account(account):
            logger.info(f"[OK] 用户注册成功: {username} ({phone})")
            return True, "注册成功", account
        else:
            return False, "注册失败，请稍后重试", None
    
    # ==================== 5.2 用户登录 ====================
    def login(
        self,
        username_or_phone: str,
        password: str,
        ip_address: Optional[str] = None,
    ) -> Tuple[bool, str, Optional[Account]]:
        """
        用户登录
        =========
        支持用户名或手机号登录
        
        安全机制：
          - 连续失败超过阈值 → 账户锁定 N 分钟
          - 锁定期间即使密码正确也拒绝登录
          - 记录登录日志用于审计
        
        参数：
          username_or_phone : 用户名 或 手机号
          password          : 密码
          ip_address        : 登录 IP（可选）
        返回：
          (是否成功, 消息, Account对象)
        """
        # -------- 查找账户（支持用户名或手机号） --------
        account = self.db.get_account_by_username(username_or_phone)
        if account is None:
            account = self.db.get_account_by_phone(username_or_phone)
        
        if account is None:
            # 记录失败日志（用户名不存在）
            self._log_fail(username_or_phone, ip_address, "账户不存在")
            return False, "用户名或密码错误", None
        
        username = account.username
        
        # -------- 检查账户是否激活 --------
        if not account.is_active:
            self._log_fail(username, ip_address, "账户已被禁用")
            return False, "账户已被禁用，请联系管理员", None
        
        # -------- 登录失败锁定检查（Redis 计数，DB 兜底） --------
        recent_fails = self._fail_count(username)
        
        if recent_fails >= LOGIN_FAIL_LOCK_THRESHOLD:
            self._log_fail(username, ip_address, f"登录已锁定（{recent_fails}次失败）")
            return (
                False,
                f"登录过于频繁，账户已锁定 {LOGIN_LOCK_MINUTES} 分钟，请稍后再试",
                None,
            )
        
        # -------- 验证密码 --------
        if not PasswordManager.verify_password(password, account.password_hash):
            remaining = LOGIN_FAIL_LOCK_THRESHOLD - recent_fails - 1
            self._log_fail(username, ip_address, "密码错误")
            
            if remaining > 0:
                return False, f"密码错误，还剩 {remaining} 次尝试机会", None
            else:
                return (
                    False,
                    f"密码错误次数过多，账户已锁定 {LOGIN_LOCK_MINUTES} 分钟",
                    None,
                )
        
        # -------- 登录成功 --------
        # 更新最后登录时间
        self.db.update_last_login(account.account_id)
        
        # 记录登录成功日志
        self._log_login(account, ip_address)
        
        logger.info(f"[OK] 用户登录成功: {username}")
        return True, "登录成功", account
    
    # ==================== 5.3 验证码登录 ====================
    def login_with_code(
        self,
        phone: str,
        code: str,
        ip_address: Optional[str] = None,
    ) -> Tuple[bool, str, Optional[Account]]:
        """
        手机验证码登录
        ===============
        参数：
          phone      : 手机号
          code       : 验证码
          ip_address : 登录 IP（可选）
        返回：
          (是否成功, 消息, Account对象)
        """
        # -------- 查找手机号对应的账户 --------
        account = self.db.get_account_by_phone(phone)
        if account is None:
            return False, "该手机号未注册", None
        
        if not account.is_active:
            return False, "账户已被禁用，请联系管理员", None
        
        # -------- 验证验证码 --------
        valid_token = self.db.get_valid_reset_token(
            account.account_id, code
        )
        
        if valid_token is None:
            return False, "验证码无效或已过期", None
        
        # -------- 标记验证码已使用 --------
        self.db.mark_token_used(valid_token.token_id)
        
        # -------- 登录成功 --------
        self.db.update_last_login(account.account_id)
        self._log_login(account, ip_address)
        
        logger.info(f"[OK] 验证码登录成功: {phone}")
        return True, "登录成功", account
    
    # ==================== 5.4 发送验证码 ====================
    def send_verification_code(self, phone: str) -> Tuple[bool, str]:
        """
        发送手机验证码
        ===============
        用于：
          - 忘记密码时的身份验证
          - 手机号验证码登录
        
        流程：
          1. 检查手机号是否已注册
          2. 生成6位随机验证码
          3. 存储到数据库（含过期时间）
          4. 模拟发送短信（控制台打印）
        
        参数：
          phone : 手机号
        返回：
          (是否成功, 消息)
        """
        # -------- 检查手机号是否已注册 --------
        account = self.db.get_account_by_phone(phone)
        if account is None:
            return False, "该手机号未注册"
        
        # -------- 生成验证码和令牌 --------
        code = VerificationCodeManager.generate_code()
        token = str(uuid.uuid4())
        expires_at = datetime.now() + timedelta(
            minutes=VERIFICATION_CODE_EXPIRE_MINUTES
        )
        
        # -------- 存储到数据库 --------
        reset_token = PasswordResetToken(
            account_id=account.account_id,
            token=token,
            code=code,
            expires_at=expires_at,
        )
        
        if self.db.save_reset_token(reset_token):
            # 模拟发送短信
            VerificationCodeManager.simulate_send_sms(phone, code)
            return True, f"验证码已发送至 {phone}，有效期 {VERIFICATION_CODE_EXPIRE_MINUTES} 分钟"
        else:
            return False, "验证码发送失败，请稍后重试"
    
    # ==================== 5.5 忘记密码 - 验证验证码 ====================
    def verify_code_and_get_token(
        self, phone: str, code: str
    ) -> Tuple[bool, str, Optional[str]]:
        """
        验证手机验证码并获取重置令牌
        =============================
        参数：
          phone : 手机号
          code  : 验证码
        返回：
          (是否成功, 消息, 重置token)
        """
        account = self.db.get_account_by_phone(phone)
        if account is None:
            return False, "该手机号未注册", None
        
        valid_token = self.db.get_valid_reset_token(
            account.account_id, code
        )
        
        if valid_token is None:
            return False, "验证码无效或已过期", None
        
        # 返回重置令牌给前端，用于后续重置密码的凭证
        return True, "验证通过", valid_token.token
    
    # ==================== 5.6 忘记密码 - 重置密码 ====================
    def reset_password(
        self, phone: str, code: str, new_password: str
    ) -> Tuple[bool, str]:
        """
        重置密码（完整流程：验证码验证 + 密码更新）
        ============================================
        参数：
          phone       : 手机号
          code        : 验证码
          new_password : 新密码
        返回：
          (是否成功, 消息)
        """
        # -------- 校验新密码强度 --------
        valid, error_msg = PasswordValidator.validate(new_password)
        if not valid:
            return False, error_msg
        
        # -------- 查找账户 --------
        account = self.db.get_account_by_phone(phone)
        if account is None:
            return False, "该手机号未注册"
        
        # -------- 验证验证码 --------
        valid_token = self.db.get_valid_reset_token(
            account.account_id, code
        )
        
        if valid_token is None:
            return False, "验证码无效或已过期"
        
        # -------- 更新密码 --------
        new_password_hash = PasswordManager.hash_password(new_password)
        
        if self.db.update_account_password(
            account.account_id, new_password_hash
        ):
            # 标记验证码已使用
            self.db.mark_token_used(valid_token.token_id)
            logger.info(f"[OK] 密码重置成功: {account.username}")
            return True, "密码重置成功，请使用新密码登录"
        else:
            return False, "密码重置失败，请稍后重试"
    
    # ==================== 5.7 修改密码（已登录状态） ====================
    def change_password(
        self, account_id: str, old_password: str, new_password: str
    ) -> Tuple[bool, str]:
        """
        已登录用户修改密码
        ===================
        参数：
          account_id  : 账户 ID
          old_password : 旧密码
          new_password : 新密码
        返回：
          (是否成功, 消息)
        """
        # -------- 获取账户信息 --------
        account = self.db.get_account_by_id(account_id)
        if account is None:
            return False, "账户不存在"
        
        # -------- 验证旧密码 --------
        if not PasswordManager.verify_password(
            old_password, account.password_hash
        ):
            return False, "旧密码错误"
        
        # -------- 校验新密码强度 --------
        valid, error_msg = PasswordValidator.validate(new_password)
        if not valid:
            return False, error_msg
        
        # -------- 新旧密码不能相同 --------
        if old_password == new_password:
            return False, "新密码不能与旧密码相同"
        
        # -------- 更新密码 --------
        new_password_hash = PasswordManager.hash_password(new_password)
        
        if self.db.update_account_password(account_id, new_password_hash):
            logger.info(f"[OK] 密码修改成功: {account.username}")
            return True, "密码修改成功"
        else:
            return False, "密码修改失败，请稍后重试"
    
    # ==================== 6. 日志记录 ====================
    def _log_login(self, account: Account, ip_address: Optional[str] = None):
        """记录登录成功日志（并清空失败计数）"""
        log = LoginLog(
            account_id=account.account_id,
            username=account.username,
            action="login",
            ip_address=ip_address,
            detail="登录成功",
        )
        self.db.save_login_log(log)
        self._clear_fail_count(account.username)

    def _log_fail(self, username: str, ip_address: Optional[str] = None, detail: str = ""):
        """记录登录失败日志（并累加 Redis 失败计数）"""
        log = LoginLog(
            username=username,
            action="fail",
            ip_address=ip_address,
            detail=detail,
        )
        self.db.save_login_log(log)
        self._record_fail(username)

    # -------- 6.1 登录失败计数（Redis 优先，数据库兜底） --------
    def _fail_key(self, username: str) -> str:
        return f"loginfail:{username}"

    def _record_fail(self, username: str) -> None:
        """失败一次：INCR + 首次设置过期（与 LOGIN_LOCK_MINUTES 对齐）"""
        from redis_client import get_redis
        get_redis().incr(
            self._fail_key(username), ttl=LOGIN_LOCK_MINUTES * 60
        )

    def _clear_fail_count(self, username: str) -> None:
        from redis_client import get_redis
        get_redis().delete(self._fail_key(username))

    def _fail_count(self, username: str) -> int:
        """
        读取窗口内失败次数
        ==================
        Redis 有值以 Redis 为准（多进程共享）；
        Redis 不可用（None）时回源数据库 login_logs 计数。
        """
        from redis_client import get_redis
        val = get_redis().get(self._fail_key(username))
        if val is not None:
            try:
                return int(val)
            except (TypeError, ValueError):
                pass
        return self.db.count_recent_fails(username, LOGIN_LOCK_MINUTES)
