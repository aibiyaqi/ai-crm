"""
app.py - Streamlit 可视化前端界面（集成用户认证）
====================================================
功能分区：
  1. 页面配置与全局美化样式
  2. 初始化后端组件（LLM / DB / RAG / Search / Workflow / Auth）
  3. 登录页面（密码登录 / 验证码登录 / 注册 / 忘记密码）
  4. 主应用（对话 / 客户信息 / 知识库 / 统计）
  5. 管理员后台（用户管理 / 登录日志审计）
  6. 角色权限路由控制

运行方式：
  cd AI_CRM && streamlit run app.py
"""

# ==================== 1. 导入依赖 ====================
import streamlit as st
import sys
import os
import uuid
from datetime import datetime

# 将项目根目录加入 sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 启动时校验关键环境变量
from config import validate_env_or_exit
validate_env_or_exit()

from logger import logger


# ==================== 2. 页面配置 ====================
st.set_page_config(
    page_title="AI CRM 智能客户经营平台",
    page_icon="[AI]",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==================== 3. 全局美化 CSS ====================
st.markdown("""
<style>
    /* ===== 全局样式 ===== */
    @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css');
    
    .main-title {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* ===== 聊天消息 ===== */
    .chat-user {
        background: linear-gradient(135deg, #e3f2fd, #bbdefb);
        padding: 12px 18px;
        border-radius: 18px 18px 5px 18px;
        margin: 8px 0;
        color: #1a237e;
        border: 1px solid #90caf9;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .chat-assistant {
        background: linear-gradient(135deg, #f5f5f5, #e8e8e8);
        padding: 12px 18px;
        border-radius: 18px 18px 18px 5px;
        margin: 8px 0;
        color: #333;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* ===== 侧边栏 ===== */
    .sidebar-title {
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    /* ===== 状态标签 ===== */
    .status-online { color: #4caf50; font-weight: 600; }
    .status-offline { color: #f44336; font-weight: 600; }
    
    /* ===== 数据卡片 ===== */
    .metric-card {
        background: linear-gradient(135deg, #f8f9fa, #ffffff);
        padding: 15px;
        border-radius: 12px;
        text-align: center;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        transition: transform 0.2s;
    }
    .metric-card:hover { transform: translateY(-2px); }
    
    /* ===== 登录/注册卡片 ===== */
    .auth-container {
        max-width: 420px;
        margin: 0 auto;
        padding: 2rem;
        background: white;
        border-radius: 16px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.08);
        border: 1px solid #e8e8e8;
    }
    .auth-title {
        text-align: center;
        font-size: 1.6rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .auth-subtitle {
        text-align: center;
        color: #666;
        font-size: 0.9rem;
        margin-bottom: 1.5rem;
    }
    .auth-divider {
        margin: 1.5rem 0;
        border: none;
        border-top: 1px solid #e8e8e8;
    }
    .auth-footer {
        text-align: center;
        color: #888;
        font-size: 0.85rem;
        margin-top: 1rem;
    }
    
    /* ===== 按钮优化 ===== */
    div.stButton > button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s;
    }
    div.stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    /* ===== 输入框优化 ===== */
    div.stTextInput > div > div > input {
        border-radius: 8px;
        border: 1px solid #ddd;
    }
    div.stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 2px rgba(102,126,234,0.1);
    }
    
    /* ===== 分割线 ===== */
    hr {
        margin: 1.5rem 0;
        border: none;
        border-top: 1px solid #e8e8e8;
    }
</style>
""", unsafe_allow_html=True)


# ==================== 4. 初始化后端组件 ====================
@st.cache_resource
def init_backend():
    """初始化所有后端组件（仅执行一次）"""
    components = {
        "llm": None,
        "db": None,
        "workflow": None,
        "rag": None,
        "auth": None,
        "rag_ready": False,
        "search_ready": False,
    }
    
    # -------- 4.1 初始化数据库 --------
    from memory_sqlite import SQLiteMemory
    try:
        db = SQLiteMemory()
        components["db"] = db
        logger.info("[OK] 数据库初始化成功")
    except Exception as e:
        st.error(f"[FAIL] 数据库初始化失败: {e}")
        return components
    
    # -------- 4.2 初始化认证服务 --------
    from auth import AuthService
    try:
        auth = AuthService(db)
        components["auth"] = auth
        logger.info("[OK] 认证服务初始化成功")
    except Exception as e:
        st.error(f"[FAIL] 认证服务初始化失败: {e}")
        return components
    
    # -------- 4.3 初始化 LLM --------
    from llms import QwenLLM
    try:
        llm = QwenLLM()
        components["llm"] = llm
        logger.info("[OK] LLM 初始化成功")
    except Exception as e:
        st.error(f"[FAIL] LLM 初始化失败: {e}")
        return components
    
    # -------- 4.4 初始化 RAG（可选） --------
    try:
        from rag_law import RAGLaw
        rag = RAGLaw()
        components["rag"] = rag
        components["rag_ready"] = True
        logger.info("[OK] RAG 知识库初始化成功")
    except Exception as e:
        logger.warning(f"[WARN] RAG 未加载: {e}")
    
    # -------- 4.5 初始化搜索工具（可选） --------
    try:
        from web_search import WebSearchTool
        searcher = WebSearchTool()
        components["searcher"] = searcher
        components["search_ready"] = True
        logger.info("[OK] 搜索工具初始化成功")
    except Exception as e:
        logger.warning(f"[WARN] 搜索工具未加载: {e}")
    
    # -------- 4.6 初始化工作流 --------
    from agent import CRMWorkflow
    try:
        workflow = CRMWorkflow(
            llm=components["llm"],
            db=db,
            rag=components.get("rag"),
            searcher=components.get("searcher"),
        )
        components["workflow"] = workflow
        logger.info("[OK] LangGraph 工作流初始化成功")
    except Exception as e:
        st.error(f"[FAIL] 工作流初始化失败: {e}")
    
    return components


# ==================== 5. 初始化会话状态 ====================
def init_session_state():
    """初始化所有会话状态"""
    if "backend" not in st.session_state:
        st.session_state.backend = init_backend()
    
    # -------- 认证状态 --------
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "current_account" not in st.session_state:
        st.session_state.current_account = None
    if "auth_page" not in st.session_state:
        st.session_state.auth_page = "login"  # login / register / forgot
    if "session_token" not in st.session_state:
        st.session_state.session_token = None
    
    # -------- 多对话框状态 --------
    if "sessions" not in st.session_state:
        st.session_state.sessions = []        # 会话列表 [{session_id, title}]
    if "current_session_id" not in st.session_state:
        st.session_state.current_session_id = None  # 当前活跃会话 ID
    if "session_messages" not in st.session_state:
        st.session_state.session_messages = {}  # {session_id: [messages]}
    
    # -------- 应用状态 --------
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "web_search_enabled" not in st.session_state:
        st.session_state.web_search_enabled = True
    if "current_page" not in st.session_state:
        st.session_state.current_page = "[CHAT] 对话"
    
    # -------- 登录尝试状态 --------
    if "login_mode" not in st.session_state:
        st.session_state.login_mode = "password"  # password / code


# ==================== 6. 登录页面 ====================
def render_login_page():
    """渲染登录页面（密码登录 + 验证码登录）"""
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown('<div class="auth-container">', unsafe_allow_html=True)
        st.markdown('<p class="auth-title">[LOCK] AI CRM 登录</p>', unsafe_allow_html=True)
        st.markdown('<p class="auth-subtitle">智能客户经营平台 · 安全认证</p>', unsafe_allow_html=True)
        
        # -------- 登录方式切换 --------
        login_mode = st.session_state.login_mode
        tab1, tab2 = st.tabs(["[KEY] 密码登录", "[PHONE] 验证码登录"])
        
        with tab1:
            render_password_login()
        
        with tab2:
            render_code_login()
        
        st.markdown('<hr class="auth-divider">', unsafe_allow_html=True)
        
        # -------- 辅助操作 --------
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            if st.button("[MEMO] 注册账号", use_container_width=True):
                st.session_state.auth_page = "register"
                st.rerun()
        with col_b:
            if st.button("[LOCK] 忘记密码", use_container_width=True):
                st.session_state.auth_page = "forgot"
                st.rerun()
        with col_c:
            pass
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # 底部信息
        st.markdown(
            '<p class="auth-footer">[IDEA] 首次使用？联系管理员获取账号 · 建议使用 Chrome 浏览器</p>',
            unsafe_allow_html=True,
        )


def render_password_login():
    """密码登录表单"""
    auth = st.session_state.backend["auth"]
    if auth is None:
        st.error("[FAIL] 认证服务未初始化")
        return
    
    with st.form("password_login_form"):
        username = st.text_input(
            "[USER] 用户名 / 手机号",
            placeholder="请输入用户名或手机号",
            key="login_username",
        )
        password = st.text_input(
            "[KEY] 密码",
            type="password",
            placeholder="请输入密码",
            key="login_password",
        )
        
        col1, col2 = st.columns([1, 1])
        with col1:
            submitted = st.form_submit_button("[UNLOCK] 登 录", type="primary", use_container_width=True)
        with col2:
            if st.form_submit_button("[CLEAN] 清空", use_container_width=True):
                pass
        
        if submitted:
            if not username.strip():
                st.error("[WARN] 请输入用户名或手机号")
            elif not password:
                st.error("[WARN] 请输入密码")
            else:
                success, msg, account = auth.login(
                    username_or_phone=username.strip(),
                    password=password,
                )
                if success and account:
                    st.session_state.authenticated = True
                    st.session_state.current_account = account
                    st.session_state.messages = []
                    # ===== 生成持久化登录 Token =====
                    db = st.session_state.backend.get("db")
                    if db:
                        import uuid
                        token = str(uuid.uuid4())
                        db.save_session_token(token, account.account_id, expires_days=7)
                        st.session_state.session_token = token
                        st.query_params["token"] = token
                        _load_account_sessions(db, account.account_id)
                    logger.info(f"[OK] 用户登录成功: {account.username}")
                    st.rerun()
                else:
                    st.error(f"[FAIL] {msg}")


def render_code_login():
    """验证码登录表单"""
    auth = st.session_state.backend["auth"]
    if auth is None:
        st.error("[FAIL] 认证服务未初始化")
        return
    
    with st.form("code_login_form"):
        phone = st.text_input(
            "[PHONE] 手机号",
            placeholder="请输入注册时使用的手机号",
            max_chars=11,
            key="code_login_phone",
        )
        
        col_code, col_btn = st.columns([3, 1])
        with col_code:
            code = st.text_input(
                "[ENVELOPE] 验证码",
                placeholder="6位数字验证码",
                max_chars=6,
                key="code_login_code",
            )
        with col_btn:
            st.markdown("<br>", unsafe_allow_html=True)
            send_clicked = st.form_submit_button("[MAIL] 发送验证码", use_container_width=True)
        
        submitted = st.form_submit_button("[UNLOCK] 登 录", type="primary", use_container_width=True)
        
        if send_clicked:
            if not phone.strip() or len(phone.strip()) != 11:
                st.error("[WARN] 请输入正确的11位手机号")
            else:
                success, msg = auth.send_verification_code(phone.strip())
                if success:
                    st.success(f"[OK] {msg}")
                else:
                    st.error(f"[FAIL] {msg}")
        
        if submitted:
            if not phone.strip():
                st.error("[WARN] 请输入手机号")
            elif not code.strip():
                st.error("[WARN] 请输入验证码")
            else:
                success, msg, account = auth.login_with_code(
                    phone=phone.strip(),
                    code=code.strip(),
                )
                if success and account:
                    st.session_state.authenticated = True
                    st.session_state.current_account = account
                    st.session_state.messages = []
                    # ===== 生成持久化登录 Token =====
                    db = st.session_state.backend.get("db")
                    if db:
                        import uuid
                        token = str(uuid.uuid4())
                        db.save_session_token(token, account.account_id, expires_days=7)
                        st.session_state.session_token = token
                        st.query_params["token"] = token
                        _load_account_sessions(db, account.account_id)
                    logger.info(f"[OK] 验证码登录成功: {account.username}")
                    st.rerun()
                else:
                    st.error(f"[FAIL] {msg}")


# ==================== 7. 注册页面 ====================
def render_register_page():
    """渲染注册页面"""
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown('<div class="auth-container">', unsafe_allow_html=True)
        st.markdown('<p class="auth-title">[MEMO] 注册新账号</p>', unsafe_allow_html=True)
        st.markdown('<p class="auth-subtitle">创建您的 AI CRM 账号，开启智能客户经营之旅</p>', unsafe_allow_html=True)
        
        auth = st.session_state.backend["auth"]
        if auth is None:
            st.error("[FAIL] 认证服务未初始化")
            return
        
        with st.form("register_form"):
            username = st.text_input(
                "[USER] 用户名",
                placeholder="4-20位字母数字组合",
                key="reg_username",
            )
            
            col_ph, col_em = st.columns(2)
            with col_ph:
                phone = st.text_input(
                    "[PHONE] 手机号",
                    placeholder="11位手机号",
                    max_chars=11,
                    key="reg_phone",
                )
            with col_em:
                email = st.text_input(
                    "[EMAIL] 邮箱（可选）",
                    placeholder="example@email.com",
                    key="reg_email",
                )
            
            password = st.text_input(
                "[KEY] 密码",
                type="password",
                placeholder="至少8位，含大小写字母和数字",
                key="reg_password",
            )
            confirm_pwd = st.text_input(
                "[KEY] 确认密码",
                type="password",
                placeholder="再次输入密码",
                key="reg_confirm",
            )
            
            # 密码强度提示
            st.markdown(
                '<p style="font-size:0.8rem;color:#888;">'
                '[IDEA] 密码要求：至少8位，包含大写字母、小写字母和数字</p>',
                unsafe_allow_html=True,
            )
            
            agree = st.checkbox("我已阅读并同意《用户服务协议》和《隐私政策》")
            
            col1, col2 = st.columns(2)
            with col1:
                submitted = st.form_submit_button("[OK] 注 册", type="primary", use_container_width=True)
            with col2:
                if st.form_submit_button("[BACK] 返回登录", use_container_width=True):
                    st.session_state.auth_page = "login"
                    st.rerun()
            
            if submitted:
                if not username.strip():
                    st.error("[WARN] 请输入用户名")
                elif not phone.strip() or len(phone.strip()) != 11:
                    st.error("[WARN] 请输入正确的11位手机号")
                elif not password:
                    st.error("[WARN] 请输入密码")
                elif password != confirm_pwd:
                    st.error("[WARN] 两次输入的密码不一致")
                elif not agree:
                    st.error("[WARN] 请阅读并同意用户协议")
                else:
                    success, msg, account = auth.register(
                        username=username.strip(),
                        phone=phone.strip(),
                        password=password,
                        email=email.strip() if email.strip() else None,
                    )
                    if success:
                        st.success("[OK] 注册成功！请返回登录")
                        st.balloons()
                        # 3秒后自动跳转登录页
                        st.session_state.auth_page = "login"
                        st.rerun()
                    else:
                        st.error(f"[FAIL] {msg}")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown(
            '<p class="auth-footer">[LOCK] 您的信息将受到严格保护，不会泄露给第三方</p>',
            unsafe_allow_html=True,
        )


# ==================== 8. 忘记密码页面 ====================
def render_forgot_password_page():
    """渲染忘记密码页面"""
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown('<div class="auth-container">', unsafe_allow_html=True)
        st.markdown('<p class="auth-title">[LOCK] 重置密码</p>', unsafe_allow_html=True)
        st.markdown('<p class="auth-subtitle">通过手机验证码重置您的密码</p>', unsafe_allow_html=True)
        
        auth = st.session_state.backend["auth"]
        if auth is None:
            st.error("[FAIL] 认证服务未初始化")
            return
        
        # Step 1: 输入手机号 + 获取验证码
        with st.form("forgot_password_form"):
            phone = st.text_input(
                "[PHONE] 手机号",
                placeholder="请输入注册时使用的手机号",
                max_chars=11,
                key="forgot_phone",
            )
            
            col_code, col_btn = st.columns([3, 1])
            with col_code:
                code = st.text_input(
                    "[ENVELOPE] 验证码",
                    placeholder="6位数字验证码",
                    max_chars=6,
                    key="forgot_code",
                )
            with col_btn:
                st.markdown("<br>", unsafe_allow_html=True)
                send_clicked = st.form_submit_button("[MAIL] 获取", use_container_width=True)
            
            new_password = st.text_input(
                "[KEY] 新密码",
                type="password",
                placeholder="至少8位，含大小写字母和数字",
                key="forgot_new_pwd",
            )
            confirm_pwd = st.text_input(
                "[KEY] 确认新密码",
                type="password",
                placeholder="再次输入新密码",
                key="forgot_confirm_pwd",
            )
            
            st.markdown(
                '<p style="font-size:0.8rem;color:#888;">'
                '[IDEA] 密码要求：至少8位，包含大写字母、小写字母和数字</p>',
                unsafe_allow_html=True,
            )
            
            col1, col2 = st.columns(2)
            with col1:
                submitted = st.form_submit_button("[OK] 确认重置", type="primary", use_container_width=True)
            with col2:
                if st.form_submit_button("[BACK] 返回登录", use_container_width=True):
                    st.session_state.auth_page = "login"
                    st.rerun()
            
            if send_clicked:
                if not phone.strip() or len(phone.strip()) != 11:
                    st.error("[WARN] 请输入正确的11位手机号")
                else:
                    success, msg = auth.send_verification_code(phone.strip())
                    if success:
                        st.success(f"[OK] {msg}")
                    else:
                        st.error(f"[FAIL] {msg}")
            
            if submitted:
                if not phone.strip():
                    st.error("[WARN] 请输入手机号")
                elif not code.strip():
                    st.error("[WARN] 请输入验证码")
                elif not new_password:
                    st.error("[WARN] 请输入新密码")
                elif new_password != confirm_pwd:
                    st.error("[WARN] 两次输入的密码不一致")
                else:
                    success, msg = auth.reset_password(
                        phone=phone.strip(),
                        code=code.strip(),
                        new_password=new_password,
                    )
                    if success:
                        st.success("[OK] 密码重置成功！请使用新密码登录")
                        st.session_state.auth_page = "login"
                        st.rerun()
                    else:
                        st.error(f"[FAIL] {msg}")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown(
            '<p class="auth-footer">[LOCK] 重置密码后，旧密码将立即失效</p>',
            unsafe_allow_html=True,
        )


# ==================== 9. 认证后的侧边栏 ====================
def render_authenticated_sidebar():
    """渲染登录后的侧边栏（含角色信息 + 多对话管理）"""
    account = st.session_state.current_account
    if account is None:
        return
    
    with st.sidebar:
        # -------- 用户信息 --------
        role_badge = "[SHIELD] 管理员" if account.role == "admin" else "[USER] 普通用户"
        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, #667eea22, #764ba222);
                padding: 15px;
                border-radius: 12px;
                margin-bottom: 1rem;
                border: 1px solid #667eea33;
            ">
                <div style="font-size:1.1rem;font-weight:600;">{account.username}</div>
                <div style="font-size:0.85rem;color:#666;margin-top:4px;">{role_badge}</div>
                <div style="font-size:0.8rem;color:#888;margin-top:2px;">[PHONE] {account.phone}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        # -------- 系统状态 --------
        st.markdown("### [AI] 系统状态")
        backend = st.session_state.backend
        
        col1, col2 = st.columns(2)
        with col1:
            llm_ok = backend["llm"] is not None
            st.markdown(f"LLM: {'[ONLINE] 在线' if llm_ok else '[OFFLINE] 离线'}")
        with col2:
            db_ok = backend["db"] is not None
            st.markdown(f"数据库: {'[ONLINE] 在线' if db_ok else '[OFFLINE] 离线'}")
        
        col3, col4 = st.columns(2)
        with col3:
            st.markdown(f"RAG: {'[ONLINE] 就绪' if backend['rag_ready'] else '[NONE] 未加载'}")
        with col4:
            st.markdown(f"搜索: {'[ONLINE] 就绪' if backend['search_ready'] else '[NONE] 未加载'}")
        
        st.divider()
        
        # ========== 多对话管理 ==========
        st.markdown("### [CHAT] 对话会话")
        
        # 新建对话按钮
        if st.button("[+] 新建对话", type="primary", use_container_width=True):
            from models import ChatSession
            db = backend.get("db")
            if db:
                new_session = ChatSession(
                    account_id=account.account_id,
                    title="新对话",
                )
                db.save_session(new_session)
                st.session_state.sessions.append({
                    "session_id": new_session.session_id,
                    "title": new_session.title,
                })
                st.session_state.session_messages[new_session.session_id] = []
                st.session_state.current_session_id = new_session.session_id
                st.session_state.messages = []
                st.rerun()
        
        # 会话列表（用 radio 显示，当前高亮）
        if st.session_state.sessions:
            session_options = {
                s["session_id"]: s["title"][:20] + ("..." if len(s["title"]) > 20 else "")
                for s in st.session_state.sessions
            }
            current_id = st.session_state.current_session_id
            
            # 找到当前会话在列表中的索引
            ids = [s["session_id"] for s in st.session_state.sessions]
            default_idx = 0
            if current_id in ids:
                default_idx = ids.index(current_id)
            
            selected_id = st.radio(
                "切换会话",
                options=ids,
                format_func=lambda x: session_options.get(x, x[:8]),
                index=default_idx,
                label_visibility="collapsed",
            )
            
            if selected_id != current_id:
                st.session_state.current_session_id = selected_id
                # 切换时把对应会话的消息加载到 st.session_state.messages
                if selected_id in st.session_state.session_messages:
                    st.session_state.messages = st.session_state.session_messages[selected_id].copy()
                else:
                    st.session_state.messages = []
                st.rerun()
            
            # 删除当前会话按钮
            if len(st.session_state.sessions) > 1:
                if st.button("🗑️ 删除当前对话", type="secondary", use_container_width=True):
                    db = backend.get("db")
                    if db:
                        db.delete_session(current_id)
                    # 从状态中移除
                    st.session_state.sessions = [
                        s for s in st.session_state.sessions
                        if s["session_id"] != current_id
                    ]
                    if current_id in st.session_state.session_messages:
                        del st.session_state.session_messages[current_id]
                    # 切换到第一个会话
                    if st.session_state.sessions:
                        first_id = st.session_state.sessions[0]["session_id"]
                        st.session_state.current_session_id = first_id
                        st.session_state.messages = st.session_state.session_messages.get(first_id, []).copy()
                    st.rerun()
        else:
            st.info("暂无会话，点击上方新建")
        
        st.caption(f"当前会话: {len(st.session_state.messages)} 条消息")
        
        st.divider()
        
        # -------- 页面导航 --------
        st.markdown("### [FOLDER] 导航")
        
        # 普通用户可见页面（知识库管理仅管理员可见）
        user_pages = ["[CHAT] 对话", "[USERS] 客户信息", "[PHONE] 跟进", "[CHART] 统计"]
        admin_pages = [
            "[USERS] 客户信息", "[PHONE] 跟进", "[BOOKS] 知识库", "[CHART] 统计",
            "[SHIELD] 用户管理", "[MONITOR] 监控大盘",
        ]
        
        if account.role == "admin":
            available_pages = ["[CHAT] 对话"] + admin_pages
        else:
            available_pages = user_pages
        
        selected = st.radio(
            "选择页面",
            available_pages,
            index=available_pages.index(st.session_state.current_page)
            if st.session_state.current_page in available_pages
            else 0,
            label_visibility="collapsed",
        )
        st.session_state.current_page = selected
        
        st.divider()
        
        # -------- 联网搜索开关 --------
        st.markdown("### [GLOBE] 联网搜索")
        st.session_state.web_search_enabled = st.toggle(
            "启用联网搜索",
            value=st.session_state.web_search_enabled,
            help="开启后 AI 可联网查询实时信息",
        )
        if not backend["search_ready"]:
            st.warning("搜索工具未加载")
        
        st.divider()
        
        # -------- 安全退出 --------
        st.markdown("### [LOCK] 账户")
        if st.button("[DOOR] 退出登录", type="secondary", use_container_width=True):
            # 删除持久化 token
            db = backend.get("db")
            token = st.session_state.session_token
            if db and token:
                db.delete_session_token(token)
            st.session_state.authenticated = False
            st.session_state.current_account = None
            st.session_state.session_token = None
            st.session_state.messages = []
            st.session_state.sessions = []
            st.session_state.current_session_id = None
            st.session_state.session_messages = {}
            st.session_state.current_page = "[CHAT] 对话"
            st.query_params.clear()
            st.rerun()


# ==================== 10. 对话页面 ====================
def render_chat_page():
    """渲染对话页面（基于 session_id 的消息隔离）"""
    st.markdown('<p class="main-title">[CHAT] 智能客户经营助手</p>', unsafe_allow_html=True)
    st.caption("基于 LangGraph 工作流 + 通义千问大模型，智能理解客户需求")
    
    session_id = st.session_state.current_session_id
    if session_id is None:
        st.info("[LEFT] 请在侧边栏新建或选择一个对话")
        return
    
    display_messages = st.session_state.session_messages.get(session_id, [])
    
    chat_container = st.container()
    
    with chat_container:
        for msg in display_messages:
            if msg["role"] == "user":
                st.markdown(
                    f'<div class="chat-user">[PERSON] **你:** {msg["content"]}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="chat-assistant">[AI] **AI:** {msg["content"]}</div>',
                    unsafe_allow_html=True,
                )
        
        if not display_messages:
            st.info(
                "[WAVE] 你好！我是你的 AI 客户经营助手。我可以：\n\n"
                "• 回答客户咨询 [CHAT]\n"
                "• 查询客户信息 [USERS]\n"
                "• 检索知识库文档 [BOOKS]\n"
                "• 联网搜索实时信息 [GLOBE]\n\n"
                "**在下方输入框开始对话吧！**"
            )
    
    st.divider()
    
    col_input, col_btn = st.columns([5, 1])
    
    with col_input:
        user_input = st.text_input(
            "输入您的问题",
            key="user_input",
            placeholder="例如：你好，我想查询客户信息",
            label_visibility="collapsed",
        )
    
    with col_btn:
        send_clicked = st.button("[OUTBOX] 发送", type="primary", use_container_width=True)
    
    if send_clicked and user_input.strip():
        backend = st.session_state.backend
        db = backend.get("db")
        account = st.session_state.current_account
        
        if backend["workflow"] is None:
            st.error("[FAIL] 后端组件未初始化，请检查配置")
            return
        
        # ===== 添加到当前会话的消息列表 =====
        user_msg = {"role": "user", "content": user_input.strip()}
        if session_id not in st.session_state.session_messages:
            st.session_state.session_messages[session_id] = []
        st.session_state.session_messages[session_id].append(user_msg)
        st.session_state.messages = st.session_state.session_messages[session_id].copy()
        
        # ===== 自动更新对话标题（用第一条用户消息） =====
        if db and session_id and st.session_state.sessions:
            for s in st.session_state.sessions:
                if s["session_id"] == session_id and s["title"] == "新对话":
                    new_title = user_input.strip()[:30]
                    db.update_session_title(session_id, new_title)
                    s["title"] = new_title
                    db.update_session_time(session_id)
                    break
        
        customer_info = {}
        try:
            customers = db.list_all_customers() if db else []
            if customers:
                c = customers[0]
                customer_info = {
                    "name": c.name,
                    "company": c.company or "",
                    "position": c.position or "",
                    "phone": c.phone or "",
                    "email": c.email or "",
                    "tags": c.tags or [],
                    "notes": c.notes or "",
                }
                logger.info(f"[USER] 已关联客户画像: {c.name}")
        except Exception as e:
            logger.warning(f"[WARN] 加载客户画像失败: {e}")
        
        with st.spinner("🤔 AI 正在思考..."):
            try:
                user_id = account.account_id if account else "streamlit_user"
                result = backend["workflow"].run(
                    query=user_input.strip(),
                    user_id=user_id,
                    customer_info=customer_info,
                    session_id=session_id,
                )
                
                assistant_msg = {"role": "assistant", "content": result}
                st.session_state.session_messages[session_id].append(assistant_msg)
                st.session_state.messages = st.session_state.session_messages[session_id].copy()
                
                if db and session_id:
                    db.update_session_time(session_id)
                    
            except Exception as e:
                error_msg = f"[FAIL] 处理出错: {e}"
                st.session_state.session_messages[session_id].append({
                    "role": "assistant",
                    "content": error_msg,
                })
                st.session_state.messages = st.session_state.session_messages[session_id].copy()
                logger.error(f"工作流执行失败: {e}")
        
        st.rerun()
    
    elif send_clicked and not user_input.strip():
        st.warning("[WARN] 请输入问题")


# ==================== 11. 客户信息页面 ====================
def render_customer_page():
    """渲染客户信息管理页面"""
    st.markdown('<p class="main-title">[USERS] 客户信息管理</p>', unsafe_allow_html=True)
    st.caption("查看和管理客户画像数据")
    
    backend = st.session_state.backend
    
    if backend["db"] is None:
        st.error("[FAIL] 数据库未初始化")
        return
    
    # -------- 11.1 新增客户 --------
    st.subheader("[+] 新增客户")
    
    with st.expander("点击展开新增客户表单", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            new_name = st.text_input("客户姓名 *", placeholder="必填", key="new_cust_name")
            new_company = st.text_input("公司", placeholder="可选", key="new_cust_company")
            new_position = st.text_input("职位", placeholder="可选", key="new_cust_position")
        
        with col2:
            new_phone = st.text_input("电话", placeholder="可选", key="new_cust_phone")
            new_email = st.text_input("邮箱", placeholder="可选", key="new_cust_email")
            new_tags = st.text_input("标签（逗号分隔）", placeholder="例如：高价值,意向客户", key="new_cust_tags")
        
        new_notes = st.text_area("备注", placeholder="可选", key="new_cust_notes")
        
        if st.button("[DISK] 保存客户", type="primary", use_container_width=True):
            if not new_name.strip():
                st.warning("[WARN] 客户姓名不能为空")
            else:
                try:
                    from models import CustomerProfile
                    
                    profile = CustomerProfile(
                        name=new_name.strip(),
                        company=new_company.strip() or None,
                        position=new_position.strip() or None,
                        phone=new_phone.strip() or None,
                        email=new_email.strip() or None,
                        notes=new_notes.strip() or None,
                        tags=[t.strip() for t in new_tags.split(",") if t.strip()] if new_tags else [],
                    )
                    
                    if backend["db"].save_customer(profile):
                        st.success(f"[OK] 客户 '{profile.name}' 已保存")
                        for key in ["new_cust_name", "new_cust_company", "new_cust_position",
                                     "new_cust_phone", "new_cust_email", "new_cust_tags", "new_cust_notes"]:
                            if key in st.session_state:
                                del st.session_state[key]
                        st.rerun()
                    else:
                        st.error("[FAIL] 保存失败")
                except Exception as e:
                    st.error(f"[FAIL] 保存失败: {e}")
    
    st.divider()
    
    # -------- 11.2 客户列表 --------
    st.subheader("[CLIPBOARD] 客户列表")
    
    try:
        customers = backend["db"].list_all_customers()
        
        if customers:
            table_data = []
            for c in customers:
                table_data.append({
                    "客户名称": c.name,
                    "公司": c.company or "-",
                    "职位": c.position or "-",
                    "电话": c.phone or "-",
                    "邮箱": c.email or "-",
                    "标签": ", ".join(c.tags) if c.tags else "-",
                    "备注": (c.notes or "-")[:20],
                })
            
            st.dataframe(
                table_data,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "客户名称": st.column_config.TextColumn("客户名称", width="medium"),
                    "公司": st.column_config.TextColumn("公司", width="medium"),
                    "标签": st.column_config.TextColumn("标签", width="medium"),
                },
            )
            
            st.caption(f"共 {len(customers)} 个客户")
        else:
            st.info("📭 暂无客户数据，请通过上方表单添加")
    
    except Exception as e:
        st.error(f"[FAIL] 查询客户失败: {e}")
    
    st.divider()
    
    # -------- 11.3 自然语言查询 --------
    st.subheader("[SEARCH] 自然语言查客户")
    
    nl_query = st.text_input(
        "输入查询描述",
        placeholder="例如：查询标签包含'高价值'的客户",
        label_visibility="collapsed",
        key="nl_customer_query",
    )
    
    if st.button("查询", key="query_customer", type="primary"):
        if nl_query:
            try:
                from sql_agent import SQLAgent
                agent = SQLAgent(st.session_state.backend["llm"])
                result = agent.query(nl_query)
                
                st.markdown("**查询结果：**")
                st.code(result, language="text")
            except Exception as e:
                st.error(f"[FAIL] 查询失败: {e}")
        else:
            st.warning("[WARN] 请输入查询描述")


# ==================== 11b. 跟进管理页面（B.4 · 跟进闭环产品化落地）====================
def render_follow_up_page():
    """
    渲染跟进管理页面
    ==================
    三个区块：
      1. 记录跟进   —— 结构化写入（谁 / 何时 / 什么方式 / 结果 / 下次时间）
      2. 待跟进提醒 —— 到期巡检（计划到期 + 超期未跟进）
      3. 转化统计   —— 分子 / 分母口径明确，无数据返回 null 而非 0
    """
    st.markdown('<p class="main-title">[PHONE] 跟进管理</p>', unsafe_allow_html=True)
    st.caption("跟进记录 → 回写画像「最近跟进时间」→ 异常筛查 R1/R2 自动复位（闭环）")

    backend = st.session_state.backend
    db = backend.get("db")
    if db is None:
        st.error("[FAIL] 数据库未初始化")
        return

    try:
        customers = db.list_all_customers()
    except Exception as e:
        st.error(f"[FAIL] 读取客户列表失败: {e}")
        return

    tab_add, tab_due, tab_stats = st.tabs(
        ["[EDIT] 记录跟进", "[ALARM] 待跟进提醒", "[CHART] 转化统计"]
    )

    # -------- 11b.1 记录跟进 --------
    with tab_add:
        st.subheader("[+] 新增跟进记录")

        if not customers:
            st.info("📭 暂无客户，请先到「客户信息」页新增客户")
        else:
            from datetime import datetime, timedelta
            from models import FOLLOW_UP_CHANNELS, FOLLOW_UP_RESULTS, FollowUp

            options = {
                f"{c.name}（{c.customer_id[:8]}…）": c.customer_id for c in customers
            }
            picked_label = st.selectbox(
                "选择客户 *", list(options.keys()), key="fu_customer"
            )
            picked_id = options.get(picked_label)

            col1, col2, col3 = st.columns(3)
            with col1:
                channel = st.selectbox("跟进方式 *", FOLLOW_UP_CHANNELS, key="fu_channel")
            with col2:
                result = st.selectbox("跟进结果 *", FOLLOW_UP_RESULTS, key="fu_result")
            with col3:
                next_days = st.number_input(
                    "N 天后再跟进（0 = 不设提醒）",
                    min_value=0, max_value=365, value=0, step=1, key="fu_next_days",
                )

            content = st.text_area(
                "跟进内容摘要",
                placeholder="例如：沟通了秋季班课包，家长关注课时单价",
                key="fu_content",
            )
            advisor_id = st.text_input("跟进人 ID", placeholder="可选", key="fu_advisor")

            if st.button("[DISK] 保存跟进记录", type="primary", use_container_width=True):
                if not picked_id:
                    st.warning("[WARN] 请选择客户")
                else:
                    next_at = (
                        datetime.now() + timedelta(days=int(next_days))
                        if next_days else None
                    )
                    record = FollowUp(
                        customer_id=picked_id,
                        advisor_id=advisor_id.strip() or None,
                        channel=channel,
                        content=content.strip() or None,
                        result=result,
                        next_follow_up_at=next_at,
                    )
                    if db.save_follow_up(record):
                        tips = f"[OK] 已记录：{picked_label}（{channel} / {result}）"
                        if next_at:
                            tips += f"，下次跟进 {next_at:%Y-%m-%d}"
                        st.success(tips)
                        st.rerun()
                    else:
                        st.error("[FAIL] 保存失败")

        st.divider()
        st.subheader("[CLIPBOARD] 最近跟进记录")
        try:
            records = db.list_follow_ups(limit=50)
            if records:
                st.dataframe(
                    [
                        {
                            "时间": (
                                r.created_at.strftime("%Y-%m-%d %H:%M")
                                if r.created_at else "-"
                            ),
                            "客户ID": (r.customer_id[:8] + "…") if r.customer_id else "-",
                            "方式": r.channel or "-",
                            "结果": r.result or "-",
                            "内容": (r.content or "-")[:30],
                            "下次跟进": (
                                r.next_follow_up_at.strftime("%Y-%m-%d")
                                if r.next_follow_up_at else "-"
                            ),
                        }
                        for r in records
                    ],
                    use_container_width=True,
                    hide_index=True,
                )
                st.caption(f"共 {len(records)} 条跟进记录")
            else:
                st.info("📭 暂无跟进记录")
        except Exception as e:
            st.error(f"[FAIL] 查询跟进记录失败: {e}")

    # -------- 11b.2 待跟进提醒 --------
    with tab_due:
        st.subheader("[ALARM] 待跟进提醒（到期巡检）")

        col1, col2 = st.columns([1, 3])
        with col1:
            due_days = st.number_input(
                "超期阈值（天）", min_value=1, max_value=180,
                value=14, step=1, key="fu_due_days",
            )
        with col2:
            st.caption(
                "两类口径：① 计划到期 = 记录里填的「下次跟进时间」已到点且此后无新跟进；"
                "② 超期未跟进 = 画像最近跟进时间超过阈值，或从未跟进。"
            )

        if st.button("[REFRESH] 立即巡检", type="primary", key="fu_scan"):
            st.session_state["fu_due_done"] = True

        if st.session_state.get("fu_due_done"):
            try:
                items = db.list_due_follow_ups(due_days=int(due_days), limit=200)
                if not items:
                    st.success("[OK] 暂无到期 / 超期未跟进客户")
                else:
                    planned = [i for i in items if i.get("due_type") == "计划到期"]
                    overdue = [i for i in items if i.get("due_type") == "超期未跟进"]
                    c1, c2, c3 = st.columns(3)
                    c1.metric("待跟进总数", len(items))
                    c2.metric("计划到期", len(planned))
                    c3.metric("超期未跟进", len(overdue))
                    st.dataframe(
                        [
                            {
                                "类型": i["due_type"],
                                "客户": i["name"],
                                "客户ID": i["customer_id"],
                                "距上次跟进": (
                                    f"{i['days']} 天"
                                    if i.get("days") is not None else "从未跟进"
                                ),
                                "原因": "；".join(i.get("reasons") or []),
                            }
                            for i in items
                        ],
                        use_container_width=True,
                        hide_index=True,
                    )
            except Exception as e:
                st.error(f"[FAIL] 巡检失败: {e}")

    # -------- 11b.3 转化统计 --------
    with tab_stats:
        st.subheader("[CHART] 跟进转化统计")

        days = st.number_input(
            "统计窗口（天）", min_value=1, max_value=365,
            value=30, step=1, key="fu_stats_days",
        )
        try:
            stats = db.follow_up_conversion_stats(days=int(days))
            rate = stats.get("conversion_rate")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("跟进记录数（分母）", stats["total"])
            c2.metric("转化跟进数（分子）", stats["converted"])
            c3.metric("跟进转化率", f"{rate * 100:.1f}%" if rate is not None else "无数据")
            c4.metric("覆盖客户数", stats["covered_customers"])

            if stats.get("by_result"):
                st.dataframe(
                    [{"跟进结果": k, "条数": v} for k, v in stats["by_result"].items()],
                    use_container_width=True,
                    hide_index=True,
                )
            st.caption(
                "口径：分母 = 窗口内跟进记录条数；分子 = 结果为「已成交 / 有意向」的条数；"
                "无数据时显示「无数据」而不是 0（不报 0 冒充）。"
            )
        except Exception as e:
            st.error(f"[FAIL] 统计失败: {e}")


# ==================== 12. 知识库管理页面 ====================
def render_knowledge_page():
    """渲染知识库管理页面"""
    st.markdown('<p class="main-title">[BOOKS] 知识库管理</p>', unsafe_allow_html=True)
    st.caption("管理 RAG 知识库文档，支持上传和检索")
    
    backend = st.session_state.backend
    
    # -------- 12.1 知识库状态 --------
    st.subheader("[CHART] 知识库状态")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if backend["rag_ready"]:
            try:
                stats = backend["rag"].get_stats()
                st.metric("[DOC] 文档总数", stats.get("total_docs", 0))
            except:
                st.metric("[DOC] 文档总数", "未知")
        else:
            st.warning("RAG 未加载")
    
    with col2:
        st.metric("[FOLDER] 存储位置", "utils/chroma_kb")
    
    with col3:
        st.metric("[SEARCH] 检索模式", "混合检索（向量+BM25）")
    
    st.divider()
    
    # -------- 12.2 文件上传 --------
    st.subheader("[OUTBOX] 上传文档到知识库")
    
    uploaded_file = st.file_uploader(
        "选择文件（.txt / .md）",
        type=["txt", "md"],
        help="支持纯文本和 Markdown 文件",
    )
    
    if uploaded_file is not None:
        if backend["rag_ready"]:
            upload_dir = "uploads"
            os.makedirs(upload_dir, exist_ok=True)
            
            file_path = os.path.join(upload_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            with st.spinner("[BOOK] 正在处理文档..."):
                try:
                    chunks = backend["rag"].add_file(file_path)
                    st.success(f"[OK] 文档已添加: {chunks} 个文档块")
                except Exception as e:
                    st.error(f"[FAIL] 添加失败: {e}")
        else:
            st.error("[FAIL] RAG 知识库未加载，无法上传文档")
    
    st.divider()
    
    # -------- 12.3 检索测试 --------
    st.subheader("[SEARCH] 检索测试")
    
    test_query = st.text_input(
        "输入检索关键词",
        placeholder="例如：什么是智能客服",
        label_visibility="collapsed",
        key="rag_test_query",
    )
    
    if st.button("检索", key="test_search", type="primary"):
        if test_query and backend["rag_ready"]:
            with st.spinner("[SEARCH] 正在检索..."):
                try:
                    results = backend["rag"].hybrid_search(test_query, top_k=3)
                    
                    if results:
                        st.markdown(f"**找到 {len(results)} 条结果：**")
                        for i, doc in enumerate(results, 1):
                            with st.expander(f"结果 {i}"):
                                st.write(doc)
                    else:
                        st.info("没有找到相关结果")
                except Exception as e:
                    st.error(f"[FAIL] 检索失败: {e}")
        elif not backend["rag_ready"]:
            st.warning("RAG 知识库未加载")
        else:
            st.warning("[WARN] 请输入检索关键词")


# ==================== 13. 统计页面 ====================
def render_stats_page():
    """渲染统计页面"""
    st.markdown('<p class="main-title">[CHART] 统计信息</p>', unsafe_allow_html=True)
    
    backend = st.session_state.backend
    
    st.subheader("[CHART_UP] Token 用量统计")
    
    if backend["llm"]:
        try:
            cost_str = backend["llm"].get_token_cost_str()
            st.info(cost_str)
        except Exception as e:
            st.error(f"[FAIL] 获取 Token 统计失败: {e}")
    else:
        st.warning("LLM 未初始化")
    
    st.divider()
    
    # -------- 13.1 系统信息 --------
    st.subheader("ℹ️ 系统信息")
    
    account = st.session_state.current_account
    username = account.username if account else "未知"
    role = "管理员" if account and account.role == "admin" else "普通用户"
    
    info_data = {
        "当前用户": f"{username}（{role}）",
        "联网搜索": "已开启" if st.session_state.web_search_enabled else "已关闭",
        "LLM 状态": "[OK] 已加载" if backend["llm"] else "[FAIL] 未加载",
        "数据库状态": "[OK] 已加载" if backend["db"] else "[FAIL] 未加载",
        "RAG 状态": "[OK] 已加载" if backend["rag_ready"] else "[NONE] 未加载",
        "搜索工具状态": "[OK] 已加载" if backend["search_ready"] else "[NONE] 未加载",
    }
    
    for key, value in info_data.items():
        st.markdown(f"- **{key}**: {value}")


# ==================== 14. 管理员后台 - 用户管理 ====================
def render_admin_page():
    """渲染管理员后台页面（用户管理 + 登录审计）"""
    st.markdown('<p class="main-title">[SHIELD] 用户管理</p>', unsafe_allow_html=True)
    st.caption("管理员专属 · 账户管理与安全审计")
    
    backend = st.session_state.backend
    db = backend["db"]
    
    if db is None:
        st.error("[FAIL] 数据库未初始化")
        return
    
    # ===== 14.1 用户列表 =====
    st.subheader("[USERS] 账户列表")
    
    try:
        accounts = db.list_all_accounts()
        
        if accounts:
            table_data = []
            for acc in accounts:
                status = "[ONLINE] 激活" if acc.is_active else "[OFFLINE] 禁用"
                role_badge = "[SHIELD] 管理员" if acc.role == "admin" else "[USER] 用户"
                last_login = acc.last_login.strftime("%Y-%m-%d %H:%M") if acc.last_login else "从未登录"
                
                table_data.append({
                    "用户名": acc.username,
                    "角色": role_badge,
                    "手机号": acc.phone,
                    "状态": status,
                    "最后登录": last_login,
                    "创建时间": acc.created_at.strftime("%Y-%m-%d"),
                    "account_id": acc.account_id,
                })
            
            st.dataframe(
                table_data,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "用户名": st.column_config.TextColumn("用户名", width="small"),
                    "角色": st.column_config.TextColumn("角色", width="small"),
                    "手机号": st.column_config.TextColumn("手机号", width="medium"),
                    "状态": st.column_config.TextColumn("状态", width="small"),
                    "最后登录": st.column_config.TextColumn("最后登录", width="medium"),
                    "创建时间": st.column_config.TextColumn("创建时间", width="small"),
                },
                column_order=["用户名", "角色", "手机号", "状态", "最后登录", "创建时间"],
            )
            
            st.caption(f"共 {len(accounts)} 个账户")
        else:
            st.info("暂无账户数据")
    
    except Exception as e:
        st.error(f"[FAIL] 查询账户失败: {e}")
    
    st.divider()
    
    # ===== 14.2 账户操作 =====
    st.subheader("[GEAR] 账户管理")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**[LOCK] 启用/禁用账户**")
        accounts = db.list_all_accounts()
        if accounts:
            account_options = {f"{a.username} ({a.phone})": a.account_id for a in accounts}
            selected_label = st.selectbox(
                "选择账户",
                options=list(account_options.keys()),
                key="admin_select_account",
            )
            selected_id = account_options[selected_label]
            selected_acc = db.get_account_by_id(selected_id)
            
            if selected_acc:
                current_status = "[ONLINE] 激活" if selected_acc.is_active else "[OFFLINE] 禁用"
                st.markdown(f"当前状态：{current_status}")
                
                if selected_acc.role == "admin":
                    st.warning("[WARN] 管理员账户不可禁用")
                else:
                    if st.button(
                        "[DENIED] 禁用" if selected_acc.is_active else "[OK] 启用",
                        type="primary" if selected_acc.is_active else "secondary",
                        use_container_width=True,
                    ):
                        if db.update_account_status(selected_id, not selected_acc.is_active):
                            st.success(f"[OK] 账户状态已更新")
                            st.rerun()
                        else:
                            st.error("[FAIL] 操作失败")
    
    with col2:
        st.markdown("**[CHART] 登录日志**")
        if st.button("[CLIPBOARD] 查看最近的登录日志", use_container_width=True):
            logs = db.get_login_logs(limit=20)
            if logs:
                log_data = []
                for log in logs:
                    action_emoji = {
                        "login": "[OK] 登录成功",
                        "logout": "[DOOR] 登出",
                        "fail": "[FAIL] 登录失败",
                    }
                    log_data.append({
                        "时间": log.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                        "用户名": log.username,
                        "操作": action_emoji.get(log.action, log.action),
                        "详情": log.detail or "-",
                    })
                
                st.dataframe(
                    log_data,
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("暂无登录日志")
    
    st.divider()
    
    # ===== 14.3 密码修改（管理员自己） =====
    st.subheader("[KEY] 修改我的密码")
    
    auth = backend["auth"]
    if auth:
        with st.form("admin_change_pwd_form"):
            old_pwd = st.text_input("当前密码", type="password", key="admin_old_pwd")
            new_pwd = st.text_input("新密码", type="password", key="admin_new_pwd")
            confirm_pwd = st.text_input("确认新密码", type="password", key="admin_confirm_pwd")
            
            if st.form_submit_button("[OK] 修改密码", type="primary", use_container_width=True):
                if not old_pwd or not new_pwd or not confirm_pwd:
                    st.error("[WARN] 请填写所有字段")
                elif new_pwd != confirm_pwd:
                    st.error("[WARN] 两次输入的密码不一致")
                else:
                    account = st.session_state.current_account
                    if account:
                        success, msg = auth.change_password(
                            account_id=account.account_id,
                            old_password=old_pwd,
                            new_password=new_pwd,
                        )
                        if success:
                            st.success("[OK] 密码修改成功")
                        else:
                            st.error(f"[FAIL] {msg}")


# ==================== 15. 辅助函数：加载账户会话 ====================
def _load_account_sessions(db, account_id):
    """从数据库加载指定账户的所有会话到 session_state"""
    from models import ChatSession
    
    sessions = db.list_sessions_by_account(account_id)
    st.session_state.sessions = [
        {"session_id": s.session_id, "title": s.title} for s in sessions
    ]
    st.session_state.session_messages = {}
    for s in sessions:
        msgs = db.get_all_session_messages(s.session_id)
        st.session_state.session_messages[s.session_id] = [
            {"role": m.role, "content": m.content} for m in msgs
        ]
    
    if sessions:
        st.session_state.current_session_id = sessions[0].session_id
        st.session_state.messages = st.session_state.session_messages.get(
            sessions[0].session_id, []
        ).copy()
    else:
        new_session = ChatSession(account_id=account_id, title="新对话")
        db.save_session(new_session)
        st.session_state.sessions.append({
            "session_id": new_session.session_id,
            "title": new_session.title,
        })
        st.session_state.session_messages[new_session.session_id] = []
        st.session_state.current_session_id = new_session.session_id
        st.session_state.messages = []


# ==================== 16. 主函数 ====================
def main():
    """
    主函数
    流程：
      1. 初始化会话状态
      2. 尝试从 URL Token 自动恢复登录
      3. 检查认证状态
         - 未登录 → 显示登录/注册/忘记密码页面
         - 已登录 → 显示主应用（含角色控制）
    """
    # 初始化会话状态
    init_session_state()
    
    db = st.session_state.backend.get("db")
    
    # ===== 自动恢复登录（从 URL query_params 读取 token） =====
    if not st.session_state.authenticated and db:
        url_token = st.query_params.get("token")
        if url_token:
            account_id = db.get_account_id_by_token(url_token)
            if account_id:
                account = db.get_account_by_id(account_id)
                if account and account.is_active:
                    st.session_state.authenticated = True
                    st.session_state.current_account = account
                    st.session_state.session_token = url_token
                    _load_account_sessions(db, account_id)
                    logger.info(f"[REFRESH] Token 自动恢复登录: {account.username}")
    
    # 检查认证状态
    if not st.session_state.authenticated:
        # -------- 认证页面路由 --------
        auth_page = st.session_state.auth_page
        
        if auth_page == "login":
            render_login_page()
        elif auth_page == "register":
            render_register_page()
        elif auth_page == "forgot":
            render_forgot_password_page()
    else:
        # -------- 已登录：主应用 --------
        # 渲染侧边栏
        render_authenticated_sidebar()
        
        # 根据当前页面渲染内容
        page = st.session_state.current_page
        
        if page == "[CHAT] 对话":
            render_chat_page()
        elif page == "[USERS] 客户信息":
            render_customer_page()
        elif page == "[PHONE] 跟进":
            render_follow_up_page()
        elif page == "[BOOKS] 知识库":
            # 仅管理员可访问
            account = st.session_state.current_account
            if account and account.role == "admin":
                render_knowledge_page()
            else:
                st.error("[FAIL] 权限不足，仅管理员可管理知识库")
                st.session_state.current_page = "[CHAT] 对话"
                st.rerun()
        elif page == "[CHART] 统计":
            render_stats_page()
        elif page == "[SHIELD] 用户管理":
            # 仅管理员可访问
            account = st.session_state.current_account
            if account and account.role == "admin":
                render_admin_page()
            else:
                st.error("[FAIL] 权限不足，仅管理员可访问此页面")
                st.session_state.current_page = "[CHAT] 对话"
                st.rerun()
        elif page == "[MONITOR] 监控大盘":
            # 仅管理员可访问：系统运行指标 + 模型效果指标 + 评测数据管理
            account = st.session_state.current_account
            if account and account.role == "admin":
                from monitor_view import render_monitor_page
                render_monitor_page()
            else:
                st.error("[FAIL] 权限不足，仅管理员可访问此页面")
                st.session_state.current_page = "[CHAT] 对话"
                st.rerun()


# ==================== 17. 入口 ====================
if __name__ == "__main__":
    main()
