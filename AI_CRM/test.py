"""
test.py - 整体功能测试（主入口）
====================================
功能分区：
  1. 环境验证（检查依赖是否安装正确）
  2. 数据库测试（SQLite 建表 + CRUD）
  3. LLM 测试（通义千问 API 调用）
  4. LangGraph 工作流测试（完整端到端）
  5. SQL Agent 测试（自然语言查数据库）
  6. 网络搜索测试（DuckDuckGo）
  7. Token 用量统计
"""

# ==================== 1. 导入依赖 ====================
import os            # 文件路径操作
import sys           # 系统参数

# 将项目根目录加入 sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from logger import logger     # 日志记录器
from config import (
    QWEN_API_KEY,             # API Key（检查是否已配置）
    QWEN_MODEL_NAME,          # 模型名称
)


# ==================== 2. 环境验证 ====================
def test_environment():
    """
    测试1: 验证环境配置
    =====================
    检查内容：
      1. .env 文件是否已配置 API Key
      2. 关键依赖是否安装正确
    """
    print("\n" + "=" * 60)
    print("[CLIPBOARD] [测试1] 环境验证")
    print("=" * 60)
    
    # 检查 API Key
    if QWEN_API_KEY and QWEN_API_KEY != "your_api_key_here":
        print(f"   [OK] API Key 已配置: {QWEN_API_KEY[:8]}...")
    else:
        print("   [FAIL] API Key 未配置！请先配置 .env 文件")
        print("      参考 .env.example 创建 .env 并填入真实 Key")
        return False
    
    # 检查依赖
    print("   [PACKAGE] 检查关键依赖...")
    # 检查 langchain
    try:
        import langchain
        ver = getattr(langchain, "__version__", "已安装")
        print(f"   [OK] langchain: {ver}")
    except ImportError:
        print("   [FAIL] langchain 未安装")
        return False
    
    # 检查 langgraph（部分版本无 __version__，用 getattr 安全读取）
    try:
        import langgraph
        ver = getattr(langgraph, "__version__", "已安装")
        print(f"   [OK] langgraph: {ver}")
    except ImportError:
        print("   [FAIL] langgraph 未安装")
        return False
    
    # 检查 dashscope
    try:
        import dashscope
        print(f"   [OK] dashscope: 已安装")
    except ImportError:
        print("   [FAIL] dashscope 未安装")
        return False
    
    # 检查 chromadb
    try:
        import chromadb
        ver = getattr(chromadb, "__version__", "已安装")
        print(f"   [OK] chromadb: {ver}")
    except ImportError:
        print("   [FAIL] chromadb 未安装")
        return False
    
    print("   [OK] 环境验证通过")
    return True


# ==================== 3. 数据库测试 ====================
def test_database():
    """
    测试2: SQLite 数据库功能
    ==========================
    测试内容：
      1. 建表（users / messages / customer_profiles）
      2. 插入测试用户
      3. 插入测试客户画像
      4. 查询客户列表
      5. 搜索客户
    """
    print("\n" + "=" * 60)
    print("🗄️  [测试2] 数据库测试")
    print("=" * 60)
    
    from memory_sqlite import SQLiteMemory
    from models import User, CustomerProfile
    
    db = SQLiteMemory()
    
    # 测试插入用户
    print("\n   [MEMO] 测试插入用户...")
    user = User(name="测试用户", phone="13900000000")
    result = db.save_user(user)
    print(f"   [OK] 用户创建: {result}")
    
    # 测试查询用户
    print(f"\n   [SEARCH] 测试查询用户...")
    found = db.get_user(user.user_id)
    print(f"   [OK] 用户查询: {found.name if found else '未找到'}")
    
    # 测试插入客户画像
    print(f"\n   [MEMO] 测试插入客户画像...")
    customer = CustomerProfile(
        name="测试科技有限公司",
        company="测试科技有限公司",
        position="CEO",
        tags=["测试", "高价值"],
    )
    result = db.save_customer(customer)
    print(f"   [OK] 客户画像创建: {result}")
    
    # 测试列出所有客户
    print(f"\n   [CLIPBOARD] 测试列出所有客户...")
    customers = db.list_all_customers()
    print(f"   [OK] 共有 {len(customers)} 个客户")
    for c in customers:
        print(f"      - {c.name} | {c.company or ''} | 标签: {c.tags}")
    
    # 测试搜索客户
    print(f"\n   [SEARCH] 测试搜索客户（关键词: 测试）...")
    results = db.search_customers("测试")
    print(f"   [OK] 找到 {len(results)} 个匹配客户")
    
    db.close()
    print("\n   [OK] 数据库测试通过")


# ==================== 4. LLM 测试 ====================
def test_llm():
    """
    测试3: 通义千问 API 调用
    ==========================
    测试内容：
      1. 初始化 QwenLLM 实例
      2. 发送测试消息
      3. 验证响应
    """
    print("\n" + "=" * 60)
    print("[AI] [测试3] 通义千问 API 测试")
    print("=" * 60)
    
    from llms import QwenLLM
    
    llm = QwenLLM()
    print(f"   [PIN] 模型: {llm.model}")
    print(f"   [PIN] 温度: {llm.temperature}")
    print(f"   [PIN] 最大Token: {llm.max_tokens}")
    
    # 发送测试消息
    print("\n   [OUTBOX] 发送测试消息...")
    response = llm.invoke([
        {"role": "system", "content": "你是AI助手"},
        {"role": "user", "content": "用一句话介绍你自己"},
    ])
    
    content = response.get("content", "")
    print(f"   [INBOX] 回复: {content[:100]}...")
    print(f"\n   [OK] LLM 测试通过")


# ==================== 5. LangGraph 工作流测试 ====================
def test_workflow():
    """
    测试4: LangGraph 工作流（端到端）
    ===================================
    测试内容：
      1. 初始化所有组件（LLM / DB / RAG / Search）
      2. 运行工作流
      3. 检查最终回复
    """
    print("\n" + "=" * 60)
    print("🔗 [测试4] LangGraph 工作流测试（端到端）")
    print("=" * 60)
    
    from llms import QwenLLM
    from memory_sqlite import SQLiteMemory
    from agent import CRMWorkflow
    
    llm = QwenLLM()
    db = SQLiteMemory()
    
    # 初始化 RAG 和搜索（捕获异常，不强制要求）
    rag = None
    searcher = None
    try:
        from rag_law import RAGLaw
        rag = RAGLaw()
        print("   [OK] RAG 已加载")
    except Exception as e:
        print(f"   [WARN]  RAG 未加载（首次需下载模型）: {e}")
    
    try:
        from web_search import WebSearchTool
        searcher = WebSearchTool()
        print("   [OK] 搜索工具已加载")
    except Exception as e:
        print(f"   [WARN]  搜索工具未加载: {e}")
    
    # 创建工作流
    print("\n   [TOOL] 创建工作流...")
    workflow = CRMWorkflow(llm=llm, db=db, rag=rag, searcher=searcher)
    
    # 测试1：常规咨询
    print("\n   [OUTBOX] [测试4.1] 常规咨询: '你好，我想了解一下你们的服务'")
    try:
        result = workflow.run("你好，我想了解一下你们的服务", user_id="u001")
        print(f"   [INBOX] 回复: {result[:150]}...")
    except Exception as e:
        print(f"   [FAIL] 工作流执行失败: {e}")
    
    # 测试2：查询客户
    print(f"\n   [OUTBOX] [测试4.2] 客户查询: '我们的客户有哪些'")
    try:
        result = workflow.run("我们的客户有哪些", user_id="u001")
        print(f"   [INBOX] 回复: {result[:200]}...")
    except Exception as e:
        print(f"   [FAIL] 工作流执行失败: {e}")
    
    print(f"\n   [OK] LangGraph 工作流测试完成")


# ==================== 6. SQL Agent 测试 ====================
def test_sql_agent():
    """
    测试5: SQL Agent（自然语言 → SQL）
    =====================================
    测试内容：
      1. 初始化 LLM 和 SQL Agent
      2. 发送自然语言查询
      3. 验证 SQL 生成和执行
    """
    print("\n" + "=" * 60)
    print("[DISK] [测试5] SQL Agent 测试")
    print("=" * 60)
    
    from llms import QwenLLM
    from sql_agent import SQLAgent
    
    llm = QwenLLM()
    agent = SQLAgent(llm)
    
    # 测试查询
    queries = [
        "查询所有客户",
        "统计有多少个用户",
        "查询标签包含'高价值'的客户",
    ]
    
    for q in queries:
        print(f"\n   [OUTBOX] 查询: '{q}'")
        try:
            result = agent.query(q)
            print(f"   [INBOX] 结果:\n{result[:300]}")
        except Exception as e:
            print(f"   [FAIL] 查询失败: {e}")


# ==================== 7. Token 统计 ====================
def test_token_stats():
    """
    测试6: Token 用量统计
    =======================
    """
    print("\n" + "=" * 60)
    print("[CHART] [测试6] Token 用量统计")
    print("=" * 60)
    
    from llms import QwenLLM
    
    llm = QwenLLM()
    cost_str = llm.get_token_cost_str()
    print(f"   [CHART] {cost_str}")


# ==================== 8. 主函数 ====================
def main():
    """
    主测试函数
    ===========
    按顺序执行所有测试
    """
    print("\n[STAR] CRM 智能客户经营平台 — 整体功能测试")
    print("=" * 60)
    
    # 测试1: 环境验证
    if not test_environment():
        print("\n[FAIL] 环境验证失败，请先配置 .env 文件")
        return
    
    # 测试2: 数据库
    test_database()
    
    # 测试3: LLM
    test_llm()
    
    # 测试4: LangGraph 工作流
    test_workflow()
    
    # 测试5: SQL Agent
    test_sql_agent()
    
    # 测试6: Token 统计
    test_token_stats()
    
    # 汇总
    print("\n" + "=" * 60)
    print("[DONE] 所有测试完成！")
    print("=" * 60)
    print("\n[PIN] 后续步骤：")
    print("   1. 运行 python test.py 进行全部测试")
    print("   2. 运行 python test_rag.py 单独测试 RAG")
    print("   3. 查看 logfile/app.log 日志文件")
    print("   4. 修改 config.py 调整参数")
    print("   5. 使用 .env.example 模板分享代码")


# ==================== 9. 入口 ====================
if __name__ == "__main__":
    main()
