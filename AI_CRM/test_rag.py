"""
test_rag.py - RAG 知识库模块测试
====================================
功能分区：
  1. 测试 Chroma 向量数据库初始化
  2. 测试文档添加（文本文件上传）
  3. 测试混合检索（向量 + BM25）
  4. 测试结果展示
"""

# ==================== 1. 导入依赖 ====================
import os                                # 文件路径操作
import sys                               # 系统参数

# 手动将本项目根目录加入 sys.path，确保模块可以被导入
# 因为 test 脚本可能从项目根目录执行
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from logger import logger                # 日志记录器


# ==================== 2. 测试主函数 ====================
def test_rag():
    """
    RAG 知识库测试函数
    ====================
    测试步骤：
      1. 初始化 RAGLaw 实例
      2. 打印知识库统计信息
      3. 测试添加文本到知识库
      4. 测试混合检索
      5. 展示检索结果
    """
    print("=" * 60)
    print("🔬 RAG 知识库测试")
    print("=" * 60)
    
    # -------- 2.1 初始化 RAG --------
    print("\n[PIN] [步骤1] 初始化 RAG 知识库...")
    from rag_law import RAGLaw
    
    try:
        rag = RAGLaw()
        print("[OK] RAG 知识库初始化成功")
    except Exception as e:
        print(f"[FAIL] RAG 初始化失败: {e}")
        print("\n[IDEA] 提示：首次运行会下载约 2GB 的 sentence-transformers 模型")
        print("   请确保网络通畅，或检查是否已安装 sentence-transformers==3.0.1")
        return
    
    # -------- 2.2 知识库统计 --------
    print("\n[PIN] [步骤2] 知识库统计信息：")
    stats = rag.get_stats()
    for key, value in stats.items():
        print(f"   - {key}: {value}")
    
    # -------- 2.3 添加测试文本 --------
    print("\n[PIN] [步骤3] 添加测试文本到知识库...")
    
    test_texts = [
        "客户经营的核心是以客户为中心，通过持续互动和个性化服务提升客户满意度和忠诚度。",
        "智能客服系统可以7x24小时自动回答客户问题，大幅降低人工客服成本。",
        "CRM系统帮助企业记录客户信息、跟踪销售机会、分析客户行为。",
        "RAG（检索增强生成）技术通过结合知识库检索和大语言模型，提高回答准确率。",
        "通义千问（Qwen）是阿里云推出的免费大语言模型，支持API调用。",
    ]
    
    for i, text in enumerate(test_texts, 1):
        try:
            chunks = rag.add_text(
                text,
                metadata={"source": f"测试文本{i}", "category": "test"}
            )
            print(f"   [OK] 测试文本 {i} 已添加 → {chunks} 个文档块")
        except Exception as e:
            print(f"   [FAIL] 测试文本 {i} 添加失败: {e}")
    
    # -------- 2.4 测试检索 --------
    print("\n[PIN] [步骤4] 测试混合检索...")
    
    test_queries = [
        "什么是智能客服",
        "CRM系统有什么用",
        "RAG技术",
    ]
    
    for query in test_queries:
        print(f"\n   [SEARCH] 查询: '{query}'")
        try:
            results = rag.hybrid_search(query, top_k=2)
            print(f"      → 找到 {len(results)} 条结果")
            for j, doc in enumerate(results, 1):
                # 截断过长内容
                content = doc[:100] + "..." if len(doc) > 100 else doc
                print(f"      [{j}] {content}")
        except Exception as e:
            print(f"      [FAIL] 检索失败: {e}")
    
    # -------- 2.5 完成 --------
    print("\n" + "=" * 60)
    print("[OK] RAG 测试完成")
    print("=" * 60)


# ==================== 3. 入口 ====================
if __name__ == "__main__":
    test_rag()
