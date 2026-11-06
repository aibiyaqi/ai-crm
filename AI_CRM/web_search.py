"""
web_search.py - 网络搜索模块
====================================
功能分区：
  1. DuckDuckGo 免费网络搜索（无需 API Key）
  2. 搜索结果格式化（标题/摘要/链接）
  3. Function Calling 工具定义（供 LLM 调用）
"""

# ==================== 1. 导入依赖 ====================
import json                          # JSON 处理，格式化搜索结果
from typing import Optional, List, Dict  # 类型提示

# DuckDuckGo 免费搜索库
# 安装：pip install duckduckgo-search==7.5.1
# 无需注册、无需 API Key、无速率限制（合理使用即可）
from duckduckgo_search import DDGS   # DuckDuckGo 搜索客户端

from logger import logger            # 日志记录器


# ==================== 2. 网络搜索类 ====================
class WebSearchTool:
    """
    网络搜索工具类
    ==============
    功能：使用 DuckDuckGo 免费搜索引擎搜索互联网
    
    为什么选 DuckDuckGo？
      [ONLINE] 个人/开发/免费 → 选 DuckDuckGo（零成本）
      [BLUE] 生产/企业/高质量 → 可选 Tavily（需要 API Key，有免费额度）
    
    使用方式：
      searcher = WebSearchTool()
      results = searcher.search("通义千问最新消息")
    """
    
    def __init__(self):
        """初始化搜索工具"""
        # DDGS 是线程安全的，可以在多次搜索中复用
        self.client = DDGS()
        logger.info("[GLOBE] DuckDuckGo 搜索工具已就绪")
    
    # ==================== 3. 搜索方法 ====================
    def search(
        self,
        query: str,
        max_results: int = 5,
        region: str = "cn-zh"
    ) -> List[Dict]:
        """
        执行网络搜索
        ==============
        参数：
          query       : 搜索关键词
          max_results : 返回结果数量（默认 5 条）
          region      : 搜索区域（cn-zh = 中文搜索结果）
        返回：
          [
              {
                  "title": "标题",
                  "body": "摘要",
                  "href": "链接"
              },
              ...
          ]
        """
        try:
            logger.info(f"[SEARCH] 正在搜索: '{query}'")
            
            # 执行搜索
            # text() 方法返回生成器，逐个 yield 搜索结果
            results = list(
                self.client.text(
                    keywords=query,       # 搜索关键词
                    region=region,        # 搜索区域
                    max_results=max_results,  # 最大结果数
                )
            )
            
            logger.info(f"[OK] 搜索完成: '{query}' → {len(results)} 条结果")
            return results
            
        except Exception as e:
            logger.error(f"[FAIL] 搜索失败: {e}")
            return [{"title": "搜索出错", "body": f"搜索请求失败: {str(e)}", "href": ""}]
    
    # ==================== 4. 搜索结果格式化 ====================
    def search_formatted(self, query: str, max_results: int = 5) -> str:
        """
        搜索并返回格式化文本（直接给 LLM 作为上下文）
        ================================================
        参数：
          query       : 搜索关键词
          max_results : 返回结果数
        返回：
          格式化文本，如：
          📰 搜索结果：通义千问最新消息
          1. 标题：...
             摘要：...
             链接：...
          ...
        """
        results = self.search(query, max_results)
        
        if not results:
            return "[FAIL] 未找到相关搜索结果"
        
        # 格式化为文本
        lines = [f"📰 搜索结果：{query}"]
        lines.append("=" * 50)
        
        for i, r in enumerate(results, 1):
            title = r.get("title", "无标题")
            body = r.get("body", "无摘要")
            href = r.get("href", "")
            
            lines.append(f"\n{i}. [PIN] {title}")
            lines.append(f"   摘要：{body[:200]}...")  # 截断过长摘要
            if href:
                lines.append(f"   链接：{href}")
        
        return "\n".join(lines)
    
    # ==================== 5. Function Calling 工具定义 ====================
    @staticmethod
    def get_tool_definition() -> List[Dict]:
        """
        获取 Function Calling 工具定义
        ================================
        供 LLM（通义千问）调用的工具函数描述
        
        返回格式符合 OpenAI Function Calling 规范：
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "搜索互联网获取实时信息",
                "parameters": {...}
            }
        }
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "搜索互联网获取实时信息。当你需要最新新闻、实时数据或非知识库内容时调用此函数。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "搜索查询关键词，应清晰具体"
                            }
                        },
                        "required": ["query"]
                    }
                }
            }
        ]
