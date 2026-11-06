"""
llms.py - 大模型实例模块
====================================
功能分区：
  1. dashscope SDK 初始化（通义千问官方调用方式）
  2. OpenAI 兼容接口封装（统一调用格式）
  3. 流式输出支持（逐步返回，实时展示）
  4. Function Calling 支持（工具调用）
  5. Token 用量统计（成本监控）
"""

# ==================== 1. 导入依赖 ====================
from http import HTTPStatus          # HTTP 状态码，用于判断 API 响应
from typing import Optional, List    # 类型提示
import json                          # JSON 序列化，处理工具调用参数
import os                            # 操作系统接口，用于读取环境变量
import re                            # 正则表达式，用于清理消息中的特殊字符

from logger import logger            # 日志记录器

# 通义千问官方 SDK（dashscope v1.20.0）
# 安装：pip install dashscope==1.20.0
from dashscope import Generation    # 文本生成 API

# [WARN] dashscope SDK 默认读取环境变量 DASHSCOPE_API_KEY
# 但我们的 .env 中变量名为 QWEN_API_KEY
# 所以需要手动加载并设置 API Key
from dotenv import load_dotenv
load_dotenv()  # 加载 .env 文件

# 从 .env 读取 QWEN_API_KEY，并设置为 dashscope 的 API Key
_api_key = os.getenv("QWEN_API_KEY")
if _api_key:
    import dashscope
    dashscope.api_key = _api_key  # 显式设置 dashscope API Key

# ==================== 辅助函数：清理消息中的特殊字符 ====================
def _sanitize_messages(messages: list) -> list:
    """
    清理消息中的特殊字符，防止 dashscope SDK latin-1 编码错误
    ============================================================
    dashscope SDK 内部使用 latin-1 编码处理 HTTP 请求，
    当消息内容包含 latin-1 无法编码的字符时（如某些 Unicode 扩展字符），
    会抛出 UnicodeEncodeError。
    
    此函数移除或替换这些有问题的字符。
    """
    sanitized = []
    for msg in messages:
        new_msg = {}
        for key, value in msg.items():
            if isinstance(value, str):
                # 移除 latin-1 无法编码的字符（保留 ASCII + 常见 Unicode）
                try:
                    value.encode("latin-1")
                except UnicodeEncodeError:
                    # 用空格替换无法编码的字符
                    value = value.encode("latin-1", errors="replace").decode("latin-1")
            new_msg[key] = value
        sanitized.append(new_msg)
    return sanitized


# ==================== 2. 通义千问调用类 ====================
class QwenLLM:
    """
    通义千问大模型封装类
    ====================
    设计目标：
      1. 统一调用接口，上层模块无需关心 SDK 差异
      2. 兼容 OpenAI 格式的 messages 参数（方便后续切换）
      3. 支持流式（stream）和非流式两种模式
    
    使用方式：
      llm = QwenLLM()
      response = llm.invoke([{"role": "user", "content": "你好"}])
    """
    
    def __init__(self):
        """
        初始化大模型实例
        -----------------
        从 config.py 读取全局配置参数：
          - model：模型名称（qwen-plus / qwen-turbo）
          - temperature：温度参数（0.3=保守）
          - max_tokens：最大输出 Token 数
        """
        # 延迟导入 config，避免循环依赖
        # config 从 .env 加载 API Key，dashscope 自动读取
        from config import (
            QWEN_MODEL_NAME,      # 模型名称，如 "qwen-plus"
            TEMPERATURE,          # 采样温度，默认 0.3
            MAX_TOKENS,           # 最大 Token 数，默认 4096
        )
        # QWEN_API_KEY 已在 config.py 中通过 load_dotenv() 加载
        # dashscope SDK 自动从环境变量 QWEN_API_KEY 读取
        
        # 赋值实例属性，供外部访问
        self.model: str = QWEN_MODEL_NAME      # 模型名称
        self.temperature: float = TEMPERATURE   # 温度参数
        self.max_tokens: int = MAX_TOKENS       # 最大 Token 数
        
        # Token 用量统计（累计计数，用于成本监控）
        self.total_prompt_tokens: int = 0       # 累计输入 Token 数
        self.total_completion_tokens: int = 0   # 累计输出 Token 数
    
    # ==================== 3. 核心调用方法（非流式） ====================
    def invoke(
        self,
        messages: List[dict],
        tools: Optional[List[dict]] = None,
    ) -> dict:
        """
        非流式调用通义千问大模型
        ==========================
        参数：
          messages  : 对话消息列表，格式 [{"role": "user", "content": "你好"}]
          tools     : 工具函数定义列表（Function Calling 用）
        
        返回：
          成功：{"role": "assistant", "content": "...", "tool_calls": [...]}
          失败：{"role": "assistant", "content": "错误信息"}
        """
        call_params = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
            "result_format": "message",
        }
        if tools:
            call_params["tools"] = tools
        
        try:
            response = Generation.call(**call_params)
        except UnicodeEncodeError as e:
            logger.error(f"[FAIL] dashscope 编码错误: {e}，尝试清理消息后重试")
            # 清理消息中的特殊字符后重试
            cleaned = _sanitize_messages(messages)
            call_params["messages"] = cleaned
            try:
                response = Generation.call(**call_params)
            except Exception as e2:
                return {"role": "assistant", "content": f"抱歉，服务编码异常: {e2}"}
        except Exception as e:
            return {"role": "assistant", "content": f"抱歉，服务暂时不可用: {e}"}
        
        if response.status_code == HTTPStatus.OK:
            assistant_message = response.output.choices[0].message
            usage = response.usage
            self.total_prompt_tokens += usage.get("input_tokens", 0)
            self.total_completion_tokens += usage.get("output_tokens", 0)
            
            result = {"role": "assistant", "content": assistant_message.content or ""}
            
            try:
                if assistant_message.tool_calls:
                    result["tool_calls"] = [
                        {
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        }
                        for tc in assistant_message.tool_calls
                    ]
            except (AttributeError, KeyError, TypeError):
                pass
            
            return result
        else:
            return {"role": "assistant", "content": f"API 调用失败 (HTTP {response.status_code})"}
    
    # ==================== 4. 流式调用方法 ====================
    def stream_invoke(
        self,
        messages: List[dict],
        tools: Optional[List[dict]] = None,
    ):
        """
        流式调用 LLM，返回文本块生成器（Generator）
        ============================================
        用 yield 逐块返回回答文本，调用方可用 for 循环消费。
        
        使用示例：
          for chunk in llm.stream_invoke(messages):
              print(chunk, end="")
        """
        call_params = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": True,
            "result_format": "message",
        }
        if tools:
            call_params["tools"] = tools
        
        try:
            responses = Generation.call(**call_params)
            for response in responses:
                if response.status_code == HTTPStatus.OK:
                    chunk = response.output.choices[0].message.content
                    if chunk:
                        yield chunk
                else:
                    yield f"\n\n[API 错误: HTTP {response.status_code}]"
        except UnicodeEncodeError as e:
            logger.error(f"[FAIL] dashscope 流式编码错误: {e}，尝试清理消息后重试")
            cleaned = _sanitize_messages(messages)
            call_params["messages"] = cleaned
            try:
                responses = Generation.call(**call_params)
                for response in responses:
                    if response.status_code == HTTPStatus.OK:
                        chunk = response.output.choices[0].message.content
                        if chunk:
                            yield chunk
            except Exception as e2:
                yield f"\n\n[编码错误，请重试: {e2}]"
        except Exception as e:
            yield f"\n\n[调用失败: {e}]"
    
    # ==================== 6. 获取 Token 用量统计 ====================
    def get_token_stats(self) -> dict:
        """
        返回当前会话的 Token 使用统计
        --------------------------------
        用于成本监控和用量报告
        
        返回：{
            "total_prompt_tokens": 1000,         # 总输入 Token
            "total_completion_tokens": 500,      # 总输出 Token
            "total_tokens": 1500                 # 总 Token 数
        }
        """
        return {
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
        }
    
    # ==================== 7. Token 用量格式化输出 ====================
    def get_token_cost_str(self) -> str:
        """
        返回 Token 用量的可读字符串
        --------------------------------
        示例：输入 1,000 Token | 输出 500 Token | 总计 1,500 Token
        """
        stats = self.get_token_stats()
        return (
            f"输入 {stats['total_prompt_tokens']:,} Token"
            f" | 输出 {stats['total_completion_tokens']:,} Token"
            f" | 总计 {stats['total_tokens']:,} Token"
        )
