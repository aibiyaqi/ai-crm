"""
tools.py - 工具函数集合模块
====================================
功能分区：
  1. 工具函数注册表（统一管理所有工具函数）
  2. Function Calling 函数实现（web_search / sql_query）
  3. 工具执行器（根据函数名分发调用）
"""

# ==================== 1. 导入依赖 ====================
import json                    # JSON 处理
from typing import Dict, Any, Callable, List  # 类型提示

from logger import logger      # 日志记录器


# ==================== 2. 工具注册表类 ====================
class ToolRegistry:
    """
    工具函数注册表
    ===============
    职责：
      1. 注册所有可用的工具函数
      2. 根据函数名路由到对应的实现
      3. 整合所有工具的 Function Calling 定义
    
    设计模式：
      注册表模式（Registry Pattern）
      - 集中管理工具函数
      - 易扩展：新增工具只需注册即可
      - 松耦合：工具实现与调用逻辑分离
    
    使用方式：
      registry = ToolRegistry()
      result = registry.execute("web_search", {"query": "新闻"})
    """
    
    def __init__(self):
        """初始化工具注册表，注册所有工具"""
        # 工具函数字典：{函数名: (函数引用, 描述)}
        # 描述用于 LLM 理解工具的作用
        self._tools: Dict[str, Dict[str, Any]] = {}
        
        # 注册内置工具
        self._register_builtin_tools()
    
    def _register_builtin_tools(self):
        """
        注册内置工具函数
        =================
        所有可用工具在这里注册，方便统一管理
        """
        # 目前注册七个工具：
        # 1. web_search               — 网络搜索
        # 2. sql_query                — SQL 查询
        # 3. save_customer_profile    — 保存客户画像（K12 字段）
        # 4. screen_anomaly_customers — 异常客户筛查
        # 5. record_follow_up         — 记录一次跟进（B.4 跟进闭环 · 产品化落地）
        # 6. list_due_follow_ups      — 待跟进提醒清单（到期巡检）
        # 7. follow_up_stats          — 跟进转化统计
        
        # web_search 使用延迟导入，避免循环依赖
        self.register(
            name="web_search",
            description="搜索互联网获取实时信息",
            handler=self._handle_web_search,
        )
        
        # sql_query 使用延迟导入
        self.register(
            name="sql_query",
            description="执行SQL查询客户数据库",
            handler=self._handle_sql_query,
        )
        
        # save_customer_profile 保存客户信息到数据库（K12 教育 CRM 字段）
        self.register(
            name="save_customer_profile",
            description=(
                "保存客户画像信息到数据库。当用户提供了姓名、电话、年级、意向课程、"
                "身份、来源等信息时调用；涉价格/权益/状态等高风险字段仅记录待人工审核。"
            ),
            handler=self._handle_save_customer,
        )
        
        # screen_anomaly_customers 异常客户筛查
        self.register(
            name="screen_anomaly_customers",
            description="筛查需要优先跟进的异常客户（高意向未跟进 / 流失风险 / 资料不全）",
            handler=self._handle_screen_anomaly_customers,
        )

        # record_follow_up 记录跟进（B.4 跟进闭环：把跟进动作结构化落库）
        self.register(
            name="record_follow_up",
            description="记录一次客户跟进（跟进方式 / 沟通内容 / 跟进结果 / 下次跟进时间）",
            handler=self._handle_record_follow_up,
        )

        # list_due_follow_ups 待跟进提醒清单（到期巡检）
        self.register(
            name="list_due_follow_ups",
            description="查询今天 / 已到期该跟进的客户清单（计划到期 + 超期未跟进）",
            handler=self._handle_list_due_follow_ups,
        )

        # follow_up_stats 跟进转化统计
        self.register(
            name="follow_up_stats",
            description="统计近 N 天的跟进量、转化量与跟进转化率",
            handler=self._handle_follow_up_stats,
        )
    
    # ==================== 3. 注册方法 ====================
    def register(self, name: str, description: str, handler: Callable):
        """
        注册一个工具函数
        ==================
        参数：
          name        : 工具函数名（唯一标识）
          description : 工具功能描述（供 LLM 理解）
          handler     : 工具函数实现（可调用对象）
        """
        self._tools[name] = {
            "description": description,
            "handler": handler,
        }
        logger.debug(f"[TOOL] 工具已注册: {name}")
    
    # ==================== 4. 工具执行方法 ====================
    def execute(self, tool_name: str, arguments: dict) -> str:
        """
        执行指定工具函数
        ==================
        参数：
          tool_name  : 工具函数名
          arguments  : 参数字典
        返回：
          工具执行结果（字符串）
        """
        if tool_name not in self._tools:
            error_msg = f"[FAIL] 未知工具: {tool_name}"
            logger.error(error_msg)
            return error_msg
        
        try:
            handler = self._tools[tool_name]["handler"]
            result = handler(**arguments)
            logger.info(f"[TOOL] 工具执行成功: {tool_name}({arguments})")
            return result
        except Exception as e:
            error_msg = f"[FAIL] 工具执行失败 {tool_name}: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    # ==================== 5. 获取工具定义 ====================
    def get_tool_definitions(self) -> List[Dict]:
        """
        获取所有工具的 Function Calling 定义
        ======================================
        返回：OpenAI 兼容的函数定义列表
        
        用于 LLM 的 tools 参数：
          llm.invoke(messages, tools=registry.get_tool_definitions())
        """
        definitions = []
        
        # web_search 工具定义
        definitions.append({
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "搜索互联网获取实时信息。当你需要最新新闻、实时数据或非知识库内容时调用。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "搜索关键词"
                        }
                    },
                    "required": ["query"]
                }
            }
        })
        
        # sql_query 工具定义
        definitions.append({
            "type": "function",
            "function": {
                "name": "sql_query",
                "description": "用自然语言查询客户数据库。当你需要查询用户信息、客户画像、对话历史时调用。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "自然语言查询，如'查询所有客户'"
                        }
                    },
                    "required": ["query"]
                }
            }
        })
        
        # save_customer_profile 工具定义（K12 教育 CRM）
        definitions.append({
            "type": "function",
            "function": {
                "name": "save_customer_profile",
                "description": (
                    "保存客户画像信息到数据库。当用户提供了姓名、电话、年级、意向课程、"
                    "身份、来源、备注等信息时调用。涉价格/课时费/状态/归属等高风险字段"
                    "不会自动落库，将转为人工审核。"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "客户姓名（必填）"
                        },
                        "phone": {
                            "type": "string",
                            "description": "联系电话"
                        },
                        "email": {
                            "type": "string",
                            "description": "邮箱"
                        },
                        "age": {
                            "type": "integer",
                            "description": "年龄（学生客户）"
                        },
                        "identity": {
                            "type": "string",
                            "description": "身份：学生/家长/成人"
                        },
                        "school_grade": {
                            "type": "string",
                            "description": "就读年级，如 初二 / 高三"
                        },
                        "intent_course": {
                            "type": "string",
                            "description": "意向课程，如 初三数学一对一"
                        },
                        "source": {
                            "type": "string",
                            "description": "客户来源渠道，如 转介绍 / 地推 / 线上"
                        },
                        "matched_course": {
                            "type": "string",
                            "description": "适配班型（高风险字段，转人工审核）"
                        },
                        "intent_price": {
                            "type": "string",
                            "description": "意向价格/预算（高风险字段，转人工审核）"
                        },
                        "hourly_fee": {
                            "type": "string",
                            "description": "课时费（高风险字段，转人工审核）"
                        },
                        "active_status": {
                            "type": "string",
                            "description": "活跃状态（高风险字段，转人工审核）"
                        },
                        "advisor_id": {
                            "type": "string",
                            "description": "归属课程顾问ID（高风险字段，转人工审核）"
                        },
                        "tags": {
                            "type": "string",
                            "description": "标签，多个用逗号分隔。例如：'高意向,初三,数学'"
                        },
                        "notes": {
                            "type": "string",
                            "description": "备注信息"
                        }
                    },
                    "required": ["name"]
                }
            }
        })
        
        # screen_anomaly_customers 工具定义
        definitions.append({
            "type": "function",
            "function": {
                "name": "screen_anomaly_customers",
                "description": (
                    "筛查需要优先跟进的异常客户，返回清单。三条规则："
                    "R1 有意向但没跟进（超 10 天没跟）；"
                    "R2 快流失或已沉睡（状态流失/沉睡，或超 30 天正价课未转化）；"
                    "R3 高风险低意向（超 45 天未激活/未转化）。"
                    "当用户询问「哪些客户该跟进」「有没有需要提醒的客户」时调用。"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "stale_days": {
                            "type": "integer",
                            "description": "R1「有意向未跟进」判定天数，默认 10 天",
                            "default": 10
                        }
                    },
                    "required": []
                }
            }
        })
        
        # record_follow_up 工具定义（B.4 跟进闭环）
        definitions.append({
            "type": "function",
            "function": {
                "name": "record_follow_up",
                "description": (
                    "记录一次客户跟进。当用户说「我刚给某某打过电话」「跟进了 X 客户，"
                    "他说有意向」「记一下这次沟通，3 天后再跟」时调用。"
                    "写入跟进记录并回写客户画像的最近跟进时间。"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "customer_id": {
                            "type": "string",
                            "description": "客户 ID（优先使用；与 name 二选一）"
                        },
                        "name": {
                            "type": "string",
                            "description": "客户姓名（未提供 customer_id 时按姓名匹配）"
                        },
                        "channel": {
                            "type": "string",
                            "description": "跟进方式：电话 / 微信 / 面谈 / 短信 / 邮件 / 其他",
                            "default": "电话"
                        },
                        "content": {
                            "type": "string",
                            "description": "跟进内容摘要（聊了什么、客户反馈）"
                        },
                        "result": {
                            "type": "string",
                            "description": "跟进结果：已联系 / 未接 / 有意向 / 已成交 / 已放弃",
                            "default": "已联系"
                        },
                        "next_follow_up_days": {
                            "type": "integer",
                            "description": "计划几天后再跟进（用于到期提醒），不填表示不再提醒"
                        },
                        "advisor_id": {
                            "type": "string",
                            "description": "跟进人（课程顾问 ID），可空"
                        }
                    },
                    "required": []
                }
            }
        })

        # list_due_follow_ups 工具定义（到期巡检）
        definitions.append({
            "type": "function",
            "function": {
                "name": "list_due_follow_ups",
                "description": (
                    "查询该优先跟进、或计划跟进已到期的客户清单。"
                    "当用户问「今天该跟谁」「哪些客户到期了」「有没有要提醒我的跟进」时调用。"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "due_days": {
                            "type": "integer",
                            "description": "超期未跟进的判定天数，默认 14 天",
                            "default": 14
                        }
                    },
                    "required": []
                }
            }
        })

        # follow_up_stats 工具定义（跟进转化统计）
        definitions.append({
            "type": "function",
            "function": {
                "name": "follow_up_stats",
                "description": (
                    "统计近 N 天的跟进量、转化量与跟进转化率。"
                    "当用户问「跟进转化率多少」「这周跟进了多少客户」时调用。"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "days": {
                            "type": "integer",
                            "description": "统计窗口天数，默认 30 天",
                            "default": 30
                        }
                    },
                    "required": []
                }
            }
        })
        
        return definitions
    
    # ==================== 6. 工具函数实现（内部） ====================
    
    def _handle_web_search(self, query: str) -> str:
        """
        网络搜索工具实现
        =================
        延迟导入 web_search 模块，避免循环依赖
        
        参数：
          query : 搜索关键词
        返回：
          搜索结果文本
        """
        from web_search import WebSearchTool
        searcher = WebSearchTool()
        return searcher.search_formatted(query)
    
    def _handle_sql_query(self, query: str) -> str:
        """
        SQL 查询工具实现
        =================
        延迟导入 sql_agent 模块
        
        参数：
          query : 自然语言查询
        返回：
          查询结果文本
        """
        from llms import QwenLLM
        from sql_agent import SQLAgent
        
        llm = QwenLLM()
        agent = SQLAgent(llm)
        return agent.query(query)

    def _handle_save_customer(
        self,
        name: str,
        phone: str = "",
        email: str = "",
        age: str = "",
        identity: str = "",
        school_grade: str = "",
        intent_course: str = "",
        matched_course: str = "",
        intent_price: str = "",
        hourly_fee: str = "",
        active_status: str = "",
        advisor_id: str = "",
        source: str = "",
        tags: str = "",
        notes: str = "",
        company: str = "",
        position: str = "",
    ) -> str:
        """
        保存客户画像工具实现（K12 教育 CRM）
        =====================================
        字段分流（对应二面「客户画像字段与更新权限」）：
          - 低风险字段（AUTO_UPDATE_FIELDS）：Agent 抽取后直接落库
          - 高风险字段（MANUAL_REVIEW_FIELDS）：不落库，仅返回「待人工审核」提示
        延迟导入 models 和 memory_sqlite，避免循环依赖。
        
        参数：
          name / phone / email / age / identity / school_grade
          intent_course / matched_course / intent_price / hourly_fee
          active_status / advisor_id / source / tags / notes / company / position
        返回：
          保存结果消息
        """
        try:
            from models import (
                CustomerProfile, AUTO_UPDATE_FIELDS, MANUAL_REVIEW_FIELDS,
            )
            from memory_sqlite import SQLiteMemory

            # 解析标签
            tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []

            # 解析年龄（容错：非数字则忽略）
            age_val = None
            if str(age).strip():
                try:
                    age_val = int(str(age).strip())
                except (ValueError, TypeError):
                    age_val = None

            # -------- 低风险字段：仅保留白名单内字段后自动落库 --------
            candidate = {
                "identity": identity.strip() or None,
                "school_grade": school_grade.strip() or None,
                "intent_course": intent_course.strip() or None,
                "source": source.strip() or None,
                "notes": notes.strip() or None,
                "tags": tag_list,
                "age": age_val,
            }
            safe_fields = {
                k: v for k, v in candidate.items() if k in AUTO_UPDATE_FIELDS
            }

            # -------- 高风险字段：不自动落库，仅记录待人工审核 --------
            pending_review = {}
            for field_name, field_value in {
                "matched_course": matched_course.strip() or None,
                "intent_price": intent_price.strip() or None,
                "hourly_fee": hourly_fee.strip() or None,
                "active_status": active_status.strip() or None,
                "advisor_id": advisor_id.strip() or None,
            }.items():
                if field_name in MANUAL_REVIEW_FIELDS and field_value:
                    pending_review[field_name] = field_value

            # 创建客户画像对象（仅含已验证的安全字段）
            profile = CustomerProfile(
                name=name.strip(),
                phone=phone.strip() or None,
                email=email.strip() or None,
                company=company.strip() or None,
                position=position.strip() or None,
                **safe_fields,
            )

            # 保存到数据库
            db = SQLiteMemory()
            if db.save_customer(profile):
                logger.info(
                    f"[USER] 对话中保存客户画像: {profile.name} (ID: {profile.customer_id})"
                )
                message = (
                    f"[OK] 客户 '{profile.name}' 已成功保存到系统！"
                    f"可以在客户信息页面查看。"
                )
                if pending_review:
                    fields = "、".join(pending_review.keys())
                    message += (
                        f"\n[REVIEW] 高风险字段（{fields}）未自动落库，"
                        f"已列入待人工审核，请顾问确认后再更新。"
                    )
                return message
            return f"[FAIL] 客户 '{profile.name}' 保存失败，请重试。"

        except Exception as e:
            logger.error(f"[FAIL] 保存客户画像异常: {e}")
            return f"[FAIL] 保存客户时发生错误: {str(e)}"

    def _handle_screen_anomaly_customers(self, stale_days: str = "10") -> str:
        """
        异常客户筛查工具实现
        =====================
        规则（对应二面「异常客户筛查」）：
          R1 有意向但没跟进：已填意向课程/预算，但超过 stale_days 天没跟（默认 10 天）
          R2 快流失或已沉睡：状态命中流失/沉睡，或超 30 天正价课仍未转化成功
          R3 高风险低意向：超 45 天仍未激活/转化成功的低意向客户

        参数：
          stale_days : R1「有意向未跟进」判定天数（默认 10 天）
        返回：
          异常客户清单文本
        """
        try:
            from memory_sqlite import SQLiteMemory

            # 容错解析天数
            try:
                days = int(str(stale_days).strip() or 10)
            except (ValueError, TypeError):
                days = 10

            db = SQLiteMemory()
            results = db.screen_anomaly_customers(stale_days=days)
            if not results:
                return f"[OK] 未发现需关注的异常客户（有意向未跟进阈值 {days} 天）。"

            lines = [
                f"[WARN] 异常客户筛查（共 {len(results)} 位，有意向未跟进阈值 {days} 天）"
            ]
            lines.append("-" * 60)
            for item in results:
                reasons = "；".join(item.get("reasons", []))
                lines.append(
                    f"- {item.get('name')}（{item.get('customer_id')}）：{reasons}"
                )
            lines.append("-" * 60)
            lines.append(
                "建议：优先回访「有意向未跟进」（R1，最急），"
                "其次挽回「流失/沉睡风险」（R2），"
                "「高风险低意向」（R3）做最后触达或归档。"
            )
            return "\n".join(lines)

        except Exception as e:
            logger.error(f"[FAIL] 异常客户筛查异常: {e}")
            return f"[FAIL] 异常客户筛查失败: {str(e)}"

    # ==================== 7. 跟进闭环工具实现（B.4 · 产品化落地） ====================

    def _handle_record_follow_up(
        self,
        customer_id: str = "",
        name: str = "",
        channel: str = "电话",
        content: str = "",
        result: str = "已联系",
        next_follow_up_days: str = "",
        advisor_id: str = "",
    ) -> str:
        """
        记录一次跟进工具实现
        =====================
        参数：
          customer_id         : 客户 ID（优先）
          name                : 客户姓名（未给 ID 时按姓名精确/模糊匹配）
          channel             : 跟进方式
          content             : 跟进内容摘要
          result              : 跟进结果（已联系/未接/有意向/已成交/已放弃）
          next_follow_up_days : 几天后再跟进（写入 next_follow_up_at 供到期提醒）
          advisor_id          : 跟进人
        返回：
          写入结果消息
        """
        try:
            from datetime import datetime, timedelta
            from models import FollowUp, FOLLOW_UP_RESULTS
            from memory_sqlite import SQLiteMemory

            db = SQLiteMemory()

            # -------- 定位客户：ID 优先，其次按姓名匹配 --------
            target_id = (customer_id or "").strip()
            target_name = (name or "").strip()
            if not target_id and target_name:
                matches = db.search_customers(target_name)
                exact = [c for c in matches if c.name == target_name]
                picked = (exact or matches)
                if picked:
                    target_id = picked[0].customer_id
                    target_name = picked[0].name

            if not target_id:
                return (
                    "[FAIL] 未找到对应客户，无法记录跟进。"
                    "请提供客户 ID 或准确的客户姓名（可先新增客户）。"
                )

            # -------- 结果 / 天数容错 --------
            result_val = (result or "已联系").strip()
            if FOLLOW_UP_RESULTS and result_val not in FOLLOW_UP_RESULTS:
                logger.warning(f"[WARN] 非标准跟进结果：{result_val}（按原样记录）")

            next_at = None
            if str(next_follow_up_days).strip():
                try:
                    next_at = datetime.now() + timedelta(
                        days=int(str(next_follow_up_days).strip())
                    )
                except (ValueError, TypeError):
                    next_at = None

            # -------- 落库（同时回写画像 last_follow_up） --------
            record = FollowUp(
                customer_id=target_id,
                advisor_id=(advisor_id or "").strip() or None,
                channel=(channel or "电话").strip(),
                content=(content or "").strip() or None,
                result=result_val,
                next_follow_up_at=next_at,
            )
            if db.save_follow_up(record):
                msg = (
                    f"[OK] 已记录跟进：{target_name or target_id}"
                    f"（{record.channel} / {record.result}）\n"
                    f"客户画像最近跟进时间已同步更新。"
                )
                if next_at:
                    msg += f"\n[REMIND] 已设定下次跟进时间：{next_at:%Y-%m-%d %H:%M}"
                return msg
            return "[FAIL] 跟进记录保存失败，请重试。"

        except Exception as e:
            logger.error(f"[FAIL] 记录跟进异常: {e}")
            return f"[FAIL] 记录跟进失败: {str(e)}"

    def _handle_list_due_follow_ups(self, due_days: str = "14") -> str:
        """
        待跟进提醒清单工具实现（到期巡检）
        ====================================
        参数：
          due_days : 超期未跟进判定天数（默认 14 天）
        返回：
          待跟进清单文本
        """
        try:
            from memory_sqlite import SQLiteMemory

            try:
                days = int(str(due_days).strip() or 14)
            except (ValueError, TypeError):
                days = 14

            db = SQLiteMemory()
            items = db.list_due_follow_ups(due_days=days)
            if not items:
                return f"[OK] 暂无到期 / 超期未跟进客户（超期阈值 {days} 天）。"

            planned = [i for i in items if i.get("due_type") == "计划到期"]
            overdue = [i for i in items if i.get("due_type") == "超期未跟进"]

            lines = [f"[WARN] 待跟进提醒（共 {len(items)} 位）"]
            lines.append("-" * 60)
            if planned:
                lines.append(f"■ 计划跟进已到期（{len(planned)} 位）")
                for i in planned:
                    lines.append(f"  - {i['name']}（{i['customer_id']}）：{'；'.join(i['reasons'])}")
            if overdue:
                lines.append(f"■ 超期未跟进（{len(overdue)} 位，阈值 {days} 天）")
                for i in overdue:
                    days_txt = f"{i['days']} 天" if i.get("days") is not None else "从未跟进"
                    lines.append(f"  - {i['name']}（{i['customer_id']}）：{days_txt}未跟进")
            lines.append("-" * 60)
            lines.append("建议：先跟进「计划到期」，再补「超期未跟进」；跟进后用 record_follow_up 记录并设定下次时间。")
            return "\n".join(lines)

        except Exception as e:
            logger.error(f"[FAIL] 待跟进清单查询异常: {e}")
            return f"[FAIL] 待跟进清单查询失败: {str(e)}"

    def _handle_follow_up_stats(self, days: str = "30") -> str:
        """
        跟进转化统计工具实现
        =====================
        参数：
          days : 统计窗口天数（默认 30 天）
        返回：
          统计结果文本（含分子分母口径）
        """
        try:
            from memory_sqlite import SQLiteMemory

            try:
                window = int(str(days).strip() or 30)
            except (ValueError, TypeError):
                window = 30

            db = SQLiteMemory()
            stats = db.follow_up_conversion_stats(days=window)
            rate = stats.get("conversion_rate")
            rate_txt = f"{rate * 100:.1f}%" if rate is not None else "无数据（不报 0 冒充）"

            lines = [
                f"[CHART] 跟进转化统计（近 {stats['window_days']} 天）",
                "-" * 60,
                f"跟进记录数（分母）：{stats['total']} 条",
                f"转化跟进数（分子：「已成交 / 有意向」）：{stats['converted']} 条",
                f"跟进转化率：{rate_txt}",
                f"覆盖客户数：{stats['covered_customers']} 位",
            ]
            if stats.get("by_result"):
                lines.append("结果分布：" + "、".join(
                    f"{k} {v} 条" for k, v in stats["by_result"].items()
                ))
            lines.append("-" * 60)
            lines.append("口径：分母 = 窗口内跟进记录条数；分子 = 结果为「已成交 / 有意向」的条数。")
            return "\n".join(lines)

        except Exception as e:
            logger.error(f"[FAIL] 跟进转化统计异常: {e}")
            return f"[FAIL] 跟进转化统计失败: {str(e)}"
