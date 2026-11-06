"""
agent.py - LangGraph 工作流编排模块（核心）
============================================
功能分区：
  1. LangGraph 图结构定义（6 个节点 + 条件分支）
  2. 节点1: 查询上下文（memory_sqlite → 用户信息 + 对话历史）
  3. 节点2: 意图识别（llms → 分类：实时信息/业务咨询/常规咨询）
  4. 节点3: 条件分支路由（根据意图选择不同执行路径）
  5. 节点4: 各路径执行（web_search / rag_law / direct_llm）
  6. 节点5: 生成个性化回答
  7. 节点6: 保存对话历史（memory_sqlite）
  8. 图编译与执行入口

工作流图示：
  用户输入 → 节点1(查上下文) → 节点2(意图识别)
    → 条件分支
      ├─ 意图1(实时信息) → web_search
      ├─ 意图2(业务咨询) → rag_law 混合检索
      └─ 意图3(常规咨询) → llms 直接回答
    → 节点5(生成回答) → 节点6(保存历史) → 返回结果
"""

# ==================== 1. 导入依赖 ====================
from typing import Literal     # 类型提示：字面量类型

# 意图识别缓存（Redis 优先，LRU 兜底）
# 背景：原进程内 OrderedDict LRU 在多进程部署下缓存不共享、容量受限、重启即失效；
#       改造为 IntentCache 门面，Redis 作为共享缓存，不可用时自动降级 LRU。
from intent_cache import IntentCache

# LangGraph 核心
# [WARN] 必须使用 langgraph==0.2.9
# StateGraph：有状态图（节点共享状态）
from langgraph.graph import StateGraph, END

from models import AgentState  # 状态模型（节点间共享数据）
from logger import logger      # 日志记录器

# 意图规则路由配置
from config import INTENT_RULES

# 运行时指标埋点（可观测性增强：模型效果指标 + 系统运行指标）
# 说明：埋点函数内部已吞掉自身异常，绝不会影响主流程
from metrics import incr, observe, timed, record_event

# 任务检查点（中断恢复）——补齐三面被追问的短板
# 说明：对话记忆（memory_sqlite）≠ 任务检查点；前者记"说了什么"，
#       后者记"工作流跑到哪个节点"，用于断点续跑 / 暂停恢复。
from checkpoint import get_checkpointer, get_checkpoint_manager


# ==================== 2. LangGraph 工作流类 ====================
class CRMWorkflow:
    """
    CRM 智能客服工作流
    ===================
    使用 LangGraph 构建的多智能体工作流
    
    节点列表：
      load_context     → 查询上下文（用户信息 + 对话历史）
      classify_intent  → 意图识别（LLM 分类）
      route_intent     → 条件路由（分发到不同路径）
      web_search       → 网络搜索路径
      rag_retrieve     → RAG 知识库检索路径
      direct_llm       → 直接 LLM 回答路径
      generate_response → 生成个性化回答
      save_history     → 保存对话历史
    
    使用方式：
      workflow = CRMWorkflow(llm, db, rag, searcher)
      result = workflow.run("你好，我想查询客户信息")
    """
    
    def __init__(self, llm, db, rag=None, searcher=None):
        """
        初始化工作流
        ==============
        参数：
          llm      : QwenLLM 实例（必填）
          db       : SQLiteMemory 实例（必填）
          rag      : RAGLaw 实例（可选，没有则跳过 RAG 路径）
          searcher : WebSearchTool 实例（可选，没有则跳过搜索路径）
        """
        self.llm = llm
        self.db = db
        self.rag = rag
        self.searcher = searcher

        # [方案B] 意图识别缓存（Redis 优先，LRU 兜底）
        # 改造点：原进程内 OrderedDict LRU → IntentCache 门面
        #   · Redis 共享缓存：多进程 / 多实例共享同一份，命中率不再被稀释
        #   · 自动降级：Redis 不可用时回落进程内 LRU，主链路不中断
        self._intent_cache = IntentCache()

        # 构建 LangGraph 图
        self.graph = self._build_graph()
        
        logger.info("[AI] LangGraph 工作流已初始化")
    
    # ==================== 3. 构建 LangGraph 图 ====================
    def _build_graph(self):
        """
        构建 LangGraph 有状态图
        ========================
        步骤：
          1. 创建 StateGraph 实例（状态类型为 AgentState）
          2. 注册所有节点（每个节点是一个函数）
          3. 注册条件边（意图路由）
          4. 注册普通边（节点间流转）
          5. 设置入口点
          6. 编译图
        """
        # -------- 3.1 创建图实例 --------
        # AgentState 是所有节点共享的状态对象
        workflow = StateGraph(AgentState)
        
        # -------- 3.2 注册节点 --------
        # 每个节点是一个函数，接收 AgentState 返回 AgentState
        workflow.add_node("load_context", self._node_load_context)
        workflow.add_node("classify_intent", self._node_classify_intent)
        workflow.add_node("web_search", self._node_web_search)
        workflow.add_node("rag_retrieve", self._node_rag_retrieve)
        workflow.add_node("direct_llm", self._node_direct_llm)
        workflow.add_node("generate_response", self._node_generate_response)
        workflow.add_node("save_history", self._node_save_history)
        
        # -------- 3.3 设置入口点 --------
        # 工作流从 load_context 节点开始
        workflow.set_entry_point("load_context")
        
        # -------- 3.4 注册边 --------
        # load_context → classify_intent（无条件流转）
        workflow.add_edge("load_context", "classify_intent")
        
        # classify_intent → 条件路由（根据意图走不同分支）
        workflow.add_conditional_edges(
            "classify_intent",          # 源节点
            self._route_intent,         # 路由函数（返回目标节点名）
            {
                "web_search": "web_search",       # 实时信息 → 搜索
                "rag_retrieve": "rag_retrieve",   # 业务咨询 → RAG
                "direct_llm": "direct_llm",       # 常规咨询 → 直接回答
            }
        )
        
        # 三个分支路径最终都汇聚到 generate_response
        workflow.add_edge("web_search", "generate_response")
        workflow.add_edge("rag_retrieve", "generate_response")
        workflow.add_edge("direct_llm", "generate_response")
        
        # generate_response → save_history → END
        workflow.add_edge("generate_response", "save_history")
        workflow.add_edge("save_history", END)
        
        # -------- 3.5 编译图（挂载检查点，支持中断恢复） --------
        # compile() 返回可执行的 CompiledGraph
        # [新增] checkpointer：每个节点执行后自动写一条状态快照，
        #        配合 thread_id 即可实现"用户终止 → 下次读档续跑"。
        try:
            self.checkpointer = get_checkpointer()
            return workflow.compile(checkpointer=self.checkpointer)
        except Exception as e:
            # 检查点不可用不能拖垮主链路：降级为无检查点编译
            logger.warning(f"[WARN] 检查点挂载失败，降级为无检查点模式: {e}")
            self.checkpointer = None
            return workflow.compile()
    
    # ==================== 4. 节点实现 ====================
    
    # -------- 4.1 节点1: 查询上下文 --------
    def _node_load_context(self, state: AgentState) -> AgentState:
        """
        节点1: 加载用户上下文
        =======================
        功能：
          1. 从 memory_sqlite 查询用户信息（不存在则自动创建）
          2. 加载最近对话历史（按会话 ID）
        
        输入：state.user_id, state.session_id, state.query
        输出：state.user_info, state.history
        """
        logger.info(f"📍 [节点1] 加载上下文: user={state.user_id}, session={state.session_id}")
        
        # 查询用户信息
        user = self.db.get_user(state.user_id)
        if user:
            state.user_info = {
                "user_id": user.user_id,
                "name": user.name,
                "phone": user.phone,
            }
        else:
            # 用户不存在 → 自动创建（避免后续 save_message 的 FOREIGN KEY 约束失败）
            from models import User
            new_user = User(
                user_id=state.user_id,
                name=state.user_id[:8],       # 用 user_id 前8位作为默认名
                phone="",
            )
            self.db.save_user(new_user)
            logger.info(f"[USER] 自动创建用户: {new_user.user_id}")
            state.user_info = {
                "user_id": new_user.user_id,
                "name": new_user.name,
                "phone": new_user.phone,
            }
        
        # 加载最近对话历史（按会话 ID，字典格式直接给 LLM）
        if state.session_id:
            state.history = self.db.get_history_dicts(state.session_id)
        else:
            state.history = []
        
        return state
    
    # -------- 4.2 节点2: 意图识别（规则兜底 + LLM 保底） --------
    def _rule_based_intent(self, query: str) -> str:
        """
        轻量级规则兜底：通过关键词匹配快速识别意图
        ==============================================
        功能：在调用 LLM 之前先用规则匹配，匹配成功则跳过 LLM 调用
        
        优先级：规则匹配 > LLM 分类
        匹配逻辑：
          1. 遍历 INTENT_RULES 中每个意图的关键词列表
          2. 统计用户问题中包含的关键词数量
          3. 如果命中数 >= min_match，则直接返回该意图
          4. 如果所有规则都未命中，返回 None（走 LLM 兜底）
        
        返回：
          "web_search" | "rag_retrieve" | None
        """
        query_lower = query.lower()
        
        for intent, config in INTENT_RULES.items():
            keywords = config["keywords"]
            min_match = config["min_match"]
            
            # 统计命中的关键词数量
            match_count = sum(1 for kw in keywords if kw.lower() in query_lower)
            
            if match_count >= min_match:
                logger.info(f"📍 [规则路由] 关键词命中 → {intent} (命中{match_count}个关键词)")
                return intent
        
        return None  # 规则未命中，需要走 LLM 兜底

    # -------- [方案B] LLM 语义意图分类（带 LRU 缓存）--------
    def _classify_intent_with_llm(self, query: str) -> str:
        """
        [方案B] 使用 LLM 进行语义级别的意图分类
        ==========================================
        当规则匹配无法确定意图时，调用 LLM 进行语义理解。
        
        优势：
          - 语义理解：能理解"我想了解一下你们的产品功能"这类需要检索的问题
          - 零配置：无需维护关键词列表
          - 自适应：能处理新出现的表达方式
        
        缓存策略（Redis 优先，LRU 兜底）：
          - 缓存键：crm:intent:{md5(query)}（由 IntentCache 内部生成）
          - Redis 共享缓存 + TTL；Redis 不可用时自动降级进程内 LRU
          - 相同问题直接命中缓存，避免重复调用 LLM
        
        返回：
          "web_search" | "rag_retrieve" | "direct_llm"
        """
        # -------- 缓存查找（Redis 优先，LRU 兜底） --------
        cached_intent = self._intent_cache.get(query)
        if cached_intent:
            logger.info(f"📍 [方案B] LLM分类缓存命中 → {cached_intent}")
            return cached_intent
        
        # -------- 调用 LLM 进行意图分类 --------
        logger.info(f"📍 [方案B] 调用 LLM 语义分类: '{query[:40]}...'")
        
        system_prompt = (
            "你是一个意图分类器。分析用户的问题，只返回一个词作为分类结果，不要解释。\n\n"
            "分类规则：\n"
            "1. rag_retrieve - 用户询问课程产品资料、班型/课时/学费说明、"
            "师资与教材介绍、课程体系、方法步骤、概念定义等需要检索知识库的问题\n"
            "2. web_search - 用户询问实时信息，如新闻、天气、股票、最新事件等\n"
            "3. direct_llm - 普通聊天、问候、感谢、简单问答、无需检索外部信息的问题\n\n"
            "示例：\n"
            "用户：初三数学一对一课时费大概多少？ → rag_retrieve\n"
            "用户：你们都有哪些班型？ → rag_retrieve\n"
            "用户：今天天气怎么样？ → web_search\n"
            "用户：你好 → direct_llm\n"
            "用户：最近有什么教育新闻 → web_search\n"
            "用户：帮我介绍一下你们的课程体系 → rag_retrieve"
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
        
        try:
            # 埋点：意图分类阶段的 LLM 调用次数与耗时
            with timed("llm_latency_ms", stage="intent"):
                response = self.llm.invoke(messages)
            incr("llm_call_total", stage="intent")
            intent = response.get("content", "").strip().lower()
            
            # 验证返回的意图是否合法
            valid_intents = {"web_search", "rag_retrieve", "direct_llm"}
            if intent not in valid_intents:
                logger.warning(
                    f"[WARN] [方案B] LLM返回了未知意图 '{intent}'，默认 direct_llm"
                )
                intent = "direct_llm"
            
            # -------- 写入缓存（Redis 优先，LRU 兜底） --------
            self._intent_cache.set(query, intent)
            
            logger.info(f"📍 [方案B] LLM分类结果 → {intent}")
            return intent
            
        except Exception as e:
            logger.error(f"[FAIL] [方案B] LLM意图分类失败: {e}")
            return "direct_llm"  # 失败时兜底走直接回答

    def _node_classify_intent(self, state: AgentState) -> AgentState:
        """
        节点2: 用户意图识别（规则匹配 + [方案B] LLM 语义兜底）
        ========================================================
        两阶段分类策略：
          阶段1: 关键词规则匹配（零成本、零延迟）
            - 快速命中已知模式的关键词
            - 适合明确的关键词触发场景
          
          阶段2: [方案B] LLM 语义分类（带缓存）
            - 规则未命中时，使用 LLM 进行语义理解
            - 通过 LRU 缓存避免重复调用
            - 适合复杂、无固定关键词的查询
        
        分类结果：
          1. "web_search"   → 用户问实时信息（新闻/天气/股票等）
          2. "rag_retrieve" → 用户问业务知识（课程产品资料/班型/课时/学费）
          3. "direct_llm"   → 常规聊天/问候/简单问题（默认兜底）
        """
        logger.info(f"📍 [节点2] 意图识别: '{state.query[:40]}...'")
        
        # -------- 阶段1: 规则匹配（零成本快速命中）--------
        rule_intent = self._rule_based_intent(state.query)
        
        if rule_intent:
            logger.info(f"📍 [节点2] [BOLT]规则命中 → {rule_intent}")
            # 埋点：规则命中（用于计算「规则命中率」，替代口头估测 60~70%）
            incr("intent_route_total", route="rule_hit")
            state.intent = rule_intent
            return state
        
        # 埋点：规则未命中 → 计数 + 记录一条 badcase 线索（供后续回流标注测试集）
        incr("intent_route_total", route="rule_miss")
        record_event("intent_rule_miss", query=state.query)
        
        # -------- 阶段2: [方案B] LLM 语义分类（带缓存）--------
        logger.info("📍 [节点2] 规则未命中，尝试 LLM 语义分类...")
        llm_intent = self._classify_intent_with_llm(state.query)
        
        if llm_intent:
            logger.info(f"📍 [节点2] [AI] LLM语义分类 → {llm_intent}")
            incr("intent_route_total", route="llm")
            state.intent = llm_intent
        else:
            # 最终兜底
            logger.info("📍 [节点2] LLM分类失败，默认 → direct_llm")
            incr("intent_route_total", route="fallback")
            state.intent = "direct_llm"
        
        return state
    
    # -------- 4.3 节点3: 条件路由 --------
    def _route_intent(self, state: AgentState) -> Literal["web_search", "rag_retrieve", "direct_llm"]:
        """
        条件路由函数
        ==============
        根据意图识别结果，返回下一个节点名称
        
        返回：
          "web_search"    → 走搜索路径
          "rag_retrieve"  → 走 RAG 路径
          "direct_llm"    → 直接回答
        """
        logger.info(f"📍 [节点3] 条件路由 → {state.intent}")
        # 埋点：意图分布 + 实际执行链路分布（1:1 路由，只在此处记一次，避免重复计数）
        incr("intent_total", intent=state.intent)
        incr("link_total", link=state.intent)
        return state.intent
    
    # -------- 4.4 节点4a: 网络搜索路径 --------
    def _node_web_search(self, state: AgentState) -> AgentState:
        """
        节点4a: 网络搜索
        ==================
        功能：调用 DuckDuckGo 搜索实时信息
        
        触发条件：意图识别为 web_search
        """
        logger.info(f"📍 [节点4a] 网络搜索: '{state.query}'")
        
        if self.searcher:
            incr("tool_call_total", tool="web_search")
            try:
                with timed("web_search_ms"):
                    state.search_result = self.searcher.search_formatted(state.query)
            except Exception:
                incr("tool_call_error_total", tool="web_search")
                raise
        else:
            state.search_result = "搜索功能未启用"
        
        return state
    
    # -------- 4.5 节点4b: RAG 知识库检索路径 --------
    def _node_rag_retrieve(self, state: AgentState) -> AgentState:
        """
        节点4b: RAG 知识库检索
        =======================
        功能：使用混合检索（向量+BM25）从知识库查找相关信息
        
        触发条件：意图识别为 rag_retrieve
        """
        logger.info(f"📍 [节点4b] RAG 检索: '{state.query}'")
        
        if self.rag:
            # 埋点：检索耗时 + 返回条数（检索质量的基础观测数据）
            with timed("rag_retrieval_ms"):
                retrieved = self.rag.hybrid_search(state.query)
            state.retrieved_docs = retrieved
            observe("rag_retrieval_docs", len(retrieved))
            if not retrieved:
                # 空召回是典型 badcase 线索（可能是分块/召回策略问题）
                record_event("rag_empty", query=state.query)
        else:
            state.retrieved_docs = ["知识库未加载"]
        
        return state
    
    # -------- 4.6 节点4c: 直接 LLM 回答路径 --------
    def _node_direct_llm(self, state: AgentState) -> AgentState:
        """
        节点4c: 直接 LLM 回答
        ======================
        功能：不检索外部信息，直接调用 LLM 回答
        
        触发条件：意图识别为 direct_llm
        """
        logger.info(f"📍 [节点4c] 直接 LLM 回答")
        # 此节点仅做标记，实际回答在 generate_response 节点生成
        return state
    
    # -------- 4.7 节点5: 生成个性化回答 --------
    def _node_generate_response(self, state: AgentState) -> AgentState:
        """
        节点5: 生成个性化回答
        ======================
        功能：
          1. 根据上下文组装 messages
          2. 调用 LLM 生成最终回答
          3. 回答风格会根据客户画像个性化
        
        上下文来源：
          - 用户信息（从 SQLite）
          - 对话历史（最近 N 轮）
          - 检索/搜索的结果
          - 当前用户问题
        """
        logger.info(f"📍 [节点5] 生成回答")
        
        # -------- 构建系统提示词 --------
        system_parts = [
            "你是一个专业的AI客户经营助手。",
            "",
            "【可用工具】",
            "你可以调用以下工具来帮助用户，请根据对话内容主动选择合适的工具：",
            "1. save_customer_profile — 当用户提供客户信息（如姓名、年龄、年级、意向课程、电话、标签等）时，",
            "   提取这些信息并调用此工具保存客户画像。注意：尽量提取完整信息后再保存。",
            "2. web_search — 当用户需要查询实时信息、新闻、行业动态、百科知识等内容时调用。",
            "3. sql_query — 当用户想查询客户数据库、统计数据或历史记录时调用。",
            "4. screen_anomaly_customers — 当用户询问「哪些客户该跟进」「有哪些异常/待提醒客户」时调用，"
            "返回三类清单：R1 有意向未跟进（超 10 天没跟）/ R2 流失沉睡风险（状态流失/沉睡，或超 30 天正价课未转化）/ "
            "R3 高风险低意向（超 45 天未激活/未转化）。",
            "",
            "【使用规则】",
            "- 用户提供客户个人信息时，应主动调用 save_customer_profile 保存",
            "- 调用工具后，根据工具返回结果生成友好、自然的回复",
            "- 如果不需要调用工具，直接回答用户问题即可",
        ]
        
        # 如果有客户画像，加入个性化信息
        if state.customer_info:
            customer = state.customer_info
            system_parts.append(
                f"当前客户信息：{customer.get('name', '未知')}"
                f"{'，年级：' + customer.get('school_grade', '') if customer.get('school_grade') else ''}"
                f"{'，意向课程：' + customer.get('intent_course', '') if customer.get('intent_course') else ''}"
                f"{'，标签：' + '、'.join(customer.get('tags', [])) if customer.get('tags') else ''}"
            )
        
        # -------- 构建对话消息 --------
        messages = [{"role": "system", "content": "\n".join(system_parts)}]
        
        # 添加对话历史（最近 MAX_HISTORY_ROUNDS 轮）
        messages.extend(state.history)
        
        # 添加检索/搜索上下文
        if state.intent == "web_search" and state.search_result:
            messages.append({
                "role": "system",
                "content": f"以下为实时搜索结果（供参考）：\n{state.search_result}"
            })
        
        if state.intent == "rag_retrieve" and state.retrieved_docs:
            context = "\n---\n".join(state.retrieved_docs)
            messages.append({
                "role": "system",
                "content": f"以下为知识库检索结果（供参考）：\n{context}"
            })
        
        # 添加用户问题
        messages.append({"role": "user", "content": state.query})
        
        # ==================== 调用 LLM（启用 Function Calling） ====================
        from tools import ToolRegistry
        import json
        import re
        
        registry = ToolRegistry()
        tool_defs = registry.get_tool_definitions()
        
        # 第一轮：调用 LLM，传入工具定义
        # LLM 可能选择直接回答，也可能选择调用工具
        # 埋点：最终回答生成的 LLM 调用耗时与次数（含 Function Calling 首轮）
        with timed("llm_latency_ms", stage="generate"):
            response = self.llm.invoke(messages, tools=tool_defs)
        incr("llm_call_total", stage="generate")
        
        # 检查 LLM 是否调用了工具函数
        tool_calls = response.get("tool_calls")
        if tool_calls:
            logger.info(f"[TOOL] LLM 请求了 {len(tool_calls)} 个工具调用")
            
            # 将助手消息（含 tool_calls）加入对话历史
            messages.append({
                "role": "assistant",
                "content": response.get("content", ""),
            })
            
            # 执行每个工具调用，并将结果加入对话
            for tc in tool_calls:
                func = tc["function"]
                func_name = func["name"]
                func_args = json.loads(func["arguments"])
                
                logger.info(f"[TOOL] 执行工具: {func_name}({func_args})")
                # 埋点：工具调用次数 / 耗时 / 失败率
                incr("tool_call_total", tool=func_name)
                try:
                    with timed("tool_latency_ms", tool=func_name):
                        tool_result = registry.execute(func_name, func_args)
                except Exception:
                    incr("tool_call_error_total", tool=func_name)
                    record_event("tool_error", tool=func_name, error="execute_failed")
                    raise
                
                messages.append({
                    "role": "tool",
                    "content": str(tool_result),
                    "name": func_name,
                })
            
            # 第二轮：调用 LLM，传入工具执行结果生成最终回答
            # 此时 messages 中已包含 tool 角色的消息，不再需要传入 tools
            logger.info("[REFRESH] 将工具结果反馈给 LLM 生成最终回答")
            with timed("llm_latency_ms", stage="generate_tool_followup"):
                response = self.llm.invoke(messages)
            incr("llm_call_total", stage="generate_tool_followup")
        
        else:
            content = response.get("content", "")
            
            # ============ 兜底方案A：空内容重试（移除 tools 参数）============
            # 根因：qwen-plus 在传入 tools 后进入"工具匹配模式"，
            #    如果对是否调用工具不确定，可能直接返回空内容
            # 策略：移除 tools 重新调用，让 LLM 正常回复
            if not content:
                logger.info("[RETRY-A] LLM 返回空内容且未调用工具，移除 tools 参数后重试")
                response = self.llm.invoke(messages)  # 不传 tools
                content = response.get("content", "")
                if content:
                    logger.info(f"[RETRY-A] 重试成功，获取到响应: {content[:60]}...")
            
            # ============ 兜底方案B：客户信息提取（关键词 + 正则双检测）============
            # 根因：qwen-plus 的 Function Calling 可能不总是触发 save_customer_profile
            # 场景：用户说"录入客户信息：明明 13岁..." 或 "小明 年龄：13岁 电话：1838..."
            # 检测策略：
            #   1. 关键词匹配（"录入", "客户信息" 等显式保存意图）
            #   2. 正则匹配（电话号码 + 姓名/年龄/性别 组合模式）
            
            # 检测条件1：关键字匹配
            SAVE_KEYWORDS = [
                "录入", "保存客户", "新增客户", "创建客户",
                "客户信息", "客户画像", "登记",
            ]
            keyword_match = any(kw in state.query for kw in SAVE_KEYWORDS)
            
            # 检测条件2：正则匹配客户信息模式
            #   - 有电话号码（手机号11位 / "电话：xxx" / "联系电话：xxx"）
            #   - 有个人信息描述（姓名/年龄/岁/性别/男/女/叫/名字）
            has_phone = bool(re.search(
                r'1[3-9]\d{9}|电话[：:\s]*\d{5,}|联系电话[：:\s]*\d{5,}',
                state.query
            ))
            has_personal_info = bool(re.search(
                r'(?:姓名|名字|叫|年龄|岁|性别|男|女|擅长|爱好|特点)',
                state.query
            ))
            pattern_match = has_phone and has_personal_info
            
            is_save_intent = keyword_match or pattern_match
            
            if is_save_intent:
                logger.info(
                    f"[FALLBACK-B] 客户信息意图命中 "
                    f"(keyword={keyword_match}, pattern={pattern_match})"
                    f"，执行兜底提取"
                )
                
                # 用 LLM 提取结构化客户信息（不依赖 Function Calling）
                extraction_prompt = [
                    {
                        "role": "system",
                        "content": (
                            "从用户输入中提取 K12 教育客户信息，以 JSON 格式返回（不要其他文字）：\n"
                            '{"name": "", "phone": "", "age": "", "identity": "", '
                            '"school_grade": "", "intent_course": "", "source": "", '
                            '"matched_course": "", "intent_price": "", "hourly_fee": "", '
                            '"tags": "", "notes": ""}\n'
                            "规则：\n"
                            "- name：客户姓名（必填）\n"
                            "- phone：联系电话\n"
                            "- age：年龄（数字）\n"
                            "- identity：身份：学生/家长/成人\n"
                            "- school_grade：就读年级，如 初二 / 高三\n"
                            "- intent_course：意向课程，如 初三数学一对一\n"
                            "- source：来源渠道，如 转介绍 / 地推 / 线上\n"
                            "- tags：标签，多个用逗号分隔（如'高意向,初三,数学'）\n"
                            "- notes：备注信息\n"
                            "- 没有的字段填空字符串"
                        )
                    },
                    {"role": "user", "content": state.query}
                ]
                
                try:
                    extraction = self.llm.invoke(extraction_prompt)
                    raw_content = extraction.get("content", "{}")
                    # 清理可能的 markdown 代码块标记
                    raw_content = re.sub(r'^```(?:json)?\s*|\s*```$', '', raw_content.strip())
                    info = json.loads(raw_content)
                    
                    if info.get("name"):
                        logger.info(
                            f"[FALLBACK-B] 提取到客户信息: "
                            f"name={info.get('name')}, phone={info.get('phone')}"
                        )
                        tool_result = registry.execute("save_customer_profile", {
                            "name": info.get("name", ""),
                            "phone": info.get("phone", ""),
                            "age": info.get("age", ""),
                            "identity": info.get("identity", ""),
                            "school_grade": info.get("school_grade", ""),
                            "intent_course": info.get("intent_course", ""),
                            "source": info.get("source", ""),
                            "matched_course": info.get("matched_course", ""),
                            "intent_price": info.get("intent_price", ""),
                            "hourly_fee": info.get("hourly_fee", ""),
                            "tags": info.get("tags", ""),
                            "notes": info.get("notes", ""),
                        })
                        
                        # 将保存结果加入消息上下文，供 LLM 生成最终回复时参考
                        messages.append({
                            "role": "system",
                            "content": (
                                f"[系统] 已自动保存客户「{info['name']}」的画像信息。"
                                f"结果：{tool_result}"
                            )
                        })
                        
                        # 用 LLM 生成友好回复
                        logger.info("[FALLBACK-B] 客户信息已保存，生成最终回复")
                        response = self.llm.invoke(messages)
                    else:
                        logger.warning(
                            f"[FALLBACK-B] 未能提取到客户姓名: {raw_content}"
                        )
                except Exception as e:
                    logger.error(f"[FALLBACK-B] 客户信息提取或保存失败: {e}")
        
        state.response = response.get("content", "")
        state.messages = messages  # 保存完整消息列表供后续使用
        
        return state
    
    # -------- 4.8 节点6: 保存对话历史 --------
    def _node_save_history(self, state: AgentState) -> AgentState:
        """
        节点6: 保存对话历史（支持多会话）
        ================================
        功能：将本轮对话保存到 SQLite
        
        保存内容：
          1. 用户消息（user）
          2. 助手回复（assistant）
        """
        logger.info(f"📍 [节点6] 保存对话历史: session={state.session_id}")
        
        from models import Message, ChatSession
        from datetime import datetime
        
        session_id = state.session_id or state.user_id
        
        # 确保会话记录存在（避免 FOREIGN KEY constraint failed）
        existing = self.db.get_session(session_id)
        if existing is None:
            fallback_session = ChatSession(
                session_id=session_id,
                account_id=state.user_id,
                title="新对话",
            )
            self.db.save_session(fallback_session)
        
        # 保存用户消息
        user_msg = Message(
            session_id=session_id,
            user_id=state.user_id,
            role="user",
            content=state.query,
        )
        self.db.save_message(user_msg)
        
        # 保存助手回复
        assistant_msg = Message(
            session_id=session_id,
            user_id=state.user_id,
            role="assistant",
            content=state.response,
        )
        self.db.save_message(assistant_msg)
        
        logger.info(f"📍 [节点6] 对话已保存")
        return state
    
    # ==================== 5. 运行工作流 ====================
    def run(
        self,
        query: str,
        user_id: str = "default_user",
        customer_info: dict = None,
        session_id: str = None,
        resume: bool = False,
    ) -> str:
        """
        运行工作流（对外入口）
        ======================
        参数：
          query         : 用户问题（resume=True 时可传 None，表示从检查点续跑）
          user_id       : 用户 ID（默认 usedefault_user）
          customer_info : 客户画像信息（可选）
          session_id    : 对话会话 ID（支持多对话框，同时作为检查点 thread_id）
          resume        : 是否从检查点恢复（用户终止后接着做）
        
        返回：
          助手回复文本
        
        使用示例：
          result = workflow.run("最新AI新闻")
          result = workflow.run("我们公司有哪些客户", user_id="u001")
          # 用户终止后接着做（从最后检查点续跑）：
          result = workflow.run(None, session_id="s-123", resume=True)
        """
        # 构造初始状态
        initial_state = AgentState(
            query=query or "",
            user_id=user_id,
            session_id=session_id or "",
            customer_info=customer_info or {},
        )

        # 检查点 config：同一 session 用同一 thread_id，实现"读档续跑"
        thread_id = session_id or user_id
        run_config = {"configurable": {"thread_id": thread_id}}

        # 执行工作流
        # [WARN] LangGraph 1.x 的 invoke() 返回 dict，用 .get() 取值
        logger.info(
            f"[START] 开始执行工作流: query='{(query or '')[:50]}...', "
            f"session={session_id}, resume={resume}"
        )
        # 埋点：端到端对话耗时 + 成功/失败（系统运行指标）
        try:
            with timed("chat_latency_ms"):
                if resume and self.checkpointer is not None:
                    # 从最后检查点继续：传 None 表示不注入新输入，直接续跑
                    logger.info(f"[CP] 从检查点恢复执行: thread={thread_id}")
                    final_state = self.graph.invoke(None, run_config)
                else:
                    final_state = self.graph.invoke(initial_state, run_config)
        except Exception:
            incr("chat_requests_total", status="error")
            record_event("chat_error", query=query or "")
            raise
        incr("chat_requests_total", status="success")
        response = final_state.get("response", "")
        observe("chat_response_chars", len(response))
        if not response:
            record_event("chat_empty_response", query=query or "")
        logger.info(f"[OK] 工作流执行完成: response_len={len(response)}")

        # 返回最终回答
        return response

    # ==================== 6. 中断恢复（检查点）对外接口 ====================
    def can_resume(self, session_id: str) -> bool:
        """判断某会话是否有可恢复的检查点（用户终止后能否接着做）"""
        if self.checkpointer is None:
            return False
        try:
            return get_checkpoint_manager().can_resume(session_id)
        except Exception as e:
            logger.warning(f"[WARN] 检查点查询失败: {e}")
            return False

    def last_checkpoint_node(self, session_id: str):
        """查看某会话最后停在哪个节点（前端提示"上次进行到…"）"""
        if self.checkpointer is None:
            return None
        try:
            return get_checkpoint_manager().last_node(session_id)
        except Exception:
            return None

    def clear_checkpoint(self, session_id: str) -> int:
        """任务彻底结束后清理检查点（释放存储成本）"""
        if self.checkpointer is None:
            return 0
        try:
            return get_checkpoint_manager().clear(session_id)
        except Exception:
            return 0
