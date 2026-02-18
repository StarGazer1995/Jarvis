"""
提示词管理模块

这个模块负责管理系统提示词、对话模板和上下文构建。
"""

import logging
import json
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from ..llm.client import LLMMessage


class PromptType(Enum):
    """提示词类型"""
    SYSTEM = "system"
    CONVERSATION = "conversation"
    TOOL_USAGE = "tool_usage"
    INTENT_RECOGNITION = "intent_recognition"
    RESPONSE_GENERATION = "response_generation"


@dataclass
class PromptTemplate:
    """提示词模板"""
    name: str
    type: PromptType
    template: str
    variables: List[str] = field(default_factory=list)
    description: str = ""
    version: str = "1.0"
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class PromptManager:
    """提示词管理器"""
    
    def __init__(self):
        """初始化提示词管理器"""
        self.templates: Dict[str, PromptTemplate] = {}
        self.logger = logging.getLogger("prompt.manager")
        self._load_default_templates()
    
    def _load_default_templates(self) -> None:
        """加载默认提示词模板"""
        
        # 系统提示词
        system_prompt = PromptTemplate(
            name="jarvis_system",
            type=PromptType.SYSTEM,
            template="""<system_instruction>
You are Jarvis, an intelligent AI assistant powered by the ARK (Adaptive Reasoning Kernel) engine.

<capabilities>
- Natural language understanding and conversation
- Intent recognition and context analysis
- Tool usage and task execution
- Adaptive reasoning and decision making
- Multi-turn conversation management
</capabilities>

<context>
<agent_name>{agent_name}</agent_name>
<current_time>{current_time}</current_time>
<user_preferences>{user_preferences}</user_preferences>
<available_tools>{available_tools}</available_tools>
</context>

<guidelines>
1. Be helpful, accurate, and conversational
2. Use available tools when appropriate
3. Maintain context across conversations
4. Provide clear and actionable responses
5. Ask for clarification when needed
6. Be proactive in offering assistance
</guidelines>

Remember: You are designed to be an adaptive and intelligent assistant that can reason about user needs and provide meaningful help.
</system_instruction>""",
            variables=["agent_name", "current_time", "user_preferences", "available_tools"],
            description="Jarvis系统的主要系统提示词"
        )
        
        # 对话生成提示词
        conversation_prompt = PromptTemplate(
            name="conversation_response",
            type=PromptType.CONVERSATION,
            template="""<task>
Based on the conversation context and user input, generate an appropriate response.
</task>

<conversation_context>
{conversation_history}
</conversation_context>

<user_input>
{user_input}
</user_input>

<intent_analysis>
<detected_intent>{intent}</detected_intent>
<confidence>{confidence}</confidence>
<entities>{entities}</entities>
</intent_analysis>

<context_information>
<user_preferences>{user_preferences}</user_preferences>
<previous_actions>{previous_actions}</previous_actions>
<available_tools>{available_tools}</available_tools>
</context_information>

<instructions>
Please generate a natural, helpful response that:
1. Addresses the user's intent appropriately
2. Uses relevant context information
3. Suggests tool usage if beneficial
4. Maintains conversational flow
5. Provides actionable information when possible
</instructions>

<response>""",
            variables=["conversation_history", "user_input", "intent", "confidence", "entities", 
                      "user_preferences", "previous_actions", "available_tools"],
            description="用于生成对话响应的提示词"
        )
        
        # 工具使用提示词
        tool_usage_prompt = PromptTemplate(
            name="tool_usage_decision",
            type=PromptType.TOOL_USAGE,
            template="""<task>
Analyze the user request and determine if any tools should be used.
</task>

<input>
<user_request>{user_input}</user_request>
<intent>{intent}</intent>
<available_tools>{available_tools}</available_tools>
</input>

<instructions>
For each potentially useful tool, consider:
1. Is this tool relevant to the user's request?
2. Do we have the required parameters?
3. Would using this tool provide value to the user?
4. Are there any dependencies or prerequisites?

Provide your analysis in the following XML format:
<analysis>
<tool_name>[tool_name]</tool_name>
<relevance>[high/medium/low]</relevance>
<required_parameters>[list of parameters]</required_parameters>
<reasoning>[why this tool should or shouldn't be used]</reasoning>
<recommended_action>[use/skip/ask_for_params]</recommended_action>
</analysis>
</instructions>

<decision>""",
            variables=["user_input", "intent", "available_tools"],
            description="用于决定是否使用工具的提示词"
        )
        
        # 意图识别提示词
        intent_recognition_prompt = PromptTemplate(
            name="intent_recognition",
            type=PromptType.INTENT_RECOGNITION,
            template="""<task>
Analyze the user input to identify the intent and extract relevant entities.
</task>

<input>
<user_input>{user_input}</user_input>
<conversation_context>{conversation_context}</conversation_context>
</input>

<intent_categories>
- greeting: User is greeting or starting conversation
- question: User is asking for information
- request: User is requesting an action or task
- tool_usage: User wants to use a specific tool or capability
- clarification: User is asking for clarification
- goodbye: User is ending the conversation
- chitchat: General conversation or small talk
- complaint: User is expressing dissatisfaction
- compliment: User is expressing appreciation
</intent_categories>

<instructions>
Please analyze and provide the output in XML format:
<analysis>
<primary_intent>[intent_category]</primary_intent>
<confidence_level>[0.0-1.0]</confidence_level>
<entities>[list of extracted entities with types]</entities>
<context_clues>[relevant context information]</context_clues>
<reasoning>[explanation of the analysis]</reasoning>
</analysis>
</instructions>

<analysis>""",
            variables=["user_input", "conversation_context"],
            description="用于识别用户意图的提示词"
        )
        
        # 响应生成提示词
        response_generation_prompt = PromptTemplate(
            name="response_generation",
            type=PromptType.RESPONSE_GENERATION,
            template="""<task>
Generate a natural and helpful response based on the analysis results.
</task>

<input>
<user_input>{user_input}</user_input>
<intent_analysis>
<intent>{intent}</intent>
<confidence>{confidence}</confidence>
</intent_analysis>
<entities>{entities}</entities>
<tool_results>{tool_results}</tool_results>
<context>{context}</context>
</input>

<requirements>
1. Be natural and conversational
2. Address the user's intent directly
3. Incorporate tool results if available
4. Maintain appropriate tone and style
5. Provide actionable information when relevant
6. Ask follow-up questions if needed
</requirements>

<response>""",
            variables=["user_input", "intent", "confidence", "entities", "tool_results", "context"],
            description="用于生成最终响应的提示词"
        )
        
        # 注册所有默认模板
        templates = [
            system_prompt,
            conversation_prompt,
            tool_usage_prompt,
            intent_recognition_prompt,
            response_generation_prompt
        ]
        
        for template in templates:
            self.templates[template.name] = template
            self.logger.debug(f"加载提示词模板: {template.name}")
        
        self.logger.info(f"加载了 {len(templates)} 个默认提示词模板")
    
    def load_default_templates(self) -> None:
        """
        公共方法：加载默认提示词模板
        """
        self._load_default_templates()
    
    def add_template(self, template: PromptTemplate) -> None:
        """
        添加提示词模板
        
        Args:
            template: 提示词模板
        """
        self.templates[template.name] = template
        self.logger.info(f"添加提示词模板: {template.name}")
    
    def get_template(self, name: str) -> Optional[PromptTemplate]:
        """
        获取提示词模板
        
        Args:
            name: 模板名称
            
            Returns:
            提示词模板，如果不存在则返回None
        """
        return self.templates.get(name)
    
    def render_template(self, name: str, **kwargs) -> str:
        """
        渲染提示词模板
        
        Args:
            name: 模板名称
            **kwargs: 模板变量
            
            Returns:
            渲染后的提示词
        """
        template = self.get_template(name)
        if not template:
            raise ValueError(f"提示词模板 '{name}' 不存在")
        
        try:
            # 为缺失的变量提供默认值
            template_vars = {}
            for var in template.variables:
                if var in kwargs:
                    template_vars[var] = kwargs[var]
                else:
                    template_vars[var] = f"[{var}]"  # 占位符
                    self.logger.warning(f"模板变量 '{var}' 未提供，使用占位符")
            
            # 添加额外的变量
            for key, value in kwargs.items():
                if key not in template_vars:
                    template_vars[key] = value
            
            rendered = template.template.format(**template_vars)
            self.logger.debug(f"渲染提示词模板: {name}")
            return rendered
            
        except Exception as e:
            self.logger.error(f"渲染提示词模板 '{name}' 失败: {e}")
            raise
    
    def build_conversation_messages(
        self,
        user_input: str,
        conversation_history: List[Dict[str, str]],
        system_context: Dict[str, Any],
        intent_info: Optional[Dict[str, Any]] = None,
        tool_results: Optional[List[Dict[str, Any]]] = None
    ) -> List[LLMMessage]:
        """
        构建对话消息列表
        
        Args:
            user_input: 用户输入
            conversation_history: 对话历史
            system_context: 系统上下文
            intent_info: 意图信息
            tool_results: 工具结果
            
            Returns:
            LLM消息列表
        """
        messages = []
        
        # 构建系统消息
        system_prompt = self.render_template(
            "jarvis_system",
            agent_name=system_context.get("agent_name", "Jarvis"),
            current_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            user_preferences=json.dumps(system_context.get("user_preferences", {}), ensure_ascii=False),
            available_tools=json.dumps(system_context.get("available_tools", []), ensure_ascii=False)
        )
        
        messages.append(LLMMessage(
            role="system",
            content=system_prompt,
            metadata={"type": "system_context"}
        ))
        
        # 添加对话历史
        for entry in conversation_history[-10:]:  # 只保留最近10轮对话
            if entry.get("role") in ["user", "assistant"]:
                messages.append(LLMMessage(
                    role=entry["role"],
                    content=entry["content"],
                    metadata={"type": "history"}
                ))
        
        # 添加当前用户输入
        messages.append(LLMMessage(
            role="user",
            content=user_input,
            metadata={"type": "current_input"}
        ))
        
        return messages
    
    def build_intent_recognition_prompt(
        self,
        user_input: str,
        conversation_context: List[Dict[str, str]]
    ) -> str:
        """
        构建意图识别提示词
        
        Args:
            user_input: 用户输入
            conversation_context: 对话上下文
            
            Returns:
            意图识别提示词
        """
        context_str = ""
        if conversation_context:
            recent_context = conversation_context[-3:]  # 最近3轮对话
            context_str = "\n".join([
                f"{entry.get('role', 'unknown')}: {entry.get('content', '')}"
                for entry in recent_context
            ])
        
        return self.render_template(
            "intent_recognition",
            user_input=user_input,
            conversation_context=context_str or "No previous context"
        )
    
    def build_tool_usage_prompt(
        self,
        user_input: str,
        intent: str,
        available_tools: List[Dict[str, Any]]
    ) -> str:
        """
        构建工具使用决策提示词
        
        Args:
            user_input: 用户输入
            intent: 识别的意图
            available_tools: 可用工具列表
            
            Returns:
            工具使用决策提示词
        """
        tools_info = []
        for tool in available_tools:
            tool_info = f"- {tool.get('name', 'unknown')}: {tool.get('description', 'No description')}"
            tools_info.append(tool_info)
        
        tools_str = "\n".join(tools_info) if tools_info else "No tools available"
        
        return self.render_template(
            "tool_usage_decision",
            user_input=user_input,
            intent=intent,
            available_tools=tools_str
        )
    
    def build_response_generation_prompt(
        self,
        user_input: str,
        intent: str,
        confidence: float,
        entities: List[Dict[str, Any]],
        tool_results: List[Dict[str, Any]],
        context: Dict[str, Any]
    ) -> str:
        """
        构建响应生成提示词
        
        Args:
            user_input: 用户输入
            intent: 识别的意图
            confidence: 置信度
            entities: 提取的实体
            tool_results: 工具执行结果
            context: 上下文信息
            
            Returns:
            响应生成提示词
        """
        entities_str = json.dumps(entities, ensure_ascii=False) if entities else "No entities"
        tool_results_str = json.dumps(tool_results, ensure_ascii=False) if tool_results else "No tool results"
        context_str = json.dumps(context, ensure_ascii=False) if context else "No additional context"
        
        return self.render_template(
            "response_generation",
            user_input=user_input,
            intent=intent,
            confidence=confidence,
            entities=entities_str,
            tool_results=tool_results_str,
            context=context_str
        )
    
    def list_templates(self) -> List[str]:
        """
        列出所有模板名称
        
        Returns:
            模板名称列表
        """
        return list(self.templates.keys())
    
    def get_template_info(self, name: str) -> Optional[Dict[str, Any]]:
        """
        获取模板信息
        
        Args:
            name: 模板名称
            
            Returns:
            模板信息字典
        """
        template = self.get_template(name)
        if not template:
            return None
        
        return {
            "name": template.name,
            "type": template.type.value,
            "description": template.description,
            "variables": template.variables,
            "version": template.version,
            "created_at": template.created_at.isoformat()
        }
    
    def export_templates(self) -> Dict[str, Any]:
        """
        导出所有模板
        
        Returns:
            模板数据字典
        """
        exported = {}
        for name, template in self.templates.items():
            exported[name] = {
                "name": template.name,
                "type": template.type.value,
                "template": template.template,
                "variables": template.variables,
                "description": template.description,
                "version": template.version,
                "created_at": template.created_at.isoformat(),
                "metadata": template.metadata
            }
        
        return exported
    
    def import_templates(self, templates_data: Dict[str, Any]) -> int:
        """
        导入模板
        
        Args:
            templates_data: 模板数据字典
            
            Returns:
            导入的模板数量
        """
        imported_count = 0
        
        for name, data in templates_data.items():
            try:
                template = PromptTemplate(
                    name=data["name"],
                    type=PromptType(data["type"]),
                    template=data["template"],
                    variables=data.get("variables", []),
                    description=data.get("description", ""),
                    version=data.get("version", "1.0"),
                    created_at=datetime.fromisoformat(data.get("created_at", datetime.now().isoformat())),
                    metadata=data.get("metadata", {})
                )
                
                self.add_template(template)
                imported_count += 1
                
            except Exception as e:
                self.logger.error(f"导入模板 '{name}' 失败: {e}")
        
        self.logger.info(f"成功导入 {imported_count} 个模板")
        return imported_count
