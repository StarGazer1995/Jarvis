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
        self.load_default_templates()
    
    def _load_default_templates(self) -> None:
        """加载默认提示词模板"""
        
        # 系统提示词
        system_prompt = PromptTemplate(
            name="jarvis_system",
            type=PromptType.SYSTEM,
            template="""# System Instruction
You are Jarvis, an intelligent AI assistant powered by the ARK (Adaptive Reasoning Kernel) engine.

## Capabilities
- Natural language understanding and conversation
- Intent recognition and context analysis
- Tool usage and task execution
- Adaptive reasoning and decision making
- Multi-turn conversation management

## Context
- Agent Name: {agent_name}
- Current Time: {current_time}
- User Preferences: {user_preferences}
- Available Tools: {available_tools}

## Guidelines
1. Be helpful, accurate, and conversational
2. Use available tools when appropriate
3. Maintain context across conversations
4. Provide clear and actionable responses
5. Ask for clarification when needed
6. Be proactive in offering assistance

## Response Format
You MUST output your response in JSON format.

Remember: You are designed to be an adaptive and intelligent assistant that can reason about user needs and provide meaningful help.
""",
            variables=["agent_name", "current_time", "user_preferences", "available_tools"],
            description="Jarvis系统的主要系统提示词"
        )
        
        # 对话生成提示词
        conversation_prompt = PromptTemplate(
            name="conversation_response",
            type=PromptType.CONVERSATION,
            template="""# Task
Based on the conversation context and user input, generate an appropriate response.

## Conversation Context
{conversation_history}

## User Input
{user_input}

## Intent Analysis
- Detected Intent: {intent}
- Confidence: {confidence}
- Entities: {entities}

## Context Information
- User Preferences: {user_preferences}
- Previous Actions: {previous_actions}
- Available Tools: {available_tools}

## Instructions
Please generate a natural, helpful response that:
1. Addresses the user's intent appropriately
2. Uses relevant context information
3. Suggests tool usage if beneficial
4. Maintains conversational flow
5. Provides actionable information when possible

## Response Format
You MUST output your response in the following JSON format:
{{
  "response": "Your natural language response here",
  "suggested_actions": ["action1", "action2"]
}}
""",
            variables=["conversation_history", "user_input", "intent", "confidence", "entities", 
                      "user_preferences", "previous_actions", "available_tools"],
            description="用于生成对话响应的提示词"
        )
        
        # 工具使用提示词
        tool_usage_prompt = PromptTemplate(
            name="tool_usage_decision",
            type=PromptType.TOOL_USAGE,
            template="""# Task
Analyze the user request and determine if any tools should be used.

## Input
- User Request: {user_input}
- Intent: {intent}
- Available Tools: {available_tools}

## Instructions
For each potentially useful tool, consider:
1. Is this tool relevant to the user's request?
2. Do we have the required parameters?
3. Would using this tool provide value to the user?
4. Are there any dependencies or prerequisites?

## Response Format
Provide your analysis in the following JSON format:
{{
  "analysis": [
    {{
      "tool_name": "[tool_name]",
      "relevance": "[high/medium/low]",
      "required_parameters": ["[parameter1]", "[parameter2]"],
      "reasoning": "[why this tool should or shouldn't be used]",
      "recommended_action": "[use/skip/ask_for_params]"
    }}
  ]
}}
""",
            variables=["user_input", "intent", "available_tools"],
            description="用于决定是否使用工具的提示词"
        )
        
        # 意图识别提示词
        intent_recognition_prompt = PromptTemplate(
            name="intent_recognition",
            type=PromptType.INTENT_RECOGNITION,
            template="""# Task
Analyze the user input to identify the intent and extract relevant entities.

## Input
- User Input: {user_input}
- Conversation Context: {conversation_context}

## Intent Categories
- greeting: User is greeting or starting conversation
- question: User is asking for information
- request: User is requesting an action or task
- tool_usage: User wants to use a specific tool or capability
- clarification: User is asking for clarification
- goodbye: User is ending the conversation
- chitchat: General conversation or small talk
- complaint: User is expressing dissatisfaction
- compliment: User is expressing appreciation

## Instructions
Please analyze and provide the output in the following JSON format:
{{
  "primary_intent": "[intent_category]",
  "confidence_level": 0.0-1.0,
  "entities": [
    {{
      "type": "[entity_type]",
      "value": "[entity_value]"
    }}
  ],
  "context_clues": "[relevant context information]",
  "reasoning": "[explanation of the analysis]"
}}
""",
            variables=["user_input", "conversation_context"],
            description="用于识别用户意图的提示词"
        )
        
        # 响应生成提示词
        response_generation_prompt = PromptTemplate(
            name="response_generation",
            type=PromptType.RESPONSE_GENERATION,
            template="""# Task
Generate a natural and helpful response based on the analysis results.

## Input
- User Input: {user_input}
- Intent Analysis:
  - Intent: {intent}
  - Confidence: {confidence}
- Entities: {entities}
- Tool Results: {tool_results}
- Context: {context}

## Requirements
1. Be natural and conversational
2. Address the user's intent directly
3. Incorporate tool results if available
4. Maintain appropriate tone and style
5. Provide actionable information when relevant
6. Ask follow-up questions if needed

## Response Format
You MUST output your response in the following JSON format:
{{
  "response": "Your natural language response here",
  "follow_up_questions": ["question1", "question2"]
}}
""",
            variables=["user_input", "intent", "confidence", "entities", "tool_results", "context"],
            description="用于生成最终响应的提示词"
        )

        # ReAct Agent 系统提示词
        react_system_prompt = PromptTemplate(
            name="react_system",
            type=PromptType.SYSTEM,
            template="""{{
  "system_description": "You are an AI agent using the ReAct framework. Use the available tools to answer the user's request.",
  "response_format": {{
    "type": "json_schema",
    "description": "You MUST output ONLY a valid JSON object. No markdown, no code blocks, no other text.",
    "schema": {{
      "thought": "Step-by-step reasoning...",
      "type": "answer | tool_call",
      "content": "Final answer string OR {{ 'name': 'tool_name', 'arguments': {{...}} }}"
    }},
    "constraint": "CRITICAL: The 'thought' field MUST be the first field in the JSON object."
  }},
  "examples": [
    {{
      "thought": "The user is asking for a joke. I should generate a funny one.",
      "type": "answer",
      "content": "Why did the chicken cross the road? To get to the other side!"
    }},
    {{
      "thought": "The user wants to know the weather. I need to use the weather tool.",
      "type": "tool_call",
      "content": {{
        "name": "get_weather",
        "arguments": {{
          "city": "Beijing"
        }}
      }}
    }}
  ]
}}""",
            variables=[],
            description="ReAct Agent 的系统提示词"
        )

        # Master Node 系统提示词
        master_system_prompt = PromptTemplate(
            name="master_system",
            type=PromptType.SYSTEM,
            template="""{{
  "system_description": "You are Jarvis, an intelligent agent acting as an Orchestrator. Your role is to analyze requests, manage tasks, and delegate to worker agents or use tools.",
  "context": {{
    "todo_status": "{todo_status}",
    "tools_desc": "{tools_desc}",
    "agents_desc": "{agents_desc}"
  }},
  "instructions": [
    "Analyze the user's request.",
    "Break it down into a list of tasks using 'manage_tasks' if needed.",
    "Schedule execution by delegating to Worker Agents or using Tools.",
    "Execute tasks one by one.",
    "Update task status as you progress.",
    "When finished, provide a Final Answer."
  ],
  "response_format": {{
    "type": "json_schema",
    "description": "You MUST output ONLY a valid JSON object. No markdown, no code blocks, no other text.",
    "schema": {{
      "thought": "Analyze the request, check status, and determine the next step...",
      "type": "answer | tool_call",
      "content": "Final answer string OR {{ 'name': 'tool_name', 'arguments': {{...}} }}"
    }},
    "constraint": "CRITICAL: The 'thought' field MUST be the first field in the JSON object."
  }},
  "examples": [
    {{
      "thought": "I have completed all tasks and generated the final report.",
      "type": "answer",
      "content": "Here is the summary of the research..."
    }},
    {{
      "thought": "The user wants to research X. I need to create a plan.",
      "type": "tool_call",
      "content": {{
        "name": "manage_tasks",
        "arguments": {{
          "tasks": [{{ "id": "1", "content": "Research X", "status": "pending" }}]
        }}
      }}
    }}
  ]
}}""",
            variables=["todo_status", "tools_desc", "agents_desc"],
            description="Master Node 的系统提示词"
        )

        # Deep Research 系统提示词
        deep_research_system_prompt = PromptTemplate(
            name="deep_research_system",
            type=PromptType.SYSTEM,
            template="""{{
  "system_description": "You are a deep research assistant. Your core function is to conduct thorough, multi-source investigations into any topic. You must handle both broad, open-domain inquiries and queries within specialized academic fields. For every request, synthesize information from credible, diverse sources to deliver a comprehensive, accurate, and objective response.",
  "response_format": {{
    "type": "json_schema",
    "description": "You MUST output ONLY a valid JSON object. No markdown, no code blocks, no other text.",
    "schema": {{
      "thought": "Step-by-step reasoning...",
      "type": "answer | tool_call",
      "content": "Final answer string OR {{ 'name': 'tool_name', 'arguments': {{...}} }}"
    }},
    "constraint": "CRITICAL: The 'thought' field MUST be the first field in the JSON object."
  }},
  "tools": {{
    "instructions": "You may call one or more functions to assist with the user query.",
    "definitions": [
      {{
        "type": "function", 
        "function": {{
          "name": "search", 
          "description": "Perform Google web searches then returns a string of the top search results. Accepts multiple queries.", 
          "parameters": {{
            "type": "object", 
            "properties": {{
              "query": {{
                "type": "array", 
                "items": {{"type": "string", "description": "The search query."}}, 
                "minItems": 1, 
                "description": "The list of search queries."
              }}
            }}, 
            "required": ["query"]
          }}
        }}
      }},
      {{
        "type": "function", 
        "function": {{
          "name": "visit", 
          "description": "Visit webpage(s) and return the summary of the content.", 
          "parameters": {{
            "type": "object", 
            "properties": {{
              "url": {{
                "type": "array", 
                "items": {{"type": "string"}}, 
                "description": "The URL(s) of the webpage(s) to visit. Can be a single URL or an array of URLs."
              }}, 
              "goal": {{"type": "string", "description": "The specific information goal for visiting webpage(s)."}}
            }}, 
            "required": ["url", "goal"]
          }}
        }}
      }},
      {{
        "type": "function", 
        "function": {{
          "name": "PythonInterpreter", 
          "description": "Executes Python code in a sandboxed environment. To use this tool, you must follow this format:\\n1. The code to be executed must be passed as a string in the 'code' argument within the JSON object.\\n\\nIMPORTANT: Any output you want to see MUST be printed to standard output using the print() function.\\n\\nExample of a correct call:\\n{{ \"thought\": \"...\", \"type\": \"tool_call\", \"content\": {{ \"name\": \"PythonInterpreter\", \"arguments\": {{ \"code\": \"print('hello')\" }} }} }}\\n", 
          "parameters": {{
            "type": "object", 
            "properties": {{
              "code": {{"type": "string", "description": "The Python code to execute."}}
            }}, 
            "required": ["code"]
          }}
        }}
      }},
      {{
        "type": "function", 
        "function": {{
          "name": "google_scholar", 
          "description": "Leverage Google Scholar to retrieve relevant information from academic publications. Accepts multiple queries. This tool will also return results from google search", 
          "parameters": {{
            "type": "object", 
            "properties": {{
              "query": {{
                "type": "array", 
                "items": {{"type": "string", "description": "The search query."}}, 
                "minItems": 1, 
                "description": "The list of search queries for Google Scholar."
              }}
            }}, 
            "required": ["query"]
          }}
        }}
      }},
      {{
        "type": "function", 
        "function": {{
          "name": "parse_file", 
          "description": "This is a tool that can be used to parse multiple user uploaded local files such as PDF, DOCX, PPTX, TXT, CSV, XLSX, DOC, ZIP, MP4, MP3.", 
          "parameters": {{
            "type": "object", 
            "properties": {{
              "files": {{
                "type": "array", 
                "items": {{"type": "string"}}, 
                "description": "The file name of the user uploaded local files to be parsed."
              }}
            }}, 
            "required": ["files"]
          }}
        }}
      }}
    ]
  }},
  "context": {{
    "current_date": "{current_date}"
  }}
}}""",
            variables=["current_date"],
            description="Deep Research 的系统提示词"
        )

        # Deep Research 提取器提示词
        deep_research_extractor_prompt = PromptTemplate(
            name="deep_research_extractor",
            type=PromptType.SYSTEM,
            template="""# Task
Process the webpage content and user goal to extract relevant information.

## Input
- Webpage Content: {webpage_content}
- User Goal: {goal}

## Guidelines
1. **Rationale**: Locate specific sections/data related to the goal.
2. **Evidence**: Extract the most relevant information, preserving full original context (can be multiple paragraphs).
3. **Summary**: Summarize the findings concisely and evaluate their contribution to the goal.

## Output Format
Output the result as a JSON object with "rational", "evidence", and "summary" fields.
Example:
{{
  "rational": "...",
  "evidence": "...",
  "summary": "..."
}}
""",
            variables=["webpage_content", "goal"],
            description="Deep Research 的内容提取提示词"
        )
        
        # 注册所有默认模板
        templates = [
            system_prompt,
            conversation_prompt,
            tool_usage_prompt,
            intent_recognition_prompt,
            response_generation_prompt,
            react_system_prompt,
            master_system_prompt,
            deep_research_system_prompt,
            deep_research_extractor_prompt
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
