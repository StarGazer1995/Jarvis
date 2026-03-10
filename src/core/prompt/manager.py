"""
提示词管理模块

这个模块负责管理系统提示词、对话模板和上下文构建。
"""

import logging
import json
from typing import Dict, List, Any, Optional
from datetime import datetime

from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage


class PromptManager:
    """提示词管理器"""

    def __init__(self):
        """初始化提示词管理器"""
        self.templates: Dict[str, ChatPromptTemplate] = {}
        self.logger = logging.getLogger("prompt.manager")
        self.load_default_templates()

    def _load_default_templates(self) -> None:
        """加载默认提示词模板"""

        # System Prompt
        self.templates["jarvis_system"] = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template("""# System Instruction
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
""")
            ]
        )

        # Conversation Response
        self.templates["conversation_response"] = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template("""# Task
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
""")
            ]
        )

        # Tool Usage
        self.templates["tool_usage_decision"] = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template("""# Task
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
""")
            ]
        )

        # Response Generation
        self.templates["response_generation"] = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template("""# Task
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
""")
            ]
        )

        # ReAct Agent System Prompt
        self.templates["react_system"] = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template("""{{
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
}}""")
            ]
        )

        # Master Node System Prompt
        self.templates["master_system"] = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template("""{{
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
}}""")
            ]
        )

        # Deep Research System Prompt
        self.templates["deep_research_system"] = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template("""{{
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
}}""")
            ]
        )

        # Deep Research Extractor Prompt
        self.templates["deep_research_extractor"] = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template("""{{
  "task": "Process the webpage content and user goal to extract relevant information.",
  "input": {{
    "webpage_content": "{webpage_content}",
    "goal": "{goal}"
  }},
  "guidelines": [
    "Rationale: Locate specific sections/data related to the goal.",
    "Evidence: Extract the most relevant information, preserving full original context (can be multiple paragraphs).",
    "Summary: Summarize the findings concisely and evaluate their contribution to the goal."
  ],
  "response_format": {{
    "type": "json_schema",
    "description": "You MUST output ONLY a valid JSON object. No markdown, no code blocks, no other text.",
    "schema": {{
      "rational": "...",
      "evidence": "...",
      "summary": "..."
    }}
  }}
}}""")
            ]
        )

        self.logger.info(f"加载了 {len(self.templates)} 个默认提示词模板")

    def load_default_templates(self) -> None:
        """
        公共方法：加载默认提示词模板
        """
        self._load_default_templates()

    def add_template(self, name: str, template: ChatPromptTemplate) -> None:
        """
        添加提示词模板

        Args:
            name: 模板名称
            template: 提示词模板
        """
        self.templates[name] = template
        self.logger.info(f"添加提示词模板: {name}")

    def get_template(self, name: str) -> Optional[ChatPromptTemplate]:
        """
        获取提示词模板

        Args:
            name: 模板名称

            Returns:
            提示词模板，如果不存在则返回None
        """
        return self.templates.get(name)

    def render_template(self, name: str, **kwargs) -> List[BaseMessage]:
        """
        渲染提示词模板

        Args:
            name: 模板名称
            **kwargs: 模板变量

            Returns:
            渲染后的消息列表
        """
        template = self.get_template(name)
        if not template:
            raise ValueError(f"提示词模板 '{name}' 不存在")

        try:
            # 渲染模板
            messages = template.format_messages(**kwargs)
            self.logger.debug(f"渲染提示词模板: {name}")
            return messages

        except Exception as e:
            self.logger.error(f"渲染提示词模板 '{name}' 失败: {e}")
            raise

    def build_conversation_messages(
        self,
        user_input: str,
        conversation_history: List[Dict[str, str]],
        system_context: Dict[str, Any],
        intent_info: Optional[Dict[str, Any]] = None,
        tool_results: Optional[List[Dict[str, Any]]] = None,
    ) -> List[BaseMessage]:
        """
        构建对话消息列表

        Args:
            user_input: 用户输入
            conversation_history: 对话历史
            system_context: 系统上下文
            intent_info: 意图信息
            tool_results: 工具结果

            Returns:
            BaseMessage消息列表
        """
        messages = []

        # 构建系统消息
        system_messages = self.render_template(
            "jarvis_system",
            agent_name=system_context.get("agent_name", "Jarvis"),
            current_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            user_preferences=json.dumps(
                system_context.get("user_preferences", {}), ensure_ascii=False
            ),
            available_tools=json.dumps(
                system_context.get("available_tools", []), ensure_ascii=False
            ),
        )
        messages.extend(system_messages)

        # 添加对话历史
        for entry in conversation_history[-10:]:  # 只保留最近10轮对话
            role = entry.get("role")
            content = entry.get("content", "")
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))

        # 添加当前用户输入
        messages.append(HumanMessage(content=user_input))

        return messages

    def build_tool_usage_prompt(
        self, user_input: str, intent: str, available_tools: List[Dict[str, Any]]
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

        messages = self.render_template(
            "tool_usage_decision",
            user_input=user_input,
            intent=intent,
            available_tools=tools_str,
        )
        return messages[0].content

    def build_response_generation_prompt(
        self,
        user_input: str,
        intent: str,
        confidence: float,
        entities: List[Dict[str, Any]],
        tool_results: List[Dict[str, Any]],
        context: Dict[str, Any],
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
        entities_str = (
            json.dumps(entities, ensure_ascii=False) if entities else "No entities"
        )
        tool_results_str = (
            json.dumps(tool_results, ensure_ascii=False)
            if tool_results
            else "No tool results"
        )
        context_str = (
            json.dumps(context, ensure_ascii=False)
            if context
            else "No additional context"
        )

        messages = self.render_template(
            "response_generation",
            user_input=user_input,
            intent=intent,
            confidence=confidence,
            entities=entities_str,
            tool_results=tool_results_str,
            context=context_str,
        )
        return messages[0].content

    def list_templates(self) -> List[str]:
        return list(self.templates.keys())
