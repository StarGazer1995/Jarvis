# 提示词系统对比报告：Jarvis vs LangChain

本报告旨在对比 Jarvis 项目当前的提示词系统与 LangChain 框架的提示词系统，分析两者在核心概念、数据结构及渲染输出上的差异，并提供迁移映射方案。

## 1. 核心概念对比

### 1.1 `PromptType` vs LangChain Message Types

**Jarvis (`PromptType`)**:
*   **定义**: `src/core/prompt/manager.py` 中的枚举 (`Enum`)。
*   **作用**: 对提示词模板进行**功能性分类**。
*   **取值**:
    *   `SYSTEM`: 系统指令。
    *   `CONVERSATION`: 对话生成。
    *   `TOOL_USAGE`: 工具使用决策。
    *   `INTENT_RECOGNITION`: 意图识别。
    *   `RESPONSE_GENERATION`: 最终响应生成。
*   **本质**: 它是对“用途”的描述，而不是对“消息角色”的描述。

**LangChain Message Types**:
*   **定义**: `langchain_core.messages` 中的类体系。
*   **作用**: 对消息在对话中的**角色（Role）进行结构化定义**。
*   **常见类型**:
    *   `SystemMessage`: 系统设定（对应 Role: system）。
    *   `HumanMessage`: 用户输入（对应 Role: user）。
    *   `AIMessage`: 模型回复（对应 Role: assistant）。
    *   `ToolMessage` / `FunctionMessage`: 工具调用结果。
    *   `ChatMessage`: 自定义 Role 的消息。
*   **对比**: Jarvis 的 `PromptType` 关注“这个提示词是用来做什么的”，而 LangChain 的 Message Types 关注“这句话是谁说的”。

### 1.2 `PromptTemplate` vs LangChain Templates

**Jarvis (`PromptTemplate`)**:
*   **定义**: `src/core/prompt/manager.py` 中的数据类 (`dataclass`)。
*   **结构**:
    *   包含 `template` (str): 纯文本模板字符串。
    *   包含 `variables` (List[str]): 变量名列表。
    *   包含 `type` (PromptType): 功能分类。
*   **特点**: 这是一个轻量级的字符串模板封装，主要用于管理文本替换。它不直接包含“角色”信息，通常在渲染后被放入 `LLMMessage` 中指定角色。

**LangChain Templates**:
*   **`PromptTemplate`**:
    *   用于生成**纯字符串**提示词。
    *   功能与 Jarvis 的 `PromptTemplate` 类似，支持 `input_variables` 和 `template_format` (f-string/jinja2)。
*   **`ChatPromptTemplate`**:
    *   用于生成**消息列表** (`List[BaseMessage]`)。
    *   由多个 `BaseMessagePromptTemplate` (如 `SystemMessagePromptTemplate`, `HumanMessagePromptTemplate`) 组成。
    *   支持复杂的聊天历史记录注入 (`MessagesPlaceholder`)。

### 1.3 渲染输出 (Rendering Output)

**Jarvis**:
*   **输出类型**: `str` (字符串)。
*   **流程**: `PromptManager.render_template` -> 接收变量 -> 执行字符串替换 -> 返回字符串。
*   **后续处理**: 调用者（如 Agent）通常需要手动将返回的字符串封装进 `LLMMessage(role="...", content=rendered_str)`。

**LangChain**:
*   **输出类型**:
    *   `format()` -> `str`: 返回单一字符串。
    *   `format_messages()` / `invoke()` -> `List[BaseMessage]`: 返回带有角色的消息对象列表。
*   **优势**: LangChain 的 `ChatPromptTemplate` 可以直接产出符合 LLM API 要求的消息列表结构，无需手动组装 Role。

## 2. 迁移映射表 (Migration Mapping Table)

如果将 Jarvis 的提示词系统迁移至 LangChain 风格，建议参考以下映射关系：

| Jarvis 概念/组件 | LangChain 对应组件 | 说明 |
| :--- | :--- | :--- |
| **PromptType.SYSTEM** | `SystemMessagePromptTemplate` | 系统提示词直接映射为 System 角色模板 |
| **PromptType.CONVERSATION** | `ChatPromptTemplate` | 对话通常包含历史记录，适合用 Chat 模板组合 |
| **PromptType.INTENT...** | `PromptTemplate` 或 `HumanMessagePromptTemplate` | 视具体用法而定，如果作为 System 指令的一部分则用 System，如果是用户输入分析则用 Human |
| **PromptTemplate (Class)** | `PromptTemplate` (Class) | 基础文本模板直接对应 |
| **PromptManager.render_template** | `template.format(**kwargs)` | 字符串格式化方法对应 |
| **LLMMessage** | `BaseMessage` (及其子类) | 数据结构对应 |
| **variables (List[str])** | `input_variables` | 变量定义对应 |

## 3. 总结与建议

1.  **角色与功能的解耦**: 当前 Jarvis 的 `PromptType` 混合了功能定义，但缺乏直接的角色绑定。LangChain 通过 `Message` 类明确了角色。
2.  **结构化能力的差异**: Jarvis 目前主要处理 String 操作。引入 LangChain 的 `ChatPromptTemplate` 概念可以更好地管理多轮对话和复杂的 Prompt 编排（如 Few-shot examples）。
3.  **迁移建议**:
    *   保留 `PromptType` 用于业务逻辑上的分类（如“这是意图识别任务”）。
    *   但在底层实现上，可以扩展 `PromptTemplate` 使其支持输出 `List[LLMMessage]`，或者引入类似 `ChatPromptTemplate` 的结构，将“角色”信息固化在模板定义中，而不是在运行时手动组装。
