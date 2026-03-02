# Project Jarvis: Prompt Engineering Evolution & Insights

This document captures the evolution of our prompt engineering strategies and the key lessons learned during the development of Project Jarvis. It serves as a guide for future prompt design and management.

## 1. Evolution of Prompt Protocols

### Phase 1: Unstructured Text (Legacy)
*   **Approach:** Early iterations used free-form text instructions.
*   **Issue:** LLM outputs were unpredictable, making it difficult to reliably extract actions or structured data. "Chatty" responses often broke downstream logic.

### Phase 2: XML Tags (Transitional)
*   **Approach:** Introduced XML-like tags (e.g., `<thought>`, `<answer>`, `<tool_call>`) to separate reasoning from content.
*   **Observation:** While better than free text, XML parsing proved fragile. LLMs would sometimes hallucinate closing tags or malform the structure, especially when generating code or complex data within the tags.

### Phase 3: JSON Protocol (Current Standard)
*   **Approach:** Migrated to a strict JSON schema with fields like `thought`, `type`, and `content`.
*   **Benefit:**
    *   **Machine Readability:** JSON is natively supported by most modern LLMs (via JSON mode) and is trivial to parse programmatically.
    *   **Structure:** Enforces a clear separation of concerns (reasoning vs. action).
    *   **Validation:** Allows the use of Pydantic models to validate the output structure.

## 2. Architecture: Centralized Prompt Management

### The "Prompt Sprawl" Problem
*   **Initial State:** Prompts were hardcoded as string constants scattered across various files (`agent.py`, `utils.py`, etc.).
*   **Consequence:** Updating the system persona or fixing a prompt bug required hunting through the entire codebase. Inconsistent styling and instruction quality became common.

### The Solution: `PromptManager`
*   **Implementation:** All prompts are now centralized in `src/core/prompt/manager.py`.
*   **Key Features:**
    *   **LangChain Integration:** Uses `ChatPromptTemplate` for structured message management.
    *   **Dynamic Injection:** Templates support runtime variable injection (e.g., `{todo_status}`, `{available_tools}`).
    *   **Versioning:** (Implicit) Changes to prompts are tracked in a single file, making git history cleaner and easier to audit.

## 3. Key Lessons Learned

### Lesson 1: Context Consistency is Critical
*   **Insight:** It is not enough to just format the *current* prompt correctly. The *entire conversation history* must respect the expected format.
*   **Example:** If we use a JSON system prompt, but the conversation history contains plain text user/assistant turns, the LLM may get confused and revert to plain text.
*   **Action:** We implemented history cleaning rules (`src/core/context/manager.py`) to ensure that even historical messages are stored or formatted to align with the current protocol (e.g., preserving `raw_json` responses).

### Lesson 2: Prompt Constraints != Guarantees
*   **Insight:** Even when explicitly told to "output ONLY JSON," LLMs will often wrap the JSON in Markdown code blocks (e.g., ` ```json ... ``` `) or add conversational filler.
*   **Action:** Robust Output Parsers are mandatory. Our `MasterNode` includes logic to strip Markdown formatting and try-catch blocks to handle malformed JSON, rather than blindly trusting the LLM.

### Lesson 3: Dynamic Prompts for Dynamic Agents
*   **Insight:** Static prompts fail for complex agents. An agent needs to know *what it has done* (Todo List status) and *what it can do* (Available Tools).
*   **Action:** Our `MasterNode` dynamically constructs the system prompt at runtime, injecting the current state of the LangGraph. This gives the LLM the "situational awareness" needed for multi-step reasoning.

## 4. Best Practices for Future Development

1.  **Always use `PromptManager`:** Never hardcode prompt strings in logic files.
2.  **Stick to JSON:** Unless there is a compelling reason (e.g., creative writing), use JSON for all internal agent communication and tool execution.
3.  **Validate Output:** Always assume the LLM might fail to follow the format. Use parsers like `JSONOutputParser` with repair capabilities.
4.  **Audit Context:** When debugging "bad" responses, check the entire message history sent to the LLM, not just the latest prompt. The error often lies in a malformed history entry.
