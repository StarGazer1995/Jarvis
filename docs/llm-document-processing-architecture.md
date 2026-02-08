# LLM集成与文档处理混合架构方案

## 📋 概述

本文档描述了Jarvis项目中LLM集成与文档处理功能的混合架构设计方案。该方案旨在在保持现有轻量级架构的基础上，引入强大的文档处理和向量搜索能力。

## 🎯 设计原则

### 核心原则：**分层集成，各司其职**

```
┌─────────────────────────────────────────┐
│           Jarvis Agent (统一入口)          │
├─────────────────────────────────────────┤
│         ARK Engine (决策引擎)             │
├─────────────────────────────────────────┤
│  Core LLM     │    Document Processing   │
│  (轻量级)      │    (LangChain组件)       │
│  - OpenAI     │    - Vector DB          │
│  - LiteLLM    │    - Text Splitters     │
│  - Anthropic  │    - Document Loaders   │
└─────────────────────────────────────────┘
```

### 设计目标

1. **保持核心架构稳定性**：现有的ARK引擎和LLM客户端保持不变
2. **模块化扩展**：文档处理功能作为独立模块集成
3. **渐进式演进**：分阶段实施，降低风险
4. **可选依赖**：文档处理功能可选择性启用

## 🏗️ 架构设计

### 1. 保持现有核心架构

#### 核心组件（保持不变）
- ✅ **ARK引擎**：继续作为主要决策中心
- ✅ **LLM客户端**：保持轻量级的OpenAI + LiteLLM组合
- ✅ **现有功能**：对话、工具调用、上下文管理等

#### 核心依赖（必需）
```bash
# 使用 uv 添加核心依赖
uv add openai litellm "anthropic>=0.8.0"
```

### 2. 新增文档处理能力模块

#### 模块结构
```python
src/capabilities/
├── __init__.py
├── document_processor/
│   ├── __init__.py
│   ├── vector_store.py      # 向量数据库集成
│   ├── text_splitter.py     # 文档分割
│   ├── document_loader.py   # 文档加载
│   ├── retrieval_engine.py  # 检索引擎
│   └── config.py           # 配置管理
```

#### 可选依赖（文档处理）
```bash
# 使用 uv 添加文档处理依赖
uv add langchain-core langchain-community chromadb faiss-cpu tiktoken pypdf python-docx
```

## 🚀 实施路线图

### 阶段1：基础向量搜索 (2-3周)

#### 目标
- 基本的文档存储和检索功能
- 集成ChromaDB或Faiss
- 通过ARK工具接口暴露功能

#### 主要任务
1. 创建基础的向量存储接口
2. 实现文档嵌入和相似性搜索
3. 在ARK引擎中注册文档搜索工具
4. 编写基础测试用例

#### 交付物
- `vector_store.py` - 向量数据库抽象层
- `retrieval_engine.py` - 基础检索功能
- 集成测试和演示脚本

### 阶段2：文档处理增强 (3-4周)

#### 目标
- 完整的文档处理流水线
- 支持多种文档格式
- 智能分块策略

#### 主要任务
1. 引入LangChain的text splitters
2. 实现多格式文档加载器（PDF、Word、Markdown等）
3. 优化文档分块和嵌入策略
4. 添加文档元数据管理

#### 交付物
- `document_loader.py` - 多格式文档加载
- `text_splitter.py` - 智能文本分割
- 文档处理配置系统
- 性能优化和缓存机制

### 阶段3：高级检索 (4-5周)

#### 目标
- 智能检索和上下文增强
- 混合检索策略
- 与对话历史的深度集成

#### 主要任务
1. 实现混合检索（向量+关键词）
2. 上下文感知的文档检索
3. 与ARK引擎的深度集成
4. 检索结果排序和过滤

#### 交付物
- 高级检索算法
- 上下文感知机制
- 完整的文档处理工作流
- 性能监控和分析工具

## 🔧 技术选型

### 向量数据库选择

#### 1. ChromaDB（推荐用于开发和中小规模）
- **优势**：轻量级，易于集成，本地部署
- **适用场景**：开发环境，中小规模文档集合
- **集成方式**：直接嵌入应用

#### 2. Faiss（推荐用于大规模生产）
- **优势**：高性能，适合大规模数据，Facebook开源
- **适用场景**：生产环境，大规模文档检索
- **集成方式**：本地库或服务化部署

#### 3. Qdrant（推荐用于云原生）
- **优势**：功能丰富，云原生，API友好
- **适用场景**：云部署，需要高级功能
- **集成方式**：服务化部署

### LangChain组件选择

#### 精选使用策略
```python
# 只使用需要的组件，避免全量依赖
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
```

#### 组件功能映射
- **文档加载**：`PyPDFLoader`, `TextLoader`, `UnstructuredWordDocumentLoader`
- **文本分割**：`RecursiveCharacterTextSplitter`, `TokenTextSplitter`
- **向量存储**：`Chroma`, `FAISS`, `Qdrant`
- **嵌入模型**：`OpenAIEmbeddings`, `HuggingFaceEmbeddings`

## 🔗 架构集成

### ARK引擎集成示例

```python
# 在ARK引擎初始化时
class ARKEngine:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # ... 现有初始化代码 ...
        
        # 可选的文档处理功能
        if self.config.get('enable_document_processing', False):
            from capabilities.document_processor import DocumentProcessor
            self.doc_processor = DocumentProcessor(
                vector_db=self.config.get('vector_db', 'chromadb'),
                embedding_model=self.config.get('embedding_model', 'text-embedding-ada-002')
            )
            self._register_document_tools()
    
    def _register_document_tools(self):
        """注册文档处理相关工具"""
        self.register_tool('search_documents', self.doc_processor.search)
        self.register_tool('add_document', self.doc_processor.add_document)
        self.register_tool('list_documents', self.doc_processor.list_documents)
        self.register_tool('delete_document', self.doc_processor.delete_document)
```

### 配置文件支持

```yaml
# config/jarvis_config.yaml
llm:
  provider: "openai"
  model: "gpt-4"
  api_key: "${OPENAI_API_KEY}"

document_processing:
  enabled: true
  vector_db: "chromadb"
  vector_db_path: "./data/vector_db"
  embedding_model: "text-embedding-ada-002"
  chunk_size: 1000
  chunk_overlap: 200
  max_documents: 10000
  
  # 支持的文档类型
  supported_formats:
    - "pdf"
    - "txt"
    - "md"
    - "docx"
  
  # 检索配置
  retrieval:
    max_results: 5
    similarity_threshold: 0.7
    enable_hybrid_search: true
```

## 📊 性能考虑

### 内存管理
- **向量缓存**：合理设置向量缓存大小
- **文档分块**：优化分块大小和重叠策略
- **批处理**：大量文档处理时使用批处理

### 存储优化
- **索引策略**：选择合适的向量索引算法
- **压缩**：对向量数据进行压缩存储
- **清理机制**：定期清理过期或无用的向量数据

### 查询优化
- **预过滤**：使用元数据进行预过滤
- **缓存**：缓存常用查询结果
- **并行处理**：支持并行向量搜索

## 🧪 测试策略

### 单元测试
- 各个组件的独立功能测试
- 模拟数据的向量搜索测试
- 文档加载和分割功能测试

### 集成测试
- ARK引擎与文档处理模块的集成测试
- 端到端的文档处理工作流测试
- 不同向量数据库的兼容性测试

### 性能测试
- 大规模文档集合的处理性能
- 向量搜索的响应时间测试
- 内存和存储使用情况监控

## 🚀 部署考虑

### 开发环境
- 使用ChromaDB本地存储
- 小规模测试数据集
- 快速迭代和调试

### 生产环境
- 考虑使用Faiss或Qdrant
- 向量数据库的备份和恢复
- 监控和日志记录

### 扩展性
- 支持分布式向量存储
- 水平扩展能力
- 负载均衡和故障转移

## ⚖️ 方案优势

### ✅ 最佳平衡
- 保持核心系统的轻量级和可控性
- 获得LangChain在文档处理方面的强大能力
- 避免架构冲突和过度依赖

### ✅ 渐进式演进
- 可以逐步引入功能，降低风险
- 每个阶段都有明确的价值交付
- 保持向后兼容性

### ✅ 灵活配置
- 可选择性启用文档处理功能
- 支持不同的向量数据库和配置
- 便于测试和部署

### ✅ 可维护性
- 清晰的模块边界
- 独立的测试和部署
- 易于调试和优化

## 📝 后续计划

### 短期目标（1-2个月）
1. 完成阶段1的基础向量搜索功能
2. 建立完整的测试框架
3. 编写详细的使用文档

### 中期目标（3-6个月）
1. 完成阶段2和阶段3的所有功能
2. 性能优化和生产环境部署
3. 用户反馈收集和功能迭代

### 长期目标（6个月以上）
1. 支持更多的文档格式和数据源
2. 高级的语义搜索和推荐功能
3. 与其他AI能力的深度集成

---

**文档版本**: v1.0  
**创建日期**: 2024-10-07  
**最后更新**: 2024-10-07  
**作者**: Jarvis开发团队