# LLM配置示例

本目录包含了各种使用场景的LLM配置示例文件，帮助您快速开始使用Project Jarvis的配置驱动LLM系统。

## 配置文件说明

### 1. `minimal.yaml` - 最小化配置
- **适用场景**: 快速开始、简单测试
- **特点**: 最简配置，只包含必要的设置
- **默认提供商**: Mock
- **功能**: 基础对话功能

### 2. `development.yaml` - 开发环境配置
- **适用场景**: 本地开发、调试
- **特点**: 详细日志、快速响应、Mock提供商
- **默认提供商**: Mock
- **功能**: 完整的开发调试功能

### 3. `testing.yaml` - 测试环境配置
- **适用场景**: 自动化测试、CI/CD
- **特点**: 确定性输出、快速执行、无缓存
- **默认提供商**: Mock
- **功能**: 适合自动化测试的配置

### 4. `production.yaml` - 生产环境配置
- **适用场景**: 生产部署
- **特点**: 高可用性、性能优化、多提供商备份
- **默认提供商**: OpenAI
- **功能**: 完整的生产级功能

### 5. `multi_provider.yaml` - 多提供商配置
- **适用场景**: 需要使用多个LLM提供商
- **特点**: 支持OpenAI、Anthropic、Azure OpenAI、Ollama等
- **默认提供商**: OpenAI
- **功能**: 负载均衡、故障转移、成本优化

## 使用方法

### 1. 复制配置文件
将所需的示例配置文件复制到项目根目录的`config`文件夹：

```bash
# 复制开发环境配置
cp config/examples/development.yaml config/llm_config.yaml

# 或者复制生产环境配置
cp config/examples/production.yaml config/llm_config.yaml
```

### 2. 设置环境变量
根据配置文件中的要求设置相应的环境变量：

```bash
# OpenAI配置
export OPENAI_API_KEY="your-openai-api-key"
export OPENAI_ORG_ID="your-organization-id"

# Anthropic配置
export ANTHROPIC_API_KEY="your-anthropic-api-key"

# Azure OpenAI配置
export AZURE_OPENAI_API_KEY="your-azure-api-key"
export AZURE_OPENAI_ENDPOINT="your-azure-endpoint"
```

### 3. 在代码中使用
```python
from src.core.llm_factory import create_llm_client

# 使用默认提供商
client = create_llm_client()

# 使用指定提供商
client = create_llm_client(provider_name="openai")

# 使用指定环境
client = create_llm_client(environment="production")

# 使用自定义配置文件
client = create_llm_client(config_path="config/custom_config.yaml")
```

## 配置文件结构

### 全局配置 (`global`)
- `default_provider`: 默认LLM提供商
- `retry`: 重试策略配置
- `timeout`: 超时配置
- `logging`: 日志配置

### 提供商配置 (`providers`)
每个提供商包含：
- `enabled`: 是否启用
- `api_key`: API密钥（支持环境变量）
- `base_url`: API基础URL
- `default_model`: 默认模型
- `models`: 模型配置
- `timeout`: 提供商特定超时
- `retry`: 提供商特定重试策略
- `rate_limit`: 速率限制

### 环境配置 (`environments`)
- `debug`: 调试模式
- `log_requests`: 是否记录请求
- `log_responses`: 是否记录响应
- `validate_config`: 是否验证配置
- `cache_enabled`: 是否启用缓存

### 功能配置 (`features`)
- `streaming`: 流式传输配置
- `context`: 上下文管理配置
- `cache`: 缓存配置
- `monitoring`: 监控配置
- `security`: 安全配置

## 环境变量支持

配置文件支持使用环境变量，格式为`${VARIABLE_NAME}`：

```yaml
providers:
  openai:
    api_key: "${OPENAI_API_KEY}"
    organization: "${OPENAI_ORG_ID}"
```

## 自定义配置

您可以基于这些示例创建自己的配置文件：

1. 选择最接近您需求的示例配置
2. 复制并重命名配置文件
3. 根据您的需求修改配置项
4. 设置相应的环境变量
5. 在代码中指定配置文件路径

## 配置验证

系统会自动验证配置文件的有效性：
- 检查必要的配置项
- 验证提供商是否已注册
- 检查API密钥是否设置
- 验证模型配置的完整性

## 故障排除

### 常见问题

1. **提供商未启用**
   - 检查`providers.<provider>.enabled`是否为`true`
   - 确认提供商已在系统中注册

2. **API密钥错误**
   - 检查环境变量是否正确设置
   - 确认API密钥格式正确

3. **模型不存在**
   - 检查`default_model`是否在`models`中定义
   - 确认模型名称拼写正确

4. **超时错误**
   - 调整`timeout`配置
   - 检查网络连接

5. **配置文件格式错误**
   - 检查YAML语法
   - 确认缩进正确

### 调试技巧

1. 启用详细日志：
   ```yaml
   global:
     logging:
       level: "DEBUG"
   ```

2. 使用Mock提供商测试：
   ```yaml
   global:
     default_provider: "mock"
   ```

3. 启用配置验证：
   ```yaml
   environments:
     <environment>:
       validate_config: true
   ```

## 更多信息

- 查看`src/core/config_loader.py`了解配置加载逻辑
- 查看`src/core/llm_factory.py`了解工厂模式实现
- 查看`examples/`目录了解使用示例