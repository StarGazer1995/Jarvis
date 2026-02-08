# ARK System Examples

This directory contains demonstration scripts that showcase the capabilities of the ARK (Adaptive Reasoning Kernel) system. Each example focuses on specific aspects of the system and can be run independently.

## Available Examples

### 1. Basic ARK Demo (`basic_ark_demo.py`)
**Purpose**: Demonstrates the core ARK engine functionality
**Features**:
- ARK engine initialization and configuration
- Context management and conversation turns
- Intent recognition and entity extraction
- Security validation and tool execution simulation
- Response generation and decision making

**Usage**:
```bash
python examples/basic_ark_demo.py --verbose
```

### 2. MCP Integration Demo (`mcp_integration_demo.py`)
**Purpose**: Shows Model Context Protocol (MCP) server integration
**Features**:
- MCP server discovery and connection
- Tool registration and management
- Secure tool execution with validation
- Configuration persistence and management
- Error handling and recovery

**Usage**:
```bash
python examples/mcp_integration_demo.py --verbose
```

### 3. Context Management Demo (`context_management_demo.py`)
**Purpose**: Demonstrates conversation context handling
**Features**:
- Multi-turn conversation management
- Context persistence and restoration
- Memory optimization and cleanup
- Session management across restarts
- Context export/import functionality

**Usage**:
```bash
python examples/context_management_demo.py --verbose
```

### 4. Intent Recognition Demo (`intent_recognition_demo.py`)
**Purpose**: Shows intent recognition and entity extraction
**Features**:
- Basic intent classification
- Named entity recognition
- Pattern matching and confidence scoring
- Multi-language support
- Context-aware intent recognition

**Usage**:
```bash
python examples/intent_recognition_demo.py --verbose
```

### 5. Security Validation Demo (`security_validation_demo.py`)
**Purpose**: Demonstrates security features and validation
**Features**:
- Security policy enforcement
- Rate limiting and throttling
- Input validation and sanitization
- Audit logging and monitoring
- Risk assessment and scoring

**Usage**:
```bash
python examples/security_validation_demo.py --verbose
```

### 6. Complete System Demo (`complete_system_demo.py`)
**Purpose**: Comprehensive demonstration of the entire ARK system
**Features**:
- Full system initialization and configuration
- Multi-turn conversation flow simulation
- Tool integration and execution
- Context persistence across sessions
- Performance monitoring and metrics
- End-to-end workflow demonstration

**Usage**:
```bash
# Run complete demonstration
python examples/complete_system_demo.py --verbose

# Run specific demonstration modules
python examples/complete_system_demo.py --demo conversation
python examples/complete_system_demo.py --demo tools
python examples/complete_system_demo.py --demo context
python examples/complete_system_demo.py --demo performance

# Quick demonstration with fewer test cases
python examples/complete_system_demo.py --quick
```

## Running Examples

### Prerequisites
1. Ensure you're in the project root directory
2. Install all dependencies: `uv sync`
3. Set up any required environment variables

### General Usage Pattern
```bash
# Basic execution
python examples/<example_name>.py

# With verbose logging
python examples/<example_name>.py --verbose

# With specific options (varies by example)
python examples/<example_name>.py --help
```

### Common Command Line Options
- `--verbose` / `-v`: Enable detailed logging output
- `--help` / `-h`: Show help message and available options
- `--demo <type>`: Select specific demonstration module (where applicable)
- `--quick`: Run abbreviated version with fewer test cases

## Example Output

Each example provides structured output showing:
- ✅ Successful operations
- ❌ Failed operations
- ⚠️ Warnings or partial failures
- 📊 Statistics and metrics
- 🎯 Performance evaluations
- 📝 Step-by-step progress

## Integration with Main System

These examples demonstrate how to:
1. **Initialize** the ARK system with proper configuration
2. **Configure** various components (security, MCP, context management)
3. **Process** user inputs and generate responses
4. **Manage** conversation context and state
5. **Execute** tools safely with security validation
6. **Monitor** system performance and health
7. **Handle** errors and edge cases gracefully

## Development and Testing

### Adding New Examples
1. Create a new Python file in the `examples/` directory
2. Follow the existing naming convention: `<feature>_demo.py`
3. Include comprehensive docstrings and comments
4. Add command-line argument parsing for flexibility
5. Provide verbose logging options
6. Update this README with the new example

### Best Practices
- **Modularity**: Each example should be self-contained
- **Documentation**: Include clear descriptions and usage instructions
- **Error Handling**: Demonstrate proper error handling patterns
- **Logging**: Use structured logging with appropriate levels
- **Performance**: Include timing and performance metrics where relevant
- **Security**: Show security best practices and validation

### Testing Examples
```bash
# Test all examples
for example in examples/*.py; do
    echo "Testing $example"
    python "$example" --quick
done

# Test specific example with verbose output
python examples/complete_system_demo.py --verbose --demo all
```

## Troubleshooting

### Common Issues
1. **Import Errors**: Ensure you're running from the project root directory
2. **Missing Dependencies**: Run `uv sync`
3. **Permission Errors**: Check file permissions and security settings
4. **Network Issues**: Some examples may require internet connectivity

### Debug Mode
Enable debug logging for detailed troubleshooting:
```bash
python examples/<example_name>.py --verbose
```

### Getting Help
- Check the example's help message: `python examples/<example_name>.py --help`
- Review the source code for detailed implementation
- Check the main project documentation in `docs/README.md`

## Contributing

When contributing new examples:
1. Follow the existing code style and structure
2. Include comprehensive error handling
3. Add appropriate logging and output formatting
4. Test with various scenarios and edge cases
5. Update this README with the new example
6. Consider adding corresponding unit tests

---

**Note**: These examples are designed for demonstration and learning purposes. For production use, additional configuration, security hardening, and error handling may be required.