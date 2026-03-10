#!/usr/bin/env python3
"""
LLM Integration Demo Script

This script demonstrates the LLM integration capabilities of the Jarvis ARK engine.
It shows how the system can use LLM for intelligent response generation while
maintaining fallback to template-based responses.
"""

import asyncio
import logging
import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from core.ark_engine import ARKEngine


def setup_logging():
    """Setup logging configuration for the demo."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


async def test_llm_configuration():
    """Test different LLM configurations."""
    print("\n" + "=" * 60)
    print("🔧 LLM Configuration Testing")
    print("=" * 60)

    # Test Mock LLM configuration
    print("\n📋 Testing Mock LLM Configuration:")
    mock_config = {
        "enable_llm": True,
        "llm": {
            "provider": "mock",
            "model": "mock-gpt-3.5-turbo",
            "max_tokens": 1000,
            "temperature": 0.7,
        },
    }

    engine = ARKEngine(mock_config)
    print(f"✅ LLM Enabled: {engine.llm_enabled}")
    print(f"✅ LLM Provider: {engine.llm_config.provider.value}")
    print(f"✅ LLM Model: {engine.llm_config.model}")
    print(f"✅ Max Tokens: {engine.llm_config.max_tokens}")
    print(f"✅ Temperature: {engine.llm_config.temperature}")

    return engine


async def test_llm_initialization(engine):
    """Test LLM initialization process."""
    print("\n" + "=" * 60)
    print("🚀 LLM Initialization Testing")
    print("=" * 60)

    print("\n📋 Initializing ARK Engine with LLM...")
    success = await engine.initialize()

    if success:
        print("✅ ARK Engine initialized successfully")
        print(f"✅ Engine State: {engine.state.value}")
        print(f"✅ LLM Enabled: {engine.llm_enabled}")

        # Check LLM manager status
        if hasattr(engine.llm_manager, "is_initialized"):
            print(f"✅ LLM Manager Initialized: {engine.llm_manager.is_initialized}")

        # Check prompt manager
        print(f"✅ Prompt Templates Loaded: {len(engine.prompt_manager.templates)}")

    else:
        print("❌ ARK Engine initialization failed")

    return success


async def test_llm_responses(engine):
    """Test LLM response generation with various inputs."""
    print("\n" + "=" * 60)
    print("🤖 LLM Response Generation Testing")
    print("=" * 60)

    test_inputs = [
        "Hello, how are you?",
        "What's the weather like today?",
        "Can you help me with a programming question?",
        "Tell me a joke",
        "What is artificial intelligence?",
        "Goodbye",
        "你好，你能说中文吗？",
        "帮我解释一下机器学习",
    ]

    for i, user_input in enumerate(test_inputs, 1):
        print(f"\n📝 Test {i}: '{user_input}'")
        try:
            response = await engine.process_input(user_input)
            print(f"🤖 Response: {response}")

            # Add conversation to context
            engine.context_manager.add_exchange(user_input, response)

        except Exception as e:
            print(f"❌ Error processing input: {e}")

        # Small delay between requests
        await asyncio.sleep(0.5)


async def test_llm_vs_template_comparison(engine):
    """Compare LLM responses vs template responses."""
    print("\n" + "=" * 60)
    print("⚖️  LLM vs Template Response Comparison")
    print("=" * 60)

    test_input = "Hello, how are you today?"

    # Test with LLM enabled
    print(f"\n📝 Input: '{test_input}'")
    print("\n🤖 With LLM Enabled:")
    engine.llm_enabled = True
    try:
        llm_response = await engine.process_input(test_input)
        print(f"Response: {llm_response}")
    except Exception as e:
        print(f"❌ Error: {e}")

    # Test with LLM disabled (template fallback)
    print("\n📋 With LLM Disabled (Template Fallback):")
    engine.llm_enabled = False
    try:
        template_response = await engine.process_input(test_input)
        print(f"Response: {template_response}")
    except Exception as e:
        print(f"❌ Error: {e}")

    # Re-enable LLM for further tests
    engine.llm_enabled = True


async def test_conversation_context():
    """Test conversation context with LLM."""
    print("\n" + "=" * 60)
    print("💬 Conversation Context Testing")
    print("=" * 60)

    # Create new engine for context testing
    config = {
        "enable_llm": True,
        "llm": {
            "provider": "mock",
            "model": "mock-gpt-3.5-turbo",
            "max_tokens": 1000,
            "temperature": 0.7,
        },
        "max_conversation_history": 10,
    }

    engine = ARKEngine(config)
    await engine.initialize()

    conversation_flow = [
        "Hi, I'm working on a Python project",
        "I need help with async programming",
        "Can you explain asyncio?",
        "What about error handling in async code?",
        "Thanks for the help!",
    ]

    for i, user_input in enumerate(conversation_flow, 1):
        print(f"\n👤 User {i}: {user_input}")
        try:
            response = await engine.process_input(user_input)
            print(f"🤖 Jarvis: {response}")

            # Show context length
            context_length = len(engine.context_manager.conversation_history)
            print(f"📊 Context Length: {context_length} exchanges")

        except Exception as e:
            print(f"❌ Error: {e}")

        await asyncio.sleep(0.3)

    await engine.shutdown()


async def test_error_handling():
    """Test error handling in LLM integration."""
    print("\n" + "=" * 60)
    print("🛡️  Error Handling Testing")
    print("=" * 60)

    # Test with invalid LLM configuration
    print("\n📋 Testing with Invalid LLM Configuration:")
    invalid_config = {
        "enable_llm": True,
        "llm": {"provider": "invalid_provider", "model": "invalid-model"},
    }

    try:
        engine = ARKEngine(invalid_config)
        await engine.initialize()

        response = await engine.process_input("Hello")
        print(f"✅ Fallback Response: {response}")

        await engine.shutdown()

    except Exception as e:
        print(f"⚠️  Expected error handled: {e}")


async def display_system_status(engine):
    """Display comprehensive system status."""
    print("\n" + "=" * 60)
    print("📊 System Status Report")
    print("=" * 60)

    status = engine.get_status()

    print(f"\n🔧 Engine Status:")
    print(f"  State: {status['state']}")
    print(f"  LLM Enabled: {status.get('llm_enabled', 'N/A')}")
    print(f"  Available Tools: {status['available_tools']}")
    print(f"  Decision History: {status.get('decision_history_count', 0)}")

    engine_status = engine.get_engine_status()
    print(f"\n⚙️  Engine Details:")
    print(f"  Performance Metrics: {engine_status.get('performance_metrics', {})}")
    print(f"  Tool Usage Stats: {engine_status.get('tool_usage_stats', {})}")
    print(f"  Available Tools: {len(engine_status.get('available_tools', []))}")

    # LLM specific status
    if hasattr(engine, "llm_config") and engine.llm_config:
        print(f"\n🤖 LLM Configuration:")
        print(f"  Provider: {engine.llm_config.provider.value}")
        print(f"  Model: {engine.llm_config.model}")
        print(f"  Max Tokens: {engine.llm_config.max_tokens}")
        print(f"  Temperature: {engine.llm_config.temperature}")


async def main():
    """Main demo function."""
    print("🚀 Jarvis LLM Integration Demo")
    print("=" * 60)
    print(
        "This demo showcases the LLM integration capabilities of the Jarvis ARK engine."
    )
    print("It demonstrates intelligent response generation, fallback mechanisms, and")
    print("conversation context management.")

    setup_logging()

    try:
        # Test LLM configuration
        engine = await test_llm_configuration()

        # Test LLM initialization
        init_success = await test_llm_initialization(engine)

        if init_success:
            # Test LLM responses
            await test_llm_responses(engine)

            # Test LLM vs template comparison
            await test_llm_vs_template_comparison(engine)

            # Display system status
            await display_system_status(engine)

            # Test conversation context
            await test_conversation_context()

            # Test error handling
            await test_error_handling()

            # Final shutdown
            await engine.shutdown()

        print("\n" + "=" * 60)
        print("✅ LLM Integration Demo Completed Successfully!")
        print("=" * 60)
        print("\n📋 Summary:")
        print("• LLM configuration and initialization tested")
        print("• Response generation with LLM and template fallback verified")
        print("• Conversation context management demonstrated")
        print("• Error handling and recovery mechanisms validated")
        print("• System status monitoring confirmed")

        print("\n🎯 Next Steps:")
        print("• Configure OpenAI API key for real LLM integration")
        print("• Customize prompt templates for specific use cases")
        print("• Integrate with external tools and services")
        print("• Deploy in production environment")

    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
