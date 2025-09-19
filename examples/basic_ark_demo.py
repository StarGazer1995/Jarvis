#!/usr/bin/env python3
"""
Basic ARK Engine Demonstration.

This script demonstrates the core functionality of the ARK (Agent Reactor Kernel) engine, including:
- Engine initialization and configuration
- Context management
- Intent recognition
- Decision making
- Tool execution simulation
- Response generation

Run from project root: python examples/basic_ark_demo.py --verbose
"""

import asyncio
import logging
import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.ark_engine import ARKEngine, ARKState, ARKDecision
from core.context_manager import ConversationContext, ConversationTurn
from core.intent_engine import ARKIntentEngine, IntentType
from core.mcp_client import ARKMCPClient
from core.security_manager import ARKSecurityManager


async def demonstrate_basic_ark_usage():
    """
    Demonstrate basic ARK engine usage with a simple conversation flow.
    """
    print("🚀 Starting Basic ARK Engine Demonstration")
    print("=" * 50)
    
    # Initialize components
    print("\n📦 Initializing ARK Components...")
    
    # Create ARK engine
    ark_engine = ARKEngine()
    print("✅ ARK Engine initialized")
    
    # Create context manager
    context_manager = ark_engine.context_manager
    print("✅ Context Manager ready")
    
    # Create intent engine
    intent_engine = ark_engine.intent_engine
    print("✅ Intent Engine ready")
    
    # Create security manager
    security_manager = ark_engine.security_manager
    print("✅ Security Manager ready")
    
    # Start the engine
    await ark_engine.start()
    print("✅ ARK Engine started")
    
    print(f"\n🔍 Engine State: {ark_engine.state}")
    
    # Demonstrate conversation flow
    print("\n💬 Demonstrating Conversation Flow")
    print("-" * 30)
    
    # Sample user inputs
    user_inputs = [
        "Hello, can you help me with file operations?",
        "I need to read a configuration file",
        "What tools are available for data processing?",
        "Can you search for information about Python?",
        "Thank you for your help!"
    ]
    
    for i, user_input in enumerate(user_inputs, 1):
        print(f"\n🗣️  User Input {i}: {user_input}")
        
        # Process the input through ARK engine
        try:
            response = await ark_engine.process_input(user_input)
            print(f"🤖 ARK Response: {response.content}")
            print(f"📊 Confidence: {response.confidence:.2f}")
            print(f"🎯 Intent: {response.intent_result.intent_type if response.intent_result else 'Unknown'}")
            
            if response.tools_used:
                print(f"🔧 Tools Used: {', '.join(response.tools_used)}")
            
            if response.context_updates:
                print(f"📝 Context Updates: {len(response.context_updates)} items")
                
        except Exception as e:
            print(f"❌ Error processing input: {e}")
    
    # Show conversation context
    print("\n📚 Conversation Context Summary")
    print("-" * 30)
    context = context_manager.get_current_context()
    print(f"Total Turns: {len(context.turns)}")
    print(f"Context ID: {context.context_id}")
    print(f"Created: {context.created_at}")
    
    # Show recent memory
    recent_memory = context_manager.get_recent_memory(limit=3)
    print(f"\n🧠 Recent Memory ({len(recent_memory)} items):")
    for memory in recent_memory:
        print(f"  - {memory['type']}: {memory['content'][:50]}...")
    
    # Stop the engine
    await ark_engine.stop()
    print("\n🛑 ARK Engine stopped")
    print("✅ Demonstration completed successfully!")


async def demonstrate_intent_recognition():
    """
    Demonstrate the intent recognition capabilities.
    """
    print("\n🎯 Demonstrating Intent Recognition")
    print("=" * 40)
    
    intent_engine = ARKIntentEngine()
    
    # Sample inputs with different intents
    test_inputs = [
        ("Can you help me read a file?", IntentType.TOOL_REQUEST),
        ("What's the weather like today?", IntentType.INFORMATION_REQUEST),
        ("Hello there!", IntentType.GREETING),
        ("Thank you for your help", IntentType.GRATITUDE),
        ("I need to process some data", IntentType.TASK_REQUEST),
        ("Goodbye!", IntentType.FAREWELL),
        ("This is just a random statement", IntentType.UNKNOWN)
    ]
    
    for text, expected_intent in test_inputs:
        result = await intent_engine.analyze_intent(text)
        
        print(f"\n📝 Input: '{text}'")
        print(f"🎯 Detected Intent: {result.intent_type}")
        print(f"📊 Confidence: {result.confidence:.2f}")
        
        if result.entities:
            print(f"🏷️  Entities: {[f'{e.text}({e.entity_type})' for e in result.entities]}")
        
        # Check if detection matches expectation
        status = "✅" if result.intent_type == expected_intent else "⚠️"
        print(f"{status} Expected: {expected_intent}")


async def demonstrate_security_validation():
    """
    Demonstrate security validation and policy enforcement.
    """
    print("\n🔒 Demonstrating Security Validation")
    print("=" * 40)
    
    security_manager = ARKSecurityManager()
    
    # Show available policies
    policies = security_manager.list_policies()
    print(f"📋 Available Security Policies: {', '.join(policies)}")
    
    # Test different security scenarios
    from core.security_manager import (
        SecurityContext, ValidationRequest, PermissionType
    )
    import time
    
    # Create a security context
    context = SecurityContext(
        user_id="demo_user",
        session_id="demo_session",
        tool_name="file_reader",
        tool_version="1.0.0",
        execution_id="demo_exec",
        timestamp=time.time()
    )
    
    # Test scenarios
    scenarios = [
        {
            "name": "Safe File Read",
            "permissions": {PermissionType.READ, PermissionType.FILESYSTEM},
            "policy": "low",
            "resources": ["file:///tmp/safe_file.txt"]
        },
        {
            "name": "Network Request",
            "permissions": {PermissionType.NETWORK},
            "policy": "medium",
            "resources": ["https://api.example.com/data"]
        },
        {
            "name": "System Operation",
            "permissions": {PermissionType.SYSTEM},
            "policy": "high",
            "resources": ["system://process_list"]
        },
        {
            "name": "Critical Operation",
            "permissions": {PermissionType.SYSTEM, PermissionType.WRITE},
            "policy": "critical",
            "resources": ["system://config_update"]
        }
    ]
    
    for scenario in scenarios:
        print(f"\n🧪 Testing: {scenario['name']}")
        
        request = ValidationRequest(
            context=context,
            tool_parameters={"action": scenario['name'].lower().replace(' ', '_')},
            requested_permissions=scenario['permissions'],
            target_resources=scenario['resources']
        )
        
        response = await security_manager.validate_tool_execution(
            request, scenario['policy']
        )
        
        print(f"🔍 Policy: {scenario['policy']}")
        print(f"📊 Result: {response.result}")
        print(f"💬 Message: {response.message}")
        
        if response.allowed_permissions:
            print(f"✅ Allowed: {[p.value for p in response.allowed_permissions]}")
        if response.denied_permissions:
            print(f"❌ Denied: {[p.value for p in response.denied_permissions]}")
    
    # Show security status
    status = await security_manager.get_security_status()
    print(f"\n📈 Security Status:")
    print(f"  - Policies: {status['policies_count']}")
    print(f"  - Rate Limiter: {'Active' if status['rate_limiter_active'] else 'Inactive'}")
    print(f"  - Approval Queue: {status['approval_queue_size']} items")


async def demonstrate_context_management():
    """
    Demonstrate context management and memory capabilities.
    """
    print("\n🧠 Demonstrating Context Management")
    print("=" * 40)
    
    from core.context_manager import ConversationContext, ConversationTurn
    
    # Create a conversation context
    context = ConversationContext()
    print(f"📝 Created context: {context.context_id}")
    
    # Simulate a conversation
    conversation_turns = [
        ("user", "Hello, I'm working on a Python project"),
        ("assistant", "Hello! I'd be happy to help with your Python project. What do you need assistance with?"),
        ("user", "I need to read data from a CSV file"),
        ("assistant", "I can help you read CSV data. You can use the pandas library or Python's built-in csv module."),
        ("user", "Can you show me an example with pandas?"),
        ("assistant", "Sure! Here's an example: import pandas as pd; df = pd.read_csv('your_file.csv')")
    ]
    
    for role, content in conversation_turns:
        turn = ConversationTurn(
            role=role,
            content=content,
            timestamp=time.time()
        )
        context.add_turn(turn)
        print(f"➕ Added {role} turn: {content[:50]}...")
    
    print(f"\n📊 Context Statistics:")
    print(f"  - Total turns: {len(context.turns)}")
    print(f"  - User turns: {len([t for t in context.turns if t.role == 'user'])}")
    print(f"  - Assistant turns: {len([t for t in context.turns if t.role == 'assistant'])}")
    
    # Demonstrate memory extraction
    context.add_memory("topic", "Python CSV processing")
    context.add_memory("user_skill", "beginner")
    context.add_memory("preferred_library", "pandas")
    
    print(f"\n🧠 Memory Items:")
    for key, value in context.memory.items():
        print(f"  - {key}: {value}")
    
    # Show context summary
    summary = context.get_summary()
    print(f"\n📋 Context Summary: {summary}")


async def main():
    """
    Main demonstration function.
    """
    parser = argparse.ArgumentParser(description="ARK Engine Basic Demonstration")
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--demo",
        choices=["all", "basic", "intent", "security", "context"],
        default="all",
        help="Choose which demonstration to run"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    print("🎭 ARK Engine Demonstration Suite")
    print("=" * 50)
    
    try:
        if args.demo in ["all", "basic"]:
            await demonstrate_basic_ark_usage()
        
        if args.demo in ["all", "intent"]:
            await demonstrate_intent_recognition()
        
        if args.demo in ["all", "security"]:
            await demonstrate_security_validation()
        
        if args.demo in ["all", "context"]:
            await demonstrate_context_management()
        
        print("\n🎉 All demonstrations completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠️  Demonstration interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        logging.exception("Demonstration error")
        return 1
    
    return 0


if __name__ == "__main__":
    import time
    exit_code = asyncio.run(main())
    sys.exit(exit_code)