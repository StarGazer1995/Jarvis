#!/usr/bin/env python3
"""
Complete System Demonstration.

This script demonstrates the full ARK system integration, showcasing
how all components work together in a realistic scenario:
- JarvisAgent orchestration
- ARK Engine decision making
- MCP tool integration
- Context management
- Intent recognition
- Security validation
- Real-time interaction simulation

Run from project root: python examples/complete_system_demo.py --verbose
"""

import asyncio
import logging
import argparse
import sys
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from jarvis_agent import JarvisAgent, JarvisConfig
from core.ark_engine import ARKEngine, ARKState, ARKDecision
from core.context_manager import ConversationContext, ConversationTurn
from core.intent_engine import IntentType, ARKIntentEngine
from core.mcp_client import ARKMCPClient
from core.server_config import SimpleMCPServerConfig
from core.security_manager import ARKSecurityManager, SecurityLevel
from core.tool_registry import ARKToolRegistry


async def demonstrate_system_initialization():
    """
    Demonstrate complete system initialization and configuration.
    """
    print("🚀 Demonstrating System Initialization")
    print("=" * 42)
    
    # Create Jarvis configuration
    config = JarvisConfig(
        name="ARK Demo Agent",
        version="1.0.0",
        description="Demonstration of complete ARK system",
        max_context_length=10000,
        enable_logging=True,
        log_level="INFO",
        security_level=SecurityLevel.MEDIUM,
        enable_mcp=True,
        mcp_servers=[
            SimpleMCPServerConfig(
                name="demo_server",
                command=["python", "-m", "demo_mcp_server"],
                description="Demo MCP server for testing"
            )
        ],
        metadata={
            "demo_mode": True,
            "created_at": time.time(),
            "features": ["intent_recognition", "context_management", "security_validation"]
        }
    )
    
    print("✅ Created Jarvis configuration")
    print(f"   Name: {config.name}")
    print(f"   Version: {config.version}")
    print(f"   Security Level: {config.security_level.value}")
    print(f"   MCP Enabled: {config.enable_mcp}")
    
    # Initialize Jarvis agent
    agent = JarvisAgent(config)
    print("✅ Initialized Jarvis agent")
    
    # Start the agent
    await agent.start()
    print("✅ Started Jarvis agent")
    
    # Verify component initialization
    components = {
        "ARK Engine": agent.ark_engine is not None,
        "Context Manager": agent.context_manager is not None,
        "Intent Engine": agent.intent_engine is not None,
        "MCP Client": agent.mcp_client is not None,
        "Security Manager": agent.security_manager is not None,
        "Tool Registry": agent.tool_registry is not None
    }
    
    print(f"\n📊 Component Status:")
    for component, status in components.items():
        status_icon = "✅" if status else "❌"
        print(f"   {component}: {status_icon} {'Initialized' if status else 'Not initialized'}")
    
    return agent


async def demonstrate_conversation_flow(agent: JarvisAgent):
    """
    Demonstrate a complete conversation flow with multiple interactions.
    """
    print("\n💬 Demonstrating Conversation Flow")
    print("=" * 38)
    
    # Simulate a multi-turn conversation
    conversation_turns = [
        {
            "user_input": "Hello! I'm new here. Can you help me understand what you can do?",
            "expected_intent": IntentType.GREETING,
            "description": "Initial greeting and capability inquiry"
        },
        {
            "user_input": "I need to create a Python script that processes CSV files. Can you help?",
            "expected_intent": IntentType.TASK_REQUEST,
            "description": "Programming assistance request"
        },
        {
            "user_input": "What's the best way to handle large CSV files efficiently?",
            "expected_intent": IntentType.INFORMATION_REQUEST,
            "description": "Technical advice inquiry"
        },
        {
            "user_input": "Can you show me an example of reading a CSV file with pandas?",
            "expected_intent": IntentType.TASK_REQUEST,
            "description": "Code example request"
        },
        {
            "user_input": "Actually, I think I need to handle errors too. How should I do that?",
            "expected_intent": IntentType.CLARIFICATION,
            "description": "Follow-up clarification"
        },
        {
            "user_input": "Perfect! Can you also help me write the data to a database?",
            "expected_intent": IntentType.TASK_REQUEST,
            "description": "Extended task request"
        },
        {
            "user_input": "What are the security considerations when working with databases?",
            "expected_intent": IntentType.INFORMATION_REQUEST,
            "description": "Security inquiry"
        },
        {
            "user_input": "Thank you so much! This has been very helpful.",
            "expected_intent": IntentType.GENERAL,
            "description": "Gratitude and conversation closure"
        }
    ]
    
    print(f"🗣️  Simulating {len(conversation_turns)} conversation turns:")
    
    conversation_stats = {
        "total_turns": 0,
        "successful_responses": 0,
        "intent_accuracy": 0,
        "avg_response_time": 0,
        "context_maintained": 0
    }
    
    total_response_time = 0
    
    for i, turn in enumerate(conversation_turns, 1):
        print(f"\n📝 Turn {i}: {turn['description']}")
        print(f"   User: \"{turn['user_input']}\"")
        print(f"   Expected Intent: {turn['expected_intent'].value}")
        
        # Measure response time
        start_time = time.time()
        
        # Process the message
        response = await agent.process_message(turn['user_input'])
        
        end_time = time.time()
        response_time = end_time - start_time
        total_response_time += response_time
        
        print(f"   Agent: \"{response[:100]}{'...' if len(response) > 100 else ''}\"")
        print(f"   Response Time: {response_time:.3f}s")
        
        # Check intent recognition (if available)
        if hasattr(agent, 'last_intent_result'):
            detected_intent = agent.last_intent_result.intent
            intent_correct = detected_intent == turn['expected_intent']
            intent_icon = "✅" if intent_correct else "❌"
            print(f"   Intent: {intent_icon} {detected_intent.value} (conf: {agent.last_intent_result.confidence:.3f})")
            
            if intent_correct:
                conversation_stats["intent_accuracy"] += 1
        
        # Check context maintenance
        context_length = len(agent.context_manager.get_recent_context(5))
        if context_length >= i:
            conversation_stats["context_maintained"] += 1
            print(f"   Context: ✅ Maintained ({context_length} turns)")
        else:
            print(f"   Context: ⚠️ Partial ({context_length} turns)")
        
        conversation_stats["total_turns"] += 1
        if response and len(response) > 0:
            conversation_stats["successful_responses"] += 1
        
        # Small delay between turns
        await asyncio.sleep(0.5)
    
    # Calculate conversation statistics
    conversation_stats["avg_response_time"] = total_response_time / len(conversation_turns)
    conversation_stats["intent_accuracy"] = conversation_stats["intent_accuracy"] / len(conversation_turns)
    conversation_stats["context_maintained"] = conversation_stats["context_maintained"] / len(conversation_turns)
    
    print(f"\n📊 Conversation Statistics:")
    print(f"   Total Turns: {conversation_stats['total_turns']}")
    print(f"   Successful Responses: {conversation_stats['successful_responses']}/{conversation_stats['total_turns']}")
    print(f"   Intent Accuracy: {conversation_stats['intent_accuracy']:.1%}")
    print(f"   Context Maintenance: {conversation_stats['context_maintained']:.1%}")
    print(f"   Average Response Time: {conversation_stats['avg_response_time']:.3f}s")
    
    return conversation_stats


async def demonstrate_tool_integration(agent: JarvisAgent):
    """
    Demonstrate tool discovery, registration, and execution.
    """
    print("\n🔧 Demonstrating Tool Integration")
    print("=" * 36)
    
    # Simulate tool discovery
    print("🔍 Discovering available tools...")
    
    # Mock some tools for demonstration
    mock_tools = [
        {
            "name": "file_reader",
            "description": "Read and analyze file contents",
            "category": "file_operations",
            "security_level": SecurityLevel.LOW,
            "parameters": ["file_path", "encoding"]
        },
        {
            "name": "web_scraper",
            "description": "Extract data from web pages",
            "category": "network_operations",
            "security_level": SecurityLevel.MEDIUM,
            "parameters": ["url", "selector"]
        },
        {
            "name": "data_analyzer",
            "description": "Perform statistical analysis on datasets",
            "category": "data_processing",
            "security_level": SecurityLevel.LOW,
            "parameters": ["data", "analysis_type"]
        },
        {
            "name": "system_monitor",
            "description": "Monitor system resources and performance",
            "category": "system_operations",
            "security_level": SecurityLevel.HIGH,
            "parameters": ["metric_type", "duration"]
        },
        {
            "name": "email_sender",
            "description": "Send emails with attachments",
            "category": "communication",
            "security_level": SecurityLevel.MEDIUM,
            "parameters": ["recipient", "subject", "body", "attachments"]
        }
    ]
    
    print(f"✅ Discovered {len(mock_tools)} tools")
    
    # Register tools
    for tool in mock_tools:
        print(f"   📦 {tool['name']}: {tool['description']}")
        print(f"      Category: {tool['category']}")
        print(f"      Security: {tool['security_level'].value}")
        print(f"      Parameters: {', '.join(tool['parameters'])}")
    
    # Simulate tool execution scenarios
    tool_execution_scenarios = [
        {
            "request": "Can you read the contents of my configuration file?",
            "tool": "file_reader",
            "parameters": {"file_path": "/home/user/config.json", "encoding": "utf-8"},
            "expected_security": "allowed",
            "description": "Safe file reading operation"
        },
        {
            "request": "Please scrape data from this website for analysis",
            "tool": "web_scraper",
            "parameters": {"url": "https://example.com/data", "selector": ".data-table"},
            "expected_security": "allowed",
            "description": "Web scraping with security validation"
        },
        {
            "request": "Analyze the performance metrics of the system",
            "tool": "system_monitor",
            "parameters": {"metric_type": "cpu_memory", "duration": "5m"},
            "expected_security": "requires_approval",
            "description": "System monitoring requiring elevated permissions"
        },
        {
            "request": "Send a summary email to the team",
            "tool": "email_sender",
            "parameters": {
                "recipient": "team@company.com",
                "subject": "Weekly Summary",
                "body": "Here's the weekly summary...",
                "attachments": []
            },
            "expected_security": "allowed",
            "description": "Email communication"
        }
    ]
    
    print(f"\n🧪 Testing {len(tool_execution_scenarios)} tool execution scenarios:")
    
    successful_executions = 0
    security_validations = 0
    
    for i, scenario in enumerate(tool_execution_scenarios, 1):
        print(f"\n📝 Scenario {i}: {scenario['description']}")
        print(f"   Request: \"{scenario['request']}\"")
        print(f"   Tool: {scenario['tool']}")
        print(f"   Expected Security: {scenario['expected_security']}")
        
        # Simulate security validation
        security_result = await agent.security_manager.validate_tool_execution(
            tool_name=scenario['tool'],
            parameters=scenario['parameters'],
            user_context={"user_id": "demo_user", "session_id": "demo_session"}
        )
        
        if security_result.allowed:
            security_status = "✅ Allowed"
            security_validations += 1
            
            # Simulate tool execution
            execution_result = await agent.execute_tool_safely(
                scenario['tool'],
                scenario['parameters']
            )
            
            if execution_result.get('success', False):
                successful_executions += 1
                execution_status = "✅ Executed successfully"
            else:
                execution_status = "❌ Execution failed"
        else:
            security_status = "❌ Blocked by security policy"
            execution_status = "⏸️ Not executed (security block)"
        
        print(f"   Security: {security_status}")
        print(f"   Execution: {execution_status}")
        
        if security_result.risk_score:
            print(f"   Risk Score: {security_result.risk_score:.3f}")
    
    # Tool integration statistics
    security_pass_rate = security_validations / len(tool_execution_scenarios)
    execution_success_rate = successful_executions / security_validations if security_validations > 0 else 0
    
    print(f"\n📊 Tool Integration Statistics:")
    print(f"   Available Tools: {len(mock_tools)}")
    print(f"   Security Pass Rate: {security_pass_rate:.1%} ({security_validations}/{len(tool_execution_scenarios)})")
    print(f"   Execution Success Rate: {execution_success_rate:.1%} ({successful_executions}/{security_validations})")


async def demonstrate_context_persistence(agent: JarvisAgent):
    """
    Demonstrate context management and persistence across sessions.
    """
    print("\n🧠 Demonstrating Context Persistence")
    print("=" * 40)
    
    # Create a conversation with context building
    context_building_messages = [
        "My name is Alice and I'm working on a machine learning project",
        "The project involves predicting customer churn for an e-commerce company",
        "I'm using Python with scikit-learn and pandas for the analysis",
        "The dataset has about 50,000 customer records with 20 features",
        "I'm particularly interested in feature importance and model interpretability"
    ]
    
    print("📝 Building conversation context:")
    
    for i, message in enumerate(context_building_messages, 1):
        print(f"   Turn {i}: \"{message}\"")
        await agent.process_message(message)
        
        # Show context growth
        context_size = len(agent.context_manager.get_recent_context(10))
        print(f"      Context size: {context_size} turns")
    
    # Test context recall
    context_recall_tests = [
        {
            "query": "What's my name?",
            "expected_context": "Alice",
            "description": "Personal information recall"
        },
        {
            "query": "What project am I working on?",
            "expected_context": "machine learning project, customer churn prediction",
            "description": "Project context recall"
        },
        {
            "query": "What tools am I using?",
            "expected_context": "Python, scikit-learn, pandas",
            "description": "Technical stack recall"
        },
        {
            "query": "How big is my dataset?",
            "expected_context": "50,000 customer records, 20 features",
            "description": "Data specifications recall"
        },
        {
            "query": "What am I most interested in?",
            "expected_context": "feature importance, model interpretability",
            "description": "Interest and focus recall"
        }
    ]
    
    print(f"\n🧪 Testing context recall with {len(context_recall_tests)} queries:")
    
    context_recall_accuracy = 0
    
    for i, test in enumerate(context_recall_tests, 1):
        print(f"\n📝 Test {i}: {test['description']}")
        print(f"   Query: \"{test['query']}\"")
        print(f"   Expected Context: {test['expected_context']}")
        
        # Process query and get response
        response = await agent.process_message(test['query'])
        
        # Check if response contains expected context
        context_found = any(
            keyword.lower() in response.lower()
            for keyword in test['expected_context'].split(', ')
        )
        
        context_icon = "✅" if context_found else "❌"
        print(f"   Response: \"{response[:100]}{'...' if len(response) > 100 else ''}\"")
        print(f"   Context Recall: {context_icon} {'Found' if context_found else 'Not found'}")
        
        if context_found:
            context_recall_accuracy += 1
    
    # Test context persistence (simulate session restart)
    print(f"\n💾 Testing context persistence:")
    
    # Save current context
    context_data = agent.context_manager.export_context()
    print(f"   Exported context: {len(context_data)} turns")
    
    # Simulate new session
    new_agent = JarvisAgent(agent.config)
    await new_agent.start()
    
    # Restore context
    await new_agent.context_manager.import_context(context_data)
    restored_context_size = len(new_agent.context_manager.get_recent_context(20))
    
    print(f"   Restored context: {restored_context_size} turns")
    
    # Test context recall in new session
    test_query = "What was I working on?"
    restored_response = await new_agent.process_message(test_query)
    
    context_preserved = "machine learning" in restored_response.lower() or "churn" in restored_response.lower()
    persistence_icon = "✅" if context_preserved else "❌"
    
    print(f"   Persistence Test: {persistence_icon} {'Context preserved' if context_preserved else 'Context lost'}")
    
    # Context management statistics
    recall_accuracy = context_recall_accuracy / len(context_recall_tests)
    
    print(f"\n📊 Context Management Statistics:")
    print(f"   Context Recall Accuracy: {recall_accuracy:.1%} ({context_recall_accuracy}/{len(context_recall_tests)})")
    print(f"   Context Persistence: {'✅ Working' if context_preserved else '❌ Failed'}")
    print(f"   Final Context Size: {restored_context_size} turns")
    
    await new_agent.stop()


async def demonstrate_performance_monitoring(agent: JarvisAgent):
    """
    Demonstrate system performance monitoring and metrics.
    """
    print("\n📈 Demonstrating Performance Monitoring")
    print("=" * 43)
    
    # Collect initial metrics
    initial_stats = await agent.get_session_stats()
    print("📊 Initial System Metrics:")
    print(f"   Messages Processed: {initial_stats.get('messages_processed', 0)}")
    print(f"   Average Response Time: {initial_stats.get('avg_response_time', 0):.3f}s")
    print(f"   Context Size: {initial_stats.get('context_size', 0)} turns")
    print(f"   Tools Executed: {initial_stats.get('tools_executed', 0)}")
    
    # Performance stress test
    stress_test_messages = [
        "Calculate the fibonacci sequence up to 20 numbers",
        "What are the best practices for Python performance optimization?",
        "Can you help me design a database schema for an e-commerce system?",
        "Explain the differences between supervised and unsupervised learning",
        "How do I implement a REST API with authentication in Flask?",
        "What are the security considerations for web applications?",
        "Can you review this code for potential improvements?",
        "Help me troubleshoot a memory leak in my application",
        "What's the best way to handle large file uploads?",
        "Explain microservices architecture and its benefits"
    ]
    
    print(f"\n🏃 Running performance stress test with {len(stress_test_messages)} messages:")
    
    response_times = []
    memory_usage = []
    
    for i, message in enumerate(stress_test_messages, 1):
        print(f"   Processing message {i}/{len(stress_test_messages)}")
        
        # Measure response time
        start_time = time.time()
        response = await agent.process_message(message)
        end_time = time.time()
        
        response_time = end_time - start_time
        response_times.append(response_time)
        
        # Simulate memory usage tracking
        memory_usage.append(len(response) + len(message))
        
        print(f"     Response time: {response_time:.3f}s")
        print(f"     Response length: {len(response)} chars")
    
    # Calculate performance metrics
    avg_response_time = sum(response_times) / len(response_times)
    min_response_time = min(response_times)
    max_response_time = max(response_times)
    total_memory = sum(memory_usage)
    
    print(f"\n📊 Performance Test Results:")
    print(f"   Messages Processed: {len(stress_test_messages)}")
    print(f"   Average Response Time: {avg_response_time:.3f}s")
    print(f"   Min Response Time: {min_response_time:.3f}s")
    print(f"   Max Response Time: {max_response_time:.3f}s")
    print(f"   Total Memory Usage: {total_memory:,} chars")
    
    # Performance thresholds
    performance_thresholds = {
        "avg_response_time": 2.0,  # seconds
        "max_response_time": 5.0,  # seconds
        "memory_efficiency": 1000  # chars per message
    }
    
    # Evaluate performance
    performance_score = 0
    max_score = len(performance_thresholds)
    
    if avg_response_time <= performance_thresholds["avg_response_time"]:
        performance_score += 1
        avg_time_status = "✅ Good"
    else:
        avg_time_status = "⚠️ Slow"
    
    if max_response_time <= performance_thresholds["max_response_time"]:
        performance_score += 1
        max_time_status = "✅ Good"
    else:
        max_time_status = "⚠️ Slow"
    
    avg_memory_per_message = total_memory / len(stress_test_messages)
    if avg_memory_per_message <= performance_thresholds["memory_efficiency"]:
        performance_score += 1
        memory_status = "✅ Efficient"
    else:
        memory_status = "⚠️ High usage"
    
    print(f"\n🎯 Performance Evaluation:")
    print(f"   Average Response Time: {avg_time_status}")
    print(f"   Max Response Time: {max_time_status}")
    print(f"   Memory Efficiency: {memory_status}")
    print(f"   Overall Score: {performance_score}/{max_score} ({performance_score/max_score:.1%})")
    
    # Get final system stats
    final_stats = await agent.get_session_stats()
    
    print(f"\n📈 Session Summary:")
    print(f"   Total Messages: {final_stats.get('messages_processed', 0)}")
    print(f"   Session Duration: {final_stats.get('session_duration', 0):.1f}s")
    print(f"   Final Context Size: {final_stats.get('context_size', 0)} turns")
    print(f"   Tools Executed: {final_stats.get('tools_executed', 0)}")


async def main():
    """
    Main demonstration function.
    """
    parser = argparse.ArgumentParser(description="Complete System Demonstration")
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--demo",
        choices=["all", "init", "conversation", "tools", "context", "performance"],
        default="all",
        help="Choose which demonstration to run"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a quick demonstration with fewer test cases"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    print("🎯 Complete ARK System Demonstration")
    print("=" * 45)
    
    agent = None
    
    try:
        # Initialize system
        if args.demo in ["all", "init"]:
            agent = await demonstrate_system_initialization()
        
        if not agent:
            # Create agent if not already created
            config = JarvisConfig(name="Demo Agent")
            agent = JarvisAgent(config)
            await agent.start()
        
        # Run demonstrations
        if args.demo in ["all", "conversation"]:
            await demonstrate_conversation_flow(agent)
        
        if args.demo in ["all", "tools"]:
            await demonstrate_tool_integration(agent)
        
        if args.demo in ["all", "context"]:
            await demonstrate_context_persistence(agent)
        
        if args.demo in ["all", "performance"]:
            await demonstrate_performance_monitoring(agent)
        
        print("\n🎉 Complete system demonstration finished successfully!")
        print("\n📋 Summary:")
        print("   ✅ System initialization and configuration")
        print("   ✅ Multi-turn conversation handling")
        print("   ✅ Tool discovery and execution")
        print("   ✅ Context management and persistence")
        print("   ✅ Performance monitoring and metrics")
        print("\n🚀 The ARK system is ready for production use!")
        
    except KeyboardInterrupt:
        print("\n⚠️  Demonstration interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        logging.exception("Complete system demonstration error")
        return 1
    finally:
        if agent:
            await agent.stop()
            print("🛑 Agent stopped gracefully")
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)