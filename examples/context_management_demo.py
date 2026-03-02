#!/usr/bin/env python3
"""
Context Management Demonstration.

This script demonstrates the conversation context management
capabilities of the ARK system, including:
- Conversation turn creation and management
- Context persistence and retrieval
- Memory management and optimization
- Context-aware decision making
- Multi-session context handling

Run from project root: python examples/context_management_demo.py --verbose
"""

import asyncio
import logging
import argparse
import sys
import json
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.context_manager import ConversationTurn, ConversationContext
from core.intent_engine import IntentType, Entity, IntentResult


async def demonstrate_conversation_turns():
    """
    Demonstrate conversation turn creation and management.
    """
    print("💬 Demonstrating Conversation Turns")
    print("=" * 40)

    # Create sample conversation turns
    turns = []

    # Turn 1: User greeting
    turn1 = ConversationTurn(
        turn_id="turn_001",
        user_input="Hello, I need help with file management",
        intent=IntentType.TASK_REQUEST,
        entities=[
            Entity(type="domain", value="file_management", confidence=0.9),
            Entity(type="action", value="help", confidence=0.8),
        ],
        context_data={
            "user_mood": "neutral",
            "session_start": True,
            "preferred_style": "helpful",
        },
        timestamp=time.time(),
    )
    turns.append(turn1)
    print(f"✅ Created Turn 1: {turn1.turn_id}")
    print(f"   Input: {turn1.user_input}")
    print(f"   Intent: {turn1.intent.value}")
    print(f"   Entities: {len(turn1.entities)} found")

    # Turn 2: Specific request
    turn2 = ConversationTurn(
        turn_id="turn_002",
        user_input="Can you help me organize my photos by date?",
        intent=IntentType.TASK_REQUEST,
        entities=[
            Entity(type="object", value="photos", confidence=0.95),
            Entity(type="action", value="organize", confidence=0.9),
            Entity(type="criteria", value="date", confidence=0.85),
        ],
        context_data={
            "task_complexity": "medium",
            "requires_filesystem": True,
            "estimated_time": "5-10 minutes",
        },
        timestamp=time.time() + 30,
    )
    turns.append(turn2)
    print(f"✅ Created Turn 2: {turn2.turn_id}")
    print(f"   Input: {turn2.user_input}")
    print(f"   Intent: {turn2.intent.value}")
    print(f"   Entities: {len(turn2.entities)} found")

    # Turn 3: Follow-up question
    turn3 = ConversationTurn(
        turn_id="turn_003",
        user_input="What about organizing by location as well?",
        intent=IntentType.CLARIFICATION,
        entities=[
            Entity(type="action", value="organize", confidence=0.9),
            Entity(type="criteria", value="location", confidence=0.9),
            Entity(type="modifier", value="additional", confidence=0.7),
        ],
        context_data={
            "references_previous": True,
            "extends_request": True,
            "complexity_increase": True,
        },
        timestamp=time.time() + 60,
    )
    turns.append(turn3)
    print(f"✅ Created Turn 3: {turn3.turn_id}")
    print(f"   Input: {turn3.user_input}")
    print(f"   Intent: {turn3.intent.value}")
    print(f"   References previous: {turn3.context_data.get('references_previous')}")

    # Demonstrate turn serialization
    print(f"\n📄 Turn Serialization Example:")
    turn_dict = turn1.to_dict()
    print(f"   Serialized keys: {list(turn_dict.keys())}")

    # Reconstruct from dict
    reconstructed = ConversationTurn.from_dict(turn_dict)
    print(f"   Reconstruction successful: {reconstructed.turn_id == turn1.turn_id}")

    return turns


async def demonstrate_conversation_context():
    """
    Demonstrate conversation context management.
    """
    print("\n🧠 Demonstrating Conversation Context")
    print("=" * 42)

    # Create conversation context
    context = ConversationContext(
        session_id="demo_session_001",
        user_id="demo_user",
        max_turns=50,
        memory_limit_mb=10,
    )
    print(f"✅ Created context for session: {context.session_id}")
    print(f"   Max turns: {context.max_turns}")
    print(f"   Memory limit: {context.memory_limit_mb}MB")

    # Create sample conversation turns
    sample_turns = [
        {
            "input": "I want to learn Python programming",
            "intent": IntentType.INFORMATION_REQUEST,
            "entities": [
                Entity(type="subject", value="Python", confidence=0.95),
                Entity(type="action", value="learn", confidence=0.9),
            ],
            "context": {"learning_goal": "programming", "experience_level": "beginner"},
        },
        {
            "input": "What are the best resources for beginners?",
            "intent": IntentType.INFORMATION_REQUEST,
            "entities": [
                Entity(type="target", value="resources", confidence=0.9),
                Entity(type="level", value="beginners", confidence=0.85),
            ],
            "context": {"references_previous": True, "seeking_recommendations": True},
        },
        {
            "input": "Can you recommend some practice projects?",
            "intent": IntentType.TASK_REQUEST,
            "entities": [
                Entity(type="object", value="projects", confidence=0.9),
                Entity(type="purpose", value="practice", confidence=0.85),
            ],
            "context": {"skill_building": True, "hands_on_learning": True},
        },
        {
            "input": "How long does it typically take to become proficient?",
            "intent": IntentType.INFORMATION_REQUEST,
            "entities": [
                Entity(type="duration", value="time", confidence=0.8),
                Entity(type="level", value="proficient", confidence=0.9),
            ],
            "context": {"planning_timeline": True, "goal_setting": True},
        },
    ]

    # Add turns to context
    print(f"\n📝 Adding conversation turns...")
    for i, turn_data in enumerate(sample_turns, 1):
        turn = ConversationTurn(
            turn_id=f"turn_{i:03d}",
            user_input=turn_data["input"],
            intent=turn_data["intent"],
            entities=turn_data["entities"],
            context_data=turn_data["context"],
            timestamp=time.time() + (i * 30),
        )

        await context.add_turn(turn)
        print(f"   ✅ Added turn {i}: {turn.turn_id}")

    # Show context statistics
    print(f"\n📊 Context Statistics:")
    stats = await context.get_context_stats()
    print(f"   - Total turns: {stats['total_turns']}")
    print(f"   - Memory usage: {stats['memory_usage_mb']:.2f}MB")
    print(f"   - Session duration: {stats['session_duration_minutes']:.1f} minutes")
    print(f"   - Intent distribution: {stats['intent_distribution']}")

    # Demonstrate context retrieval
    print(f"\n🔍 Context Retrieval Examples:")

    # Get recent turns
    recent_turns = await context.get_recent_turns(count=2)
    print(f"   Recent turns ({len(recent_turns)}):")
    for turn in recent_turns:
        print(f"     • {turn.turn_id}: {turn.user_input[:50]}...")

    # Get turns by intent
    info_requests = await context.get_turns_by_intent(IntentType.INFORMATION_REQUEST)
    print(f"   Information requests: {len(info_requests)} found")

    # Search context
    search_results = await context.search_context("Python")
    print(f"   Search 'Python': {len(search_results)} matches")

    return context


async def demonstrate_memory_management():
    """
    Demonstrate memory management and optimization.
    """
    print("\n🧹 Demonstrating Memory Management")
    print("=" * 40)

    # Create context with small memory limit for demonstration
    context = ConversationContext(
        session_id="memory_demo_session",
        user_id="demo_user",
        max_turns=10,
        memory_limit_mb=1,  # Very small limit for demo
    )
    print(f"✅ Created context with 1MB memory limit")

    # Add many turns to trigger memory management
    print(f"\n📝 Adding turns to trigger memory management...")

    for i in range(15):  # More than max_turns
        turn = ConversationTurn(
            turn_id=f"memory_turn_{i:03d}",
            user_input=f"This is test message number {i + 1} with some content to use memory",
            intent=IntentType.GENERAL,
            entities=[Entity(type="number", value=str(i + 1), confidence=0.9)],
            context_data={
                "test_data": f"Large context data for turn {i + 1}"
                * 10,  # Increase memory usage
                "iteration": i + 1,
                "memory_test": True,
            },
            timestamp=time.time() + (i * 5),
        )

        await context.add_turn(turn)

        # Check memory usage
        stats = await context.get_context_stats()
        print(
            f"   Turn {i + 1:2d}: {stats['total_turns']} turns, "
            f"{stats['memory_usage_mb']:.2f}MB used"
        )

        # Show when cleanup occurs
        if stats["total_turns"] < i + 1:
            print(
                f"   🧹 Memory cleanup occurred! Turns reduced to {stats['total_turns']}"
            )

    # Show final state
    final_stats = await context.get_context_stats()
    print(f"\n📊 Final Memory State:")
    print(f"   - Turns retained: {final_stats['total_turns']}")
    print(f"   - Memory usage: {final_stats['memory_usage_mb']:.2f}MB")
    print(f"   - Cleanup events: {final_stats.get('cleanup_count', 0)}")

    # Demonstrate memory optimization
    print(f"\n⚡ Memory Optimization:")
    await context.optimize_memory()
    optimized_stats = await context.get_context_stats()
    print(f"   - Memory after optimization: {optimized_stats['memory_usage_mb']:.2f}MB")
    print(f"   - Turns after optimization: {optimized_stats['total_turns']}")


async def demonstrate_context_persistence():
    """
    Demonstrate context persistence and loading.
    """
    print("\n💾 Demonstrating Context Persistence")
    print("=" * 40)

    # Create context with sample data
    original_context = ConversationContext(
        session_id="persistence_demo",
        user_id="demo_user",
        max_turns=20,
        memory_limit_mb=5,
    )

    # Add sample turns
    sample_data = [
        ("Hello, I need help with data analysis", IntentType.TASK_REQUEST),
        ("Can you help me with Python pandas?", IntentType.INFORMATION_REQUEST),
        ("Show me how to read CSV files", IntentType.TASK_REQUEST),
        ("What about data visualization?", IntentType.INFORMATION_REQUEST),
    ]

    for i, (input_text, intent) in enumerate(sample_data, 1):
        turn = ConversationTurn(
            turn_id=f"persist_turn_{i:03d}",
            user_input=input_text,
            intent=intent,
            entities=[Entity(type="demo", value=f"entity_{i}", confidence=0.8)],
            context_data={"demo": True, "turn_number": i},
            timestamp=time.time() + (i * 10),
        )
        await original_context.add_turn(turn)

    print(f"✅ Created context with {len(sample_data)} turns")

    # Save context to temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        context_data = await original_context.to_dict()
        json.dump(context_data, f, indent=2)
        context_file = f.name

    try:
        print(f"💾 Saved context to: {context_file}")

        # Load context from file
        with open(context_file, "r") as f:
            loaded_data = json.load(f)

        loaded_context = ConversationContext.from_dict(loaded_data)
        print(f"📂 Loaded context: {loaded_context.session_id}")

        # Compare original and loaded contexts
        original_stats = await original_context.get_context_stats()
        loaded_stats = await loaded_context.get_context_stats()

        print(f"\n🔍 Context Comparison:")
        print(f"   Original turns: {original_stats['total_turns']}")
        print(f"   Loaded turns: {loaded_stats['total_turns']}")
        print(
            f"   Session ID match: {original_context.session_id == loaded_context.session_id}"
        )
        print(f"   User ID match: {original_context.user_id == loaded_context.user_id}")

        # Verify turn data integrity
        original_turns = await original_context.get_recent_turns(count=10)
        loaded_turns = await loaded_context.get_recent_turns(count=10)

        turns_match = len(original_turns) == len(loaded_turns)
        if turns_match:
            for orig, loaded in zip(original_turns, loaded_turns):
                if (
                    orig.turn_id != loaded.turn_id
                    or orig.user_input != loaded.user_input
                ):
                    turns_match = False
                    break

        print(
            f"   Turn data integrity: {'✅ Preserved' if turns_match else '❌ Corrupted'}"
        )

    finally:
        # Clean up temporary file
        Path(context_file).unlink(missing_ok=True)
        print(f"🧹 Cleaned up temporary context file")


async def demonstrate_multi_session_context():
    """
    Demonstrate multi-session context handling.
    """
    print("\n🔄 Demonstrating Multi-Session Context")
    print("=" * 42)

    # Create multiple session contexts
    sessions = {}
    session_data = [
        {
            "id": "work_session",
            "user": "work_user",
            "topic": "project management",
            "turns": [
                ("I need to track project milestones", IntentType.TASK_REQUEST),
                (
                    "What tools are best for team collaboration?",
                    IntentType.INFORMATION_REQUEST,
                ),
                ("Can you help me create a project timeline?", IntentType.TASK_REQUEST),
            ],
        },
        {
            "id": "learning_session",
            "user": "student_user",
            "topic": "machine learning",
            "turns": [
                (
                    "I want to learn about neural networks",
                    IntentType.INFORMATION_REQUEST,
                ),
                ("Explain backpropagation algorithm", IntentType.INFORMATION_REQUEST),
                ("Show me a simple neural network example", IntentType.TASK_REQUEST),
            ],
        },
        {
            "id": "personal_session",
            "user": "home_user",
            "topic": "home automation",
            "turns": [
                ("Help me set up smart home devices", IntentType.TASK_REQUEST),
                ("What's the best smart thermostat?", IntentType.INFORMATION_REQUEST),
                ("How do I automate my lighting?", IntentType.TASK_REQUEST),
            ],
        },
    ]

    # Create and populate sessions
    for session_info in session_data:
        context = ConversationContext(
            session_id=session_info["id"],
            user_id=session_info["user"],
            max_turns=20,
            memory_limit_mb=5,
        )

        # Add turns to session
        for i, (input_text, intent) in enumerate(session_info["turns"], 1):
            turn = ConversationTurn(
                turn_id=f"{session_info['id']}_turn_{i:03d}",
                user_input=input_text,
                intent=intent,
                entities=[
                    Entity(type="topic", value=session_info["topic"], confidence=0.9)
                ],
                context_data={"session_topic": session_info["topic"]},
                timestamp=time.time() + (i * 15),
            )
            await context.add_turn(turn)

        sessions[session_info["id"]] = context
        print(
            f"✅ Created session: {session_info['id']} ({len(session_info['turns'])} turns)"
        )

    # Analyze sessions
    print(f"\n📊 Multi-Session Analysis:")

    for session_id, context in sessions.items():
        stats = await context.get_context_stats()
        print(f"\n   Session: {session_id}")
        print(f"     - User: {context.user_id}")
        print(f"     - Turns: {stats['total_turns']}")
        print(f"     - Duration: {stats['session_duration_minutes']:.1f} minutes")
        print(f"     - Intents: {stats['intent_distribution']}")

        # Show recent activity
        recent_turns = await context.get_recent_turns(count=1)
        if recent_turns:
            latest = recent_turns[0]
            print(f"     - Latest: {latest.user_input[:40]}...")

    # Cross-session analysis
    print(f"\n🔍 Cross-Session Insights:")

    # Count total turns across all sessions
    total_turns = sum(
        len(await ctx.get_recent_turns(count=100)) for ctx in sessions.values()
    )
    print(f"   - Total turns across all sessions: {total_turns}")

    # Find most active session
    session_activity = {}
    for session_id, context in sessions.items():
        stats = await context.get_context_stats()
        session_activity[session_id] = stats["total_turns"]

    most_active = max(session_activity, key=session_activity.get)
    print(
        f"   - Most active session: {most_active} ({session_activity[most_active]} turns)"
    )

    # Intent distribution across sessions
    all_intents = {}
    for context in sessions.values():
        stats = await context.get_context_stats()
        for intent, count in stats["intent_distribution"].items():
            all_intents[intent] = all_intents.get(intent, 0) + count

    print(f"   - Global intent distribution: {all_intents}")


async def main():
    """
    Main demonstration function.
    """
    parser = argparse.ArgumentParser(description="Context Management Demonstration")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )
    parser.add_argument(
        "--demo",
        choices=["all", "turns", "context", "memory", "persistence", "multi"],
        default="all",
        help="Choose which demonstration to run",
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    print("🧠 Context Management Demonstration Suite")
    print("=" * 50)

    try:
        if args.demo in ["all", "turns"]:
            await demonstrate_conversation_turns()

        if args.demo in ["all", "context"]:
            await demonstrate_conversation_context()

        if args.demo in ["all", "memory"]:
            await demonstrate_memory_management()

        if args.demo in ["all", "persistence"]:
            await demonstrate_context_persistence()

        if args.demo in ["all", "multi"]:
            await demonstrate_multi_session_context()

        print("\n🎉 All context management demonstrations completed successfully!")

    except KeyboardInterrupt:
        print("\n⚠️  Demonstration interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        logging.exception("Context management demonstration error")
        return 1

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
