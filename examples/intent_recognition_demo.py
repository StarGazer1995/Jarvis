#!/usr/bin/env python3
"""
Intent Recognition Demonstration.

This script demonstrates the intent recognition and entity extraction
capabilities of the ARK system, including:
- Intent classification and confidence scoring
- Entity extraction and validation
- Pattern matching and rule-based recognition
- Machine learning-based intent detection
- Multi-language intent processing

Run from project root: python examples/intent_recognition_demo.py --verbose
"""

import asyncio
import logging
import argparse
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.intent_engine import (
    IntentType, Entity, IntentResult, ARKIntentEngine,
    IntentPattern, EntityExtractor
)


async def demonstrate_basic_intent_recognition():
    """
    Demonstrate basic intent recognition capabilities.
    """
    print("🎯 Demonstrating Basic Intent Recognition")
    print("=" * 45)
    
    # Create intent engine
    intent_engine = ARKIntentEngine()
    print("✅ Intent Engine initialized")
    
    # Sample user inputs with expected intents
    test_cases = [
        {
            "input": "Hello, how are you today?",
            "expected_intent": IntentType.GREETING,
            "description": "Simple greeting"
        },
        {
            "input": "Can you help me write a Python script?",
            "expected_intent": IntentType.TASK_REQUEST,
            "description": "Task assistance request"
        },
        {
            "input": "What is the weather like in New York?",
            "expected_intent": IntentType.INFORMATION_REQUEST,
            "description": "Information query"
        },
        {
            "input": "I think there's an error in the calculation",
            "expected_intent": IntentType.CLARIFICATION,
            "description": "Clarification or correction"
        },
        {
            "input": "Thank you for your help!",
            "expected_intent": IntentType.GENERAL,
            "description": "Gratitude expression"
        },
        {
            "input": "Could you explain how neural networks work?",
            "expected_intent": IntentType.INFORMATION_REQUEST,
            "description": "Educational query"
        },
        {
            "input": "Please create a backup of my files",
            "expected_intent": IntentType.TASK_REQUEST,
            "description": "File management task"
        },
        {
            "input": "What did I ask you about earlier?",
            "expected_intent": IntentType.CLARIFICATION,
            "description": "Context reference"
        }
    ]
    
    print(f"\n🧪 Testing {len(test_cases)} intent recognition cases:")
    
    correct_predictions = 0
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📝 Test Case {i}: {test_case['description']}")
        print(f"   Input: \"{test_case['input']}\"")
        print(f"   Expected: {test_case['expected_intent'].value}")
        
        # Recognize intent
        result = await intent_engine.recognize_intent(test_case['input'])
        
        print(f"   Predicted: {result.intent.value}")
        print(f"   Confidence: {result.confidence:.3f}")
        
        # Check if prediction is correct
        is_correct = result.intent == test_case['expected_intent']
        status = "✅" if is_correct else "❌"
        print(f"   Result: {status} {'Correct' if is_correct else 'Incorrect'}")
        
        if is_correct:
            correct_predictions += 1
        
        # Show extracted entities
        if result.entities:
            print(f"   Entities: {len(result.entities)} found")
            for entity in result.entities[:3]:  # Show first 3 entities
                print(f"     • {entity.type}: \"{entity.value}\" (conf: {entity.confidence:.3f})")
    
    # Show overall accuracy
    accuracy = correct_predictions / len(test_cases)
    print(f"\n📊 Intent Recognition Accuracy: {accuracy:.1%} ({correct_predictions}/{len(test_cases)})")


async def demonstrate_entity_extraction():
    """
    Demonstrate entity extraction capabilities.
    """
    print("\n🏷️  Demonstrating Entity Extraction")
    print("=" * 40)
    
    # Create intent engine
    intent_engine = ARKIntentEngine()
    
    # Sample inputs with rich entity content
    entity_test_cases = [
        {
            "input": "Schedule a meeting with John Smith at 3 PM tomorrow in Conference Room A",
            "expected_entities": ["person", "time", "date", "location", "action"],
            "description": "Meeting scheduling"
        },
        {
            "input": "Send an email to alice@company.com about the quarterly report due Friday",
            "expected_entities": ["email", "person", "document", "date", "action"],
            "description": "Email task"
        },
        {
            "input": "Download the Python 3.11 installer from python.org",
            "expected_entities": ["software", "version", "website", "action"],
            "description": "Software download"
        },
        {
            "input": "Calculate the distance between New York and Los Angeles in kilometers",
            "expected_entities": ["location", "location", "unit", "action"],
            "description": "Calculation request"
        },
        {
            "input": "Book a flight from London to Tokyo departing December 15th, returning December 22nd",
            "expected_entities": ["location", "location", "date", "date", "action"],
            "description": "Travel booking"
        },
        {
            "input": "Create a backup of the database server at 192.168.1.100 every Sunday at 2 AM",
            "expected_entities": ["ip_address", "frequency", "time", "action", "object"],
            "description": "Backup scheduling"
        }
    ]
    
    print(f"\n🔍 Testing entity extraction on {len(entity_test_cases)} cases:")
    
    for i, test_case in enumerate(entity_test_cases, 1):
        print(f"\n📝 Test Case {i}: {test_case['description']}")
        print(f"   Input: \"{test_case['input']}\"")
        print(f"   Expected entity types: {', '.join(test_case['expected_entities'])}")
        
        # Extract entities
        result = await intent_engine.recognize_intent(test_case['input'])
        
        print(f"   Intent: {result.intent.value} (conf: {result.confidence:.3f})")
        print(f"   Entities found: {len(result.entities)}")
        
        # Show extracted entities
        if result.entities:
            for entity in result.entities:
                print(f"     • {entity.type}: \"{entity.value}\" (conf: {entity.confidence:.3f})")
        else:
            print("     (No entities extracted)")
        
        # Analyze entity coverage
        extracted_types = {entity.type for entity in result.entities}
        expected_types = set(test_case['expected_entities'])
        
        coverage = len(extracted_types & expected_types) / len(expected_types) if expected_types else 0
        print(f"   Entity coverage: {coverage:.1%}")


async def demonstrate_pattern_matching():
    """
    Demonstrate pattern-based intent recognition.
    """
    print("\n🔍 Demonstrating Pattern Matching")
    print("=" * 38)
    
    # Create intent engine
    intent_engine = ARKIntentEngine()
    
    # Define custom patterns for demonstration
    custom_patterns = [
        IntentPattern(
            name="file_operation",
            intent=IntentType.TASK_REQUEST,
            patterns=[
                r"(?i)(create|make|generate)\s+(?:a\s+)?file",
                r"(?i)(delete|remove)\s+(?:the\s+)?file",
                r"(?i)(copy|move)\s+(?:the\s+)?file",
                r"(?i)(read|open)\s+(?:the\s+)?file"
            ],
            confidence_boost=0.2
        ),
        IntentPattern(
            name="time_query",
            intent=IntentType.INFORMATION_REQUEST,
            patterns=[
                r"(?i)what\s+time\s+is\s+it",
                r"(?i)current\s+time",
                r"(?i)what's\s+the\s+time",
                r"(?i)time\s+in\s+\w+"
            ],
            confidence_boost=0.3
        ),
        IntentPattern(
            name="calculation",
            intent=IntentType.TASK_REQUEST,
            patterns=[
                r"(?i)calculate\s+",
                r"(?i)compute\s+",
                r"(?i)what\s+is\s+\d+.*[\+\-\*\/].*\d+",
                r"(?i)solve\s+.*equation"
            ],
            confidence_boost=0.25
        ),
        IntentPattern(
            name="greeting_extended",
            intent=IntentType.GREETING,
            patterns=[
                r"(?i)good\s+(morning|afternoon|evening)",
                r"(?i)how\s+are\s+you",
                r"(?i)nice\s+to\s+meet\s+you",
                r"(?i)greetings"
            ],
            confidence_boost=0.15
        )
    ]
    
    print(f"✅ Defined {len(custom_patterns)} custom patterns")
    
    # Test pattern matching
    pattern_test_cases = [
        {
            "input": "Create a new file called data.txt",
            "expected_pattern": "file_operation",
            "description": "File creation"
        },
        {
            "input": "What time is it in London?",
            "expected_pattern": "time_query",
            "description": "Time inquiry"
        },
        {
            "input": "Calculate the square root of 144",
            "expected_pattern": "calculation",
            "description": "Mathematical calculation"
        },
        {
            "input": "Good morning! How are you today?",
            "expected_pattern": "greeting_extended",
            "description": "Extended greeting"
        },
        {
            "input": "Delete the temporary files in the cache folder",
            "expected_pattern": "file_operation",
            "description": "File deletion"
        },
        {
            "input": "What is 25 * 4 + 10?",
            "expected_pattern": "calculation",
            "description": "Arithmetic expression"
        }
    ]
    
    print(f"\n🎯 Testing pattern matching on {len(pattern_test_cases)} cases:")
    
    for i, test_case in enumerate(pattern_test_cases, 1):
        print(f"\n📝 Test Case {i}: {test_case['description']}")
        print(f"   Input: \"{test_case['input']}\"")
        print(f"   Expected pattern: {test_case['expected_pattern']}")
        
        # Test against each pattern
        matched_patterns = []
        for pattern in custom_patterns:
            if await intent_engine._match_pattern(test_case['input'], pattern):
                matched_patterns.append(pattern.name)
        
        print(f"   Matched patterns: {', '.join(matched_patterns) if matched_patterns else 'None'}")
        
        # Check if expected pattern matched
        expected_matched = test_case['expected_pattern'] in matched_patterns
        status = "✅" if expected_matched else "❌"
        print(f"   Expected match: {status} {'Found' if expected_matched else 'Not found'}")
        
        # Get full intent recognition result
        result = await intent_engine.recognize_intent(test_case['input'])
        print(f"   Final intent: {result.intent.value} (conf: {result.confidence:.3f})")


async def demonstrate_confidence_scoring():
    """
    Demonstrate confidence scoring mechanisms.
    """
    print("\n📊 Demonstrating Confidence Scoring")
    print("=" * 40)
    
    # Create intent engine
    intent_engine = ARKIntentEngine()
    
    # Test cases with varying clarity levels
    confidence_test_cases = [
        {
            "input": "Hello",
            "clarity": "high",
            "description": "Clear, unambiguous greeting"
        },
        {
            "input": "Can you help me with something?",
            "clarity": "medium",
            "description": "Vague assistance request"
        },
        {
            "input": "Write a Python function to sort a list",
            "clarity": "high",
            "description": "Specific programming task"
        },
        {
            "input": "Um, I think maybe you could possibly help?",
            "clarity": "low",
            "description": "Uncertain, hesitant request"
        },
        {
            "input": "What's the weather forecast for tomorrow in San Francisco?",
            "clarity": "high",
            "description": "Specific information request"
        },
        {
            "input": "That thing we talked about before",
            "clarity": "low",
            "description": "Vague reference to previous context"
        },
        {
            "input": "Please schedule a meeting for next Tuesday at 2 PM",
            "clarity": "high",
            "description": "Clear scheduling request"
        },
        {
            "input": "I don't know, maybe something about files?",
            "clarity": "low",
            "description": "Uncertain and vague"
        }
    ]
    
    print(f"\n🎯 Testing confidence scoring on {len(confidence_test_cases)} cases:")
    
    confidence_by_clarity = {"high": [], "medium": [], "low": []}
    
    for i, test_case in enumerate(confidence_test_cases, 1):
        print(f"\n📝 Test Case {i}: {test_case['description']}")
        print(f"   Input: \"{test_case['input']}\"")
        print(f"   Expected clarity: {test_case['clarity']}")
        
        # Get intent recognition result
        result = await intent_engine.recognize_intent(test_case['input'])
        
        print(f"   Intent: {result.intent.value}")
        print(f"   Confidence: {result.confidence:.3f}")
        
        # Categorize confidence score
        if result.confidence >= 0.8:
            actual_clarity = "high"
        elif result.confidence >= 0.5:
            actual_clarity = "medium"
        else:
            actual_clarity = "low"
        
        print(f"   Actual clarity: {actual_clarity}")
        
        # Check if clarity assessment matches expectation
        clarity_match = actual_clarity == test_case['clarity']
        status = "✅" if clarity_match else "⚠️"
        print(f"   Clarity assessment: {status} {'Matches' if clarity_match else 'Differs from'} expectation")
        
        # Store for analysis
        confidence_by_clarity[test_case['clarity']].append(result.confidence)
    
    # Analyze confidence distribution
    print(f"\n📈 Confidence Score Analysis:")
    for clarity_level in ["high", "medium", "low"]:
        scores = confidence_by_clarity[clarity_level]
        if scores:
            avg_confidence = sum(scores) / len(scores)
            min_confidence = min(scores)
            max_confidence = max(scores)
            print(f"   {clarity_level.capitalize()} clarity:")
            print(f"     - Average: {avg_confidence:.3f}")
            print(f"     - Range: {min_confidence:.3f} - {max_confidence:.3f}")
            print(f"     - Count: {len(scores)} cases")


async def demonstrate_multilingual_intent():
    """
    Demonstrate multi-language intent processing.
    """
    print("\n🌍 Demonstrating Multi-Language Intent Processing")
    print("=" * 52)
    
    # Create intent engine
    intent_engine = ARKIntentEngine()
    
    # Multi-language test cases
    multilingual_cases = [
        {
            "input": "Hello, how can I help you?",
            "language": "English",
            "expected_intent": IntentType.GREETING
        },
        {
            "input": "Hola, ¿cómo puedo ayudarte?",
            "language": "Spanish",
            "expected_intent": IntentType.GREETING
        },
        {
            "input": "Bonjour, comment puis-je vous aider?",
            "language": "French",
            "expected_intent": IntentType.GREETING
        },
        {
            "input": "Can you create a file for me?",
            "language": "English",
            "expected_intent": IntentType.TASK_REQUEST
        },
        {
            "input": "¿Puedes crear un archivo para mí?",
            "language": "Spanish",
            "expected_intent": IntentType.TASK_REQUEST
        },
        {
            "input": "Pouvez-vous créer un fichier pour moi?",
            "language": "French",
            "expected_intent": IntentType.TASK_REQUEST
        },
        {
            "input": "What is the weather like today?",
            "language": "English",
            "expected_intent": IntentType.INFORMATION_REQUEST
        },
        {
            "input": "¿Cómo está el clima hoy?",
            "language": "Spanish",
            "expected_intent": IntentType.INFORMATION_REQUEST
        }
    ]
    
    print(f"🧪 Testing intent recognition across {len(set(case['language'] for case in multilingual_cases))} languages:")
    
    results_by_language = {}
    
    for i, test_case in enumerate(multilingual_cases, 1):
        print(f"\n📝 Test Case {i}: {test_case['language']}")
        print(f"   Input: \"{test_case['input']}\"")
        print(f"   Expected: {test_case['expected_intent'].value}")
        
        # Recognize intent
        result = await intent_engine.recognize_intent(test_case['input'])
        
        print(f"   Predicted: {result.intent.value}")
        print(f"   Confidence: {result.confidence:.3f}")
        
        # Check accuracy
        is_correct = result.intent == test_case['expected_intent']
        status = "✅" if is_correct else "❌"
        print(f"   Result: {status} {'Correct' if is_correct else 'Incorrect'}")
        
        # Store results by language
        lang = test_case['language']
        if lang not in results_by_language:
            results_by_language[lang] = {"correct": 0, "total": 0, "confidences": []}
        
        results_by_language[lang]["total"] += 1
        if is_correct:
            results_by_language[lang]["correct"] += 1
        results_by_language[lang]["confidences"].append(result.confidence)
    
    # Show language-specific performance
    print(f"\n📊 Performance by Language:")
    for language, stats in results_by_language.items():
        accuracy = stats["correct"] / stats["total"]
        avg_confidence = sum(stats["confidences"]) / len(stats["confidences"])
        print(f"   {language}:")
        print(f"     - Accuracy: {accuracy:.1%} ({stats['correct']}/{stats['total']})")
        print(f"     - Avg Confidence: {avg_confidence:.3f}")


async def demonstrate_context_aware_intent():
    """
    Demonstrate context-aware intent recognition.
    """
    print("\n🧠 Demonstrating Context-Aware Intent Recognition")
    print("=" * 52)
    
    # Create intent engine
    intent_engine = ARKIntentEngine()
    
    # Simulate conversation context
    conversation_context = {
        "previous_intent": IntentType.TASK_REQUEST,
        "previous_entities": [
            Entity(type="file", value="report.pdf", confidence=0.9),
            Entity(type="action", value="create", confidence=0.8)
        ],
        "current_topic": "document_management",
        "user_preferences": {
            "preferred_format": "PDF",
            "default_location": "/Documents"
        }
    }
    
    # Context-dependent test cases
    context_cases = [
        {
            "input": "Yes, please do that",
            "context": conversation_context,
            "description": "Affirmative response to previous request",
            "expected_behavior": "Should reference previous task request"
        },
        {
            "input": "Make it in Word format instead",
            "context": conversation_context,
            "description": "Modification to previous request",
            "expected_behavior": "Should understand format change"
        },
        {
            "input": "Where should I save it?",
            "context": conversation_context,
            "description": "Follow-up question about location",
            "expected_behavior": "Should relate to file creation task"
        },
        {
            "input": "Can you also create a backup?",
            "context": conversation_context,
            "description": "Additional related request",
            "expected_behavior": "Should extend current task"
        },
        {
            "input": "Never mind, cancel that",
            "context": conversation_context,
            "description": "Cancellation request",
            "expected_behavior": "Should understand task cancellation"
        }
    ]
    
    print(f"🔄 Testing context-aware recognition on {len(context_cases)} cases:")
    print(f"📋 Current context: {conversation_context['current_topic']}")
    print(f"   Previous intent: {conversation_context['previous_intent'].value}")
    
    for i, test_case in enumerate(context_cases, 1):
        print(f"\n📝 Test Case {i}: {test_case['description']}")
        print(f"   Input: \"{test_case['input']}\"")
        print(f"   Expected: {test_case['expected_behavior']}")
        
        # Recognize intent with context
        result = await intent_engine.recognize_intent_with_context(
            test_case['input'],
            test_case['context']
        )
        
        print(f"   Intent: {result.intent.value}")
        print(f"   Confidence: {result.confidence:.3f}")
        
        # Show context influence
        if hasattr(result, 'context_influence'):
            print(f"   Context influence: {result.context_influence:.3f}")
        
        # Show extracted entities
        if result.entities:
            print(f"   Entities: {len(result.entities)} found")
            for entity in result.entities[:2]:  # Show first 2
                print(f"     • {entity.type}: \"{entity.value}\" (conf: {entity.confidence:.3f})")
        
        # Analyze context references
        context_refs = [e for e in result.entities if e.type in ["reference", "pronoun", "context"]]
        if context_refs:
            print(f"   Context references: {len(context_refs)} detected")


async def main():
    """
    Main demonstration function.
    """
    parser = argparse.ArgumentParser(description="Intent Recognition Demonstration")
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--demo",
        choices=["all", "basic", "entities", "patterns", "confidence", "multilingual", "context"],
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
    
    print("🎯 Intent Recognition Demonstration Suite")
    print("=" * 50)
    
    try:
        if args.demo in ["all", "basic"]:
            await demonstrate_basic_intent_recognition()
        
        if args.demo in ["all", "entities"]:
            await demonstrate_entity_extraction()
        
        if args.demo in ["all", "patterns"]:
            await demonstrate_pattern_matching()
        
        if args.demo in ["all", "confidence"]:
            await demonstrate_confidence_scoring()
        
        if args.demo in ["all", "multilingual"]:
            await demonstrate_multilingual_intent()
        
        if args.demo in ["all", "context"]:
            await demonstrate_context_aware_intent()
        
        print("\n🎉 All intent recognition demonstrations completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠️  Demonstration interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        logging.exception("Intent recognition demonstration error")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)