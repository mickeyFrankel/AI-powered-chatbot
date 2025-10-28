#!/usr/bin/env python3
"""
Test smart routing decisions
Shows which queries go to FREE vs AI mode
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from smart_chatbot import SmartChatbot

def test_routing():
    """Test routing logic"""
    
    chatbot = SmartChatbot()
    
    test_cases = [
        # (query, expected_mode, reason)
        ("phone number of אבי", "simple", "Direct lookup"),
        ("אבי אתרוגים", "simple", "Hebrew name"),
        ("Moishi Chen", "simple", "English name"),
        ("050-408-8646", "simple", "Phone number"),
        ("who is David Ben Gurion?", "ai", "Who question"),
        ("what is machine learning?", "ai", "What question"),
        ("explain quantum computing", "ai", "Explain request"),
        ("אבי", "simple", "Single Hebrew word"),
        ("compare A to B", "ai", "Compare request"),
        ("@gmail", "simple", "Email search"),
        ("load contacts.csv", "simple", "Command"),
        ("tell me about Einstein", "ai", "Complex request"),
        ("ברק גורדון", "simple", "Full Hebrew name"),
        ("how does TCP work?", "ai", "How question"),
        ("050", "simple", "Partial phone"),
    ]
    
    print("🧪 Testing Smart Query Routing")
    print("="*60)
    
    correct = 0
    total = len(test_cases)
    
    for query, expected, reason in test_cases:
        uses_ai = chatbot.should_use_ai(query)
        actual = "ai" if uses_ai else "simple"
        
        status = "✅" if actual == expected else "❌"
        mode_display = "🤖 AI" if uses_ai else "⚡ FREE"
        
        print(f"\n{status} Query: '{query}'")
        print(f"   Expected: {expected.upper()}")
        print(f"   Got: {actual.upper()} {mode_display}")
        print(f"   Reason: {reason}")
        
        if actual == expected:
            correct += 1
    
    print("\n" + "="*60)
    print(f"📊 Results: {correct}/{total} correct ({correct/total*100:.0f}%)")
    print("="*60)
    
    if correct == total:
        print("✅ All routing decisions correct!")
    else:
        print(f"⚠️  {total - correct} incorrect routing decisions")
    
    return correct == total

if __name__ == "__main__":
    success = test_routing()
    sys.exit(0 if success else 1)
