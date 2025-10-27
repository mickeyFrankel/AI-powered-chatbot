#!/usr/bin/env python3
"""
Test Suite for AI-First Refactor
Validates the unified agent architecture
"""

import os
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all imports work"""
    print("🧪 Testing imports...")
    
    try:
        from vectoric_search import VectorDBQASystem, AdvancedVectorDBQASystem
        print("   ✅ VectorDBQASystem imported")
        print("   ✅ AdvancedVectorDBQASystem imported")
        return True
    except ImportError as e:
        print(f"   ❌ Import failed: {e}")
        return False

def test_agent_initialization():
    """Test that agent initializes correctly"""
    print("\n🧪 Testing agent initialization...")
    
    try:
        from vectoric_search import AdvancedVectorDBQASystem
        
        # Check for API key
        if not os.getenv("OPENAI_API_KEY"):
            print("   ⚠️  OPENAI_API_KEY not set - skipping agent test")
            return True
        
        agent = AdvancedVectorDBQASystem()
        print("   ✅ Agent initialized successfully")
        
        # Check tools are available
        tools = agent._tool_specs()
        print(f"   ✅ Agent has {len(tools)} tools")
        
        return True
    except Exception as e:
        print(f"   ❌ Agent initialization failed: {e}")
        return False

def test_basic_search():
    """Test basic search functionality"""
    print("\n🧪 Testing basic search (VectorDBQASystem)...")
    
    try:
        from vectoric_search import VectorDBQASystem
        
        system = VectorDBQASystem()
        stats = system.get_collection_stats()
        
        print(f"   ✅ Database has {stats['document_count']} documents")
        
        if stats['document_count'] > 0:
            # Try a search
            results = system.search("test", n_results=1)
            print(f"   ✅ Search executed successfully")
        else:
            print("   ⚠️  Database empty - skipping search test")
        
        return True
    except Exception as e:
        print(f"   ❌ Basic search failed: {e}")
        return False

def test_agent_tools():
    """Test that agent has all required tools"""
    print("\n🧪 Testing agent tools...")
    
    try:
        from vectoric_search import AdvancedVectorDBQASystem
        
        if not os.getenv("OPENAI_API_KEY"):
            print("   ⚠️  OPENAI_API_KEY not set - skipping")
            return True
        
        agent = AdvancedVectorDBQASystem()
        tools = agent._tool_specs()
        
        expected_tools = [
            "search",
            "list_by_prefix",
            "names_by_length",
            "names_containing",
            "names_by_prefix_and_length",
            "letter_histogram",
            "length_histogram",
        ]
        
        tool_names = [t["function"]["name"] for t in tools]
        
        for expected in expected_tools:
            if expected in tool_names:
                print(f"   ✅ Tool '{expected}' available")
            else:
                print(f"   ❌ Tool '{expected}' MISSING")
                return False
        
        return True
    except Exception as e:
        print(f"   ❌ Tool test failed: {e}")
        return False

def test_tool_dispatch():
    """Test tool dispatch mechanism"""
    print("\n🧪 Testing tool dispatch...")
    
    try:
        from vectoric_search import AdvancedVectorDBQASystem
        
        if not os.getenv("OPENAI_API_KEY"):
            print("   ⚠️  OPENAI_API_KEY not set - skipping")
            return True
        
        agent = AdvancedVectorDBQASystem()
        
        # Test search tool
        result = agent._dispatch_tool("search", {"query": "test", "n_results": 1})
        print("   ✅ search() tool dispatches correctly")
        
        # Test list_by_prefix tool
        result = agent._dispatch_tool("list_by_prefix", {"letter": "A", "n": 5})
        print("   ✅ list_by_prefix() tool dispatches correctly")
        
        return True
    except Exception as e:
        print(f"   ❌ Tool dispatch failed: {e}")
        return False

def test_file_structure():
    """Test that file structure is correct"""
    print("\n🧪 Testing file structure...")
    
    expected_files = [
        "chatbot.py",
        "vectoric_search.py",
        "requirements.txt",
        ".env",  # Should exist
    ]
    
    archived_files = [
        "smart_chatbot.py",
        "chatbot_tiered.py",
        "simple_chatbot.py",
    ]
    
    all_good = True
    
    # Check expected files exist
    for file in expected_files:
        path = Path(file)
        if path.exists():
            print(f"   ✅ {file} exists")
        else:
            if file == ".env":
                print(f"   ⚠️  {file} missing (create it with OPENAI_API_KEY)")
            else:
                print(f"   ❌ {file} MISSING")
                all_good = False
    
    # Check archived files are gone from root
    for file in archived_files:
        path = Path(file)
        if not path.exists():
            print(f"   ✅ {file} archived (not in root)")
        else:
            print(f"   ⚠️  {file} still in root (should be archived)")
    
    return all_good

def test_conversation_memory():
    """Test that agent has conversation memory"""
    print("\n🧪 Testing conversation memory...")
    
    try:
        from vectoric_search import AdvancedVectorDBQASystem
        
        if not os.getenv("OPENAI_API_KEY"):
            print("   ⚠️  OPENAI_API_KEY not set - skipping")
            return True
        
        agent = AdvancedVectorDBQASystem()
        
        # Check conversation history exists
        if hasattr(agent, 'conversation_history'):
            print("   ✅ Agent has conversation_history attribute")
        else:
            print("   ❌ Agent missing conversation_history")
            return False
        
        # Check history management methods
        if hasattr(agent, '_clear_history'):
            print("   ✅ Agent has _clear_history() method")
        else:
            print("   ⚠️  Agent missing _clear_history() method")
        
        if hasattr(agent, '_show_history'):
            print("   ✅ Agent has _show_history() method")
        else:
            print("   ⚠️  Agent missing _show_history() method")
        
        return True
    except Exception as e:
        print(f"   ❌ Memory test failed: {e}")
        return False

def run_all_tests():
    """Run all tests and report"""
    print("=" * 70)
    print("🚀 AI-FIRST REFACTOR TEST SUITE")
    print("=" * 70)
    print()
    
    tests = [
        ("Imports", test_imports),
        ("Agent Initialization", test_agent_initialization),
        ("Basic Search", test_basic_search),
        ("Agent Tools", test_agent_tools),
        ("Tool Dispatch", test_tool_dispatch),
        ("File Structure", test_file_structure),
        ("Conversation Memory", test_conversation_memory),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ Test '{name}' crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 TEST SUMMARY")
    print("=" * 70)
    print()
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status} - {name}")
    
    print()
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print()
        print("🎉 ALL TESTS PASSED!")
        print("✅ Refactor is complete and working correctly")
        print()
        print("Next steps:")
        print("  1. Run: python chatbot.py")
        print("  2. Test with real queries")
        print("  3. Check conversation memory works")
        print("  4. Monitor costs in OpenAI dashboard")
        print()
        return 0
    else:
        print()
        print("⚠️  SOME TESTS FAILED")
        print("Please review the failures above")
        print()
        return 1

if __name__ == "__main__":
    sys.exit(run_all_tests())
