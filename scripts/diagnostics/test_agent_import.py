#!/usr/bin/env python3
import sys
import os

# Add backend/core to path
sys.path.insert(0, '/Users/miryamstessman/DEV/chatbot/backend/core')

print("Testing LangChain imports...")
print("=" * 50)

try:
    from langchain.agents import create_openai_functions_agent, AgentExecutor
    print(f"✅ Import successful")
    print(f"   create_openai_functions_agent type: {type(create_openai_functions_agent)}")
    print(f"   Is None? {create_openai_functions_agent is None}")
    
    if create_openai_functions_agent is None:
        print("   ❌ PROBLEM: create_openai_functions_agent is None!")
    else:
        print("   ✅ create_openai_functions_agent is a valid function")
        
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("\nTrying alternative import...")
    try:
        from langchain.agents.openai_functions_agent.base import create_openai_functions_agent
        from langchain.agents.agent import AgentExecutor
        print(f"✅ Alternative import successful")
        print(f"   create_openai_functions_agent type: {type(create_openai_functions_agent)}")
        print(f"   Is None? {create_openai_functions_agent is None}")
    except ImportError as e2:
        print(f"❌ Alternative import also failed: {e2}")

print("\nNow testing vectoric_search.py imports...")
try:
    import vectoric_search
    print("✅ vectoric_search imported")
    
    # Check if AdvancedVectorDBQASystem exists
    if hasattr(vectoric_search, 'AdvancedVectorDBQASystem'):
        print("✅ AdvancedVectorDBQASystem found")
    else:
        print("❌ AdvancedVectorDBQASystem NOT found")
        
except Exception as e:
    print(f"❌ Failed to import vectoric_search: {e}")
    import traceback
    traceback.print_exc()
