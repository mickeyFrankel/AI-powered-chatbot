#!/usr/bin/env python3
"""
Diagnostic script for NoneType callable error
Run this to identify what's None in your vectoric_search.py
"""

import sys
import os
from pathlib import Path

# Add backend/core to path
backend_core = Path(__file__).parent / "backend" / "core"
if backend_core.exists():
    sys.path.insert(0, str(backend_core))

print("🔍 Diagnosing NoneType error...")
print("=" * 50)

# Check 1: Environment
print("\n1. Checking environment variables...")
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if api_key:
    print(f"   ✅ OPENAI_API_KEY found (starts with: {api_key[:10]}...)")
else:
    print("   ❌ OPENAI_API_KEY not found!")
    print("   → Add to .env file: OPENAI_API_KEY=sk-your-key")

# Check 2: LangChain imports
print("\n2. Checking LangChain imports...")
try:
    from langchain_openai import ChatOpenAI
    print("   ✅ ChatOpenAI imported")
except ImportError as e:
    print(f"   ❌ ChatOpenAI import failed: {e}")

try:
    from langchain_core.tools import tool
    print("   ✅ tool decorator imported")
except ImportError:
    try:
        from langchain.tools import tool
        print("   ✅ tool decorator imported (old path)")
    except ImportError as e:
        print(f"   ❌ tool import failed: {e}")

try:
    from langchain.agents import create_openai_functions_agent, AgentExecutor
    print("   ✅ create_openai_functions_agent imported")
    print(f"      Type: {type(create_openai_functions_agent)}")
    
    if create_openai_functions_agent is None:
        print("   ❌ create_openai_functions_agent is None!")
except ImportError:
    try:
        from langchain.agents.openai_functions_agent.base import create_openai_functions_agent
        from langchain.agents.agent import AgentExecutor
        print("   ✅ create_openai_functions_agent imported (alt path)")
        
        if create_openai_functions_agent is None:
            print("   ❌ create_openai_functions_agent is None!")
    except ImportError as e:
        print(f"   ❌ Agent imports failed: {e}")
        print("   → Run: pip install langchain langchain-openai")

# Check 3: Try creating ChatOpenAI
print("\n3. Testing OpenAI client creation...")
try:
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    print(f"   ✅ ChatOpenAI created: {type(llm)}")
except Exception as e:
    print(f"   ❌ Failed to create ChatOpenAI: {e}")

# Check 4: Check if vectoric_search.py exists and can be imported
print("\n4. Checking vectoric_search.py...")
vectoric_path = backend_core / "vectoric_search.py"
if vectoric_path.exists():
    print(f"   ✅ Found at: {vectoric_path}")
    try:
        # Try to import it
        import vectoric_search
        print("   ✅ Module imported successfully")
        
        # Check if AdvancedVectorDBQASystem exists
        if hasattr(vectoric_search, 'AdvancedVectorDBQASystem'):
            print("   ✅ AdvancedVectorDBQASystem class found")
        else:
            print("   ❌ AdvancedVectorDBQASystem class not found!")
            
    except Exception as e:
        print(f"   ❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
else:
    print(f"   ❌ Not found at: {vectoric_path}")
    # Try to find it
    print("\n   Searching for vectoric_search.py...")
    for p in Path(".").rglob("vectoric_search.py"):
        if ".venv" not in str(p):
            print(f"      Found: {p}")

print("\n" + "=" * 50)
print("🎯 Diagnosis complete!")
print("\nMost likely issue:")
print("→ If create_openai_functions_agent is None: Update LangChain")
print("→ If OPENAI_API_KEY missing: Add to .env")
print("→ If imports fail: pip install -r requirements.txt")
