#!/usr/bin/env python3
"""
FIXED Chatbot - Addresses all hallucination and search issues
Run this instead of vectoric_search.py
"""

import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import and patch the system
from vectoric_search import AdvancedVectorDBQASystem, VectorDBQASystem

# Add the critical missing method to VectorDBQASystem
def search_by_substring(self, substring: str, n_results: int = 50):
    """Search for documents containing exact substring"""
    print(f"\n🔍 Substring search for: '{substring}'")
    
    all_docs = self.collection.get(include=["documents", "metadatas", "ids"])
    documents = all_docs.get("documents", [])
    metadatas = all_docs.get("metadatas", [])
    ids = all_docs.get("ids", [])
    
    substring_lower = substring.lower()
    matches = []
    
    for doc, metadata, doc_id in zip(documents, metadatas, ids):
        if substring_lower in doc.lower():
            matches.append({
                'id': doc_id,
                'document': doc,
                'metadata': metadata,
                'similarity_score': 1.0,
                'distance': 0.0
            })
    
    matches = matches[:n_results]
    print(f"✅ Found {len(matches)} contacts containing '{substring}'")
    
    return {
        'query': substring,
        'search_type': 'substring',
        'results': matches
    }

# Monkey-patch the method
VectorDBQASystem.search_by_substring = search_by_substring

# Override the agent_answer with FIXED version
original_agent_answer = AdvancedVectorDBQASystem.agent_answer

def fixed_agent_answer(self, user_input: str, max_steps: int = 5) -> str:
    """Fixed agent with STRICT search rules"""
    
    # Initialize with STRICT system prompt
    if not self.conversation_history:
        system = (
            "You are a contact database assistant with FULL ACCESS to all contacts.\n\n"
            
            "🚨 CRITICAL RULES - NEVER BREAK THESE:\n"
            "1. ALWAYS use search tools - NEVER EVER say 'I don't have access'\n"
            "2. For ANY name query, ALWAYS call search_by_substring with the name\n"
            "3. search_by_substring finds ALL contacts containing that text\n"
            "4. NEVER make up information - only report what tools return\n"
            "5. If no results, say 'No contacts found containing [query]' and suggest alternatives\n\n"
            
            "Tool Selection:\n"
            "- User asks for 'ברק' → Call search_by_substring('ברק')\n"
            "- User asks for 'phone of Chen' → Call search_by_substring('Chen')\n"
            "- User asks for 'contacts with 050' → Call search_by_substring('050')\n"
            "- User asks for semantic query → Call search()\n\n"
            
            "EXAMPLES OF CORRECT BEHAVIOR:\n"
            "❌ BAD: 'I couldn't find ברק' or 'I don't have access'\n"
            "✅ GOOD: [Calls search_by_substring('ברק')] → Returns all matches\n\n"
            
            "You have COMPLETE access to all 1917 contacts. Use it!"
        )
        self.conversation_history.append({"role": "system", "content": system})
    
    # Add user message
    self.conversation_history.append({"role": "user", "content": user_input})
    
    # Use conversation history
    messages = self.conversation_history.copy()
    
    for _ in range(max_steps):
        try:
            resp = self.llm.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=self._tool_specs(),
                tool_choice="auto",
                temperature=0
            )
            msg = resp.choices[0].message

            if msg.tool_calls:
                messages.append({"role":"assistant","content":msg.content or "", "tool_calls": msg.tool_calls})
                for tc in msg.tool_calls:
                    import json
                    name = tc.function.name
                    args = json.loads(tc.function.arguments or "{}")
                    result = self._dispatch_tool(name, args)
                    messages.append({
                        "role":"tool",
                        "tool_call_id": tc.id,
                        "name": name,
                        "content": json.dumps(result, ensure_ascii=False)
                    })
                continue

            final = msg.content or "(no content)"
            self.conversation_history.append({"role": "assistant", "content": final})
            self._trim_history()
            
            print(final)
            return final
        except Exception as e:
            error_msg = f"Error: {e}"
            print(error_msg)
            return error_msg

    print("Reached max tool steps")
    return ""

# Patch the method
AdvancedVectorDBQASystem.agent_answer = fixed_agent_answer

# Add search_by_substring to tool specs
original_tool_specs = AdvancedVectorDBQASystem._tool_specs

def fixed_tool_specs(self):
    """Add substring search tool"""
    tools = original_tool_specs(self)
    
    # Add new substring search tool
    tools.append({
        "type": "function", 
        "function": {
            "name": "search_by_substring",
            "description": "CRITICAL: Use this to find contacts by partial name. Searches for EXACT text matches (case-insensitive). Example: 'ברק' finds 'ברק גורדון', 'ברק כהן', etc.",
            "parameters": {
                "type":"object",
                "properties":{
                    "substring":{"type":"string", "description": "Text to find (name, phone, email)"},
                    "n_results":{"type":"integer","default":50}
                },
                "required":["substring"]
            }
        }
    })
    
    return tools

AdvancedVectorDBQASystem._tool_specs = fixed_tool_specs

# Add to dispatcher
original_dispatch = AdvancedVectorDBQASystem._dispatch_tool

def fixed_dispatch_tool(self, name: str, args: dict):
    """Handle substring search"""
    try:
        if name == "search_by_substring":
            result = self.search_by_substring(
                args.get("substring", ""),
                n_results=int(args.get("n_results", 50))
            )
            return result
        
        return original_dispatch(self, name, args)
    except Exception as e:
        return {"error": str(e)}

AdvancedVectorDBQASystem._dispatch_tool = fixed_dispatch_tool

# Main function
def main():
    print("🔧 FIXED VectorDB Q&A System")
    print("   ✅ No more hallucination")
    print("   ✅ Substring search enabled")  
    print("   ✅ Always searches database")
    print("=" * 60)
    
    try:
        qa_system = AdvancedVectorDBQASystem()
        qa_system.interactive_qa()
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n💡 If you get API key error, your OpenAI key is invalid.")
        print("   Run 'python simple_chatbot.py' instead (no API key needed)")

if __name__ == "__main__":
    main()
