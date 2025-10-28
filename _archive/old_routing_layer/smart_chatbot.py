#!/usr/bin/env python3
"""
Smart Unified Chatbot - Routes queries intelligently
Uses FREE simple search when possible, AI only when needed
"""

import os
import sys
from pathlib import Path

# Suppress ChromaDB telemetry warnings
os.environ['ANONYMIZED_TELEMETRY'] = 'False'
import warnings
warnings.filterwarnings('ignore', message='.*telemetry.*')
warnings.filterwarnings('ignore', message='.*capture.*')

sys.path.insert(0, str(Path(__file__).parent))

from vectoric_search import VectorDBQASystem, AdvancedVectorDBQASystem
import re

class SmartChatbot:
    """Intelligently routes between simple and advanced modes"""
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory
        
        # Always initialize simple system (free)
        self.simple_system = VectorDBQASystem(persist_directory=persist_directory)
        
        # Only initialize AI if API key available
        self.advanced_system = None
        self.has_openai = bool(os.getenv("OPENAI_API_KEY"))
        
        if self.has_openai:
            try:
                self.advanced_system = AdvancedVectorDBQASystem(persist_directory=persist_directory)
                print("✅ AI mode available (will use when needed)")
            except Exception as e:
                print(f"⚠️  AI mode unavailable: {e}")
                self.has_openai = False
        else:
            print("💡 Running in free mode (no AI costs)")
    
    def should_use_ai(self, query: str) -> bool:
        """Decide if query needs AI or simple search is enough"""
        
        if not self.has_openai:
            return False
        
        query_lower = query.lower().strip()
        
        # Patterns that DON'T need AI (simple search handles these)
        simple_patterns = [
            # Direct lookups
            r'phone.*(?:number|of)',
            r'contact.*(?:info|details|for)',
            r'email.*(?:of|for)',
            r'(?:find|show|get).*(?:phone|contact|email)',
            
            # Name searches
            r'^[א-ת\s]+$',  # Pure Hebrew (likely a name)
            r'^[a-z]+\s+[a-z]+$',  # Two English words (likely name)
            
            # Partial searches
            r'^\d{3}',  # Starts with digits (phone lookup)
            r'@',  # Email search
            
            # Commands
            r'^(?:load|stats|history|clear|quit|exit)',
        ]
        
        for pattern in simple_patterns:
            if re.search(pattern, query_lower):
                return False  # Use simple search
        
        # Patterns that NEED AI (complex reasoning)
        ai_patterns = [
            # Questions requiring reasoning
            r'\b(?:who|what|when|where|why|how)\b',
            r'\b(?:explain|describe|tell me about|compare)\b',
            r'\b(?:difference between|similar to|related to)\b',
            
            # Analysis requests
            r'\b(?:analyze|summarize|recommend|suggest)\b',
            r'\b(?:should i|can you|would you)\b',
            
            # Multiple steps
            r'\b(?:first.*then|after that|also)\b',
            r'\band\b.*\band\b',  # Multiple "and"s suggest complex query
        ]
        
        for pattern in ai_patterns:
            if re.search(pattern, query_lower):
                return True  # Use AI
        
        # Default: if query is short and simple, use simple search
        # If query is long/complex, use AI
        word_count = len(query.split())
        if word_count <= 3:
            return False  # Simple search
        elif word_count >= 8:
            return True  # Probably needs AI
        
        # Medium length: check if it looks like a search or question
        if query.endswith('?'):
            return True  # Question mark suggests need for AI
        
        return False  # Default to free simple search
    
    def process_query(self, query: str):
        """Route query to appropriate system"""
        
        use_ai = self.should_use_ai(query)
        
        if use_ai:
            print("🤖 Using AI mode (advanced reasoning)")
            return self.advanced_system.agent_answer(query)
        else:
            print("⚡ Using simple mode (free, fast + fuzzy matching)")
            
            # Use fuzzy text search (forgiving of typos/variations)
            results = self.simple_system.search(query, n_results=5, hybrid=True)
            
            if not results['results']:
                print(f"\n❌ No matches found for '{query}'")
                print("\n💡 Try:")
                print("   - Fewer characters (e.g., just first name)")
                print("   - Different spelling")
                print("   - Part of phone number")
                return None
            
            # Show top results (even if similarity is not perfect)
            print(f"\n✅ Top {len(results['results'])} matches:")
            print(f"   Method: {results.get('search_method', 'unknown')}")
            
            # Show similarity threshold info
            if results['results']:
                top_score = results['results'][0].get('similarity_score', 0)
                if top_score < 0.7:
                    print(f"   ⚠️  Note: Best match is {top_score:.0%} similar (not exact)")
            
            print("\n" + "-"*60)
            
            for i, result in enumerate(results['results'], 1):
                doc = result['document']
                score = result.get('similarity_score', 0)
                
                # Color code by quality
                if score >= 0.85:
                    quality = "✅ Excellent"
                elif score >= 0.70:
                    quality = "⚠️  Good"
                else:
                    quality = "🔶 Possible"
                
                print(f"\n{i}. {quality} match ({score:.0%} similar)")
                
                # Parse and show key fields
                if '|' in doc:
                    parts = doc.split('|')
                    displayed_fields = []
                    for part in parts:
                        if ':' in part:
                            key, val = part.split(':', 1)
                            key = key.strip()
                            val = val.strip()
                            
                            # Show important fields
                            if val and key in ['First Name', 'Last Name', 'Phone 1 - Value', 
                                              'Phone 2 - Value', 'E-mail 1 - Value', 'Organization Name']:
                                print(f"   {key}: {val}")
                                displayed_fields.append(key)
                    
                    # If nothing displayed, show raw
                    if not displayed_fields:
                        preview = doc[:150] + "..." if len(doc) > 150 else doc
                        print(f"   {preview}")
                else:
                    preview = doc[:200] + "..." if len(doc) > 200 else doc
                    print(f"   {preview}")
            
            return results
    
    def interactive_qa(self):
        """Main interactive loop"""
        print("\n" + "="*60)
        print("🧠 SMART CHATBOT")
        print("   Automatically chooses: Free search OR AI reasoning")
        print("="*60)
        
        stats = self.simple_system.get_collection_stats()
        print(f"📊 Documents: {stats['document_count']}")
        print(f"💰 AI mode: {'✅ Available' if self.has_openai else '❌ Disabled (free mode only)'}")
        
        print("\n📝 Commands:")
        print("   'load <file>' - Load data")
        print("   'stats' - Show statistics")
        print("   'history' - Show conversation history (AI mode)")
        print("   'clear' - Clear history (AI mode)")
        print("   'quit' - Exit")
        print("\nℹ️  Simple searches use FREE mode, complex queries use AI")
        
        while True:
            try:
                user_input = input("\n🔍 Query: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                elif user_input.lower() == 'stats':
                    s = self.simple_system.get_collection_stats()
                    print(f"\n📊 Database Statistics:")
                    print(f"   Documents: {s['document_count']}")
                    print(f"   Model: {s['embedding_model']}")
                    print(f"   Collection: {s['collection_name']}")
                    continue
                
                elif user_input.lower().startswith('load '):
                    path = user_input[5:].strip()
                    try:
                        docs = self.simple_system.load_file(path)
                        self.simple_system.add_documents(docs)
                    except Exception as e:
                        print(f"❌ Error: {e}")
                    continue
                
                elif user_input.lower() in ('history', 'show history'):
                    if self.has_openai and self.advanced_system:
                        self.advanced_system._show_history()
                    else:
                        print("⚠️  History only available in AI mode")
                    continue
                
                elif user_input.lower() in ('clear', 'clear history'):
                    if self.has_openai and self.advanced_system:
                        self.advanced_system._clear_history()
                    else:
                        print("⚠️  History only available in AI mode")
                    continue
                
                # Process query with smart routing
                self.process_query(user_input)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                import traceback
                traceback.print_exc()

def main():
    chatbot = SmartChatbot()
    chatbot.interactive_qa()

if __name__ == "__main__":
    main()
