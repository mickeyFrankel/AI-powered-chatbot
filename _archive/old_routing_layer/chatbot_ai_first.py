#!/usr/bin/env python3
"""
AI-FIRST Smart Chatbot
Uses OpenAI to understand queries and extract search terms intelligently
"""

import os
import sys
import json
from pathlib import Path

os.environ['ANONYMIZED_TELEMETRY'] = 'False'
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))

from vectoric_search import VectorDBQASystem, AdvancedVectorDBQASystem
from openai import OpenAI

# ============================================================================
# AI QUERY ANALYZER
# ============================================================================

class AIQueryAnalyzer:
    """Uses OpenAI to intelligently understand and extract search terms"""
    
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY required for AI-first approach")
        self.llm = OpenAI(api_key=api_key)
    
    def analyze_query(self, user_query: str) -> dict:
        """
        Use AI to understand query and extract search term.
        Works with any language (Hebrew, English, etc.)
        """
        
        system_prompt = """You are a query analyzer for a contact search system.

Your job: Extract the CORE search term from user queries in ANY language.

Rules:
- "phone of X" → extract "X"
- "phone number of X" → extract "X"
- "contact info for X" → extract "X"
- "email of X" → extract "X"
- "find X" → extract "X"
- Just "X" → extract "X"
- Works in Hebrew, English, or any language

Return ONLY valid JSON (no markdown, no explanation):
{
  "search_term": "extracted term",
  "intent": "find_contact",
  "confidence": 0.95
}"""

        user_prompt = f'Query: "{user_query}"\n\nExtract the search term:'

        try:
            response = self.llm.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0,
                max_tokens=150
            )
            
            content = response.choices[0].message.content.strip()
            
            # Remove markdown code blocks if present
            if content.startswith('```'):
                content = content.split('```')[1]
                if content.startswith('json'):
                    content = content[4:]
                content = content.strip()
            
            result = json.loads(content)
            
            # Validate required fields
            if 'search_term' not in result:
                result['search_term'] = user_query
            if 'intent' not in result:
                result['intent'] = 'find_contact'
            if 'confidence' not in result:
                result['confidence'] = 0.5
            
            return result
            
        except Exception as e:
            print(f"   ⚠️  AI analysis failed: {e}")
            # Fallback: return original query
            return {
                "search_term": user_query,
                "intent": "find_contact",
                "confidence": 0.5,
                "explanation": f"AI failed, using original query"
            }

# ============================================================================
# AI-FIRST SMART CHATBOT
# ============================================================================

class AIFirstChatbot:
    def __init__(self, persist_directory: str = "./chroma_db"):
        print("🤖 Initializing AI-First Chatbot...")
        
        self.analyzer = AIQueryAnalyzer()
        self.search_system = VectorDBQASystem(persist_directory=persist_directory)
        
        print("✅ Ready!")
    
    def search(self, user_query: str, n_results: int = 5):
        """
        AI-first search:
        1. AI understands query and extracts term
        2. Execute search with extracted term
        3. Display results
        """
        
        print(f"\n{'='*70}")
        print(f"🔍 Query: '{user_query}'")
        print(f"{'='*70}")
        
        # Step 1: AI analyzes query
        print("\n🧠 AI Analysis...")
        analysis = self.analyzer.analyze_query(user_query)
        
        search_term = analysis['search_term']
        intent = analysis['intent']
        confidence = analysis.get('confidence', 0.5)
        
        # Show AI understanding
        if search_term != user_query:
            print(f"   📝 Extracted: '{search_term}' from '{user_query}'")
        else:
            print(f"   ✓ Direct search: '{search_term}'")
        
        print(f"   Intent: {intent}")
        print(f"   Confidence: {confidence:.0%}")
        
        # Step 2: Execute search with extracted term
        print(f"\n⚡ Searching for: '{search_term}'")
        
        results = self.search_system.search(search_term, n_results=n_results)
        
        # Step 3: Display results
        self._display_results(results, intent)
        
        return results
    
    def _display_results(self, results: dict, intent: str):
        """Display search results based on intent"""
        
        if not results.get('results'):
            print(f"\n❌ No matches found")
            print("\n💡 Try:")
            print("   - Check spelling")
            print("   - Use fewer words")
            print("   - Try just the first name")
            return
        
        method = results.get('search_method', 'unknown')
        result_list = results['results']
        
        print(f"\n✅ Found {len(result_list)} match(es)")
        print(f"   Method: {method}")
        print(f"\n{'-'*70}\n")
        
        for i, result in enumerate(result_list, 1):
            doc = result['document']
            score = result.get('similarity_score', 0)
            match_type = result.get('match_type', 'unknown')
            
            # Emoji by match quality
            if match_type == 'exact_substring' or score >= 0.90:
                emoji = "🎯"
            elif score >= 0.70:
                emoji = "✅"
            else:
                emoji = "🔶"
            
            print(f"{i}. {emoji} Match: {score:.0%}")
            
            # Parse and show relevant fields based on intent
            if '|' in doc:
                parts = doc.split('|')
                
                # Prioritize fields based on intent
                priority_fields = []
                if intent == 'find_phone':
                    priority_fields = ['First Name', 'Last Name', 'Phone 1 - Value', 'Phone 2 - Value']
                elif intent == 'find_email':
                    priority_fields = ['First Name', 'Last Name', 'E-mail 1 - Value', 'E-mail 2 - Value']
                else:
                    priority_fields = ['First Name', 'Last Name', 'Phone 1 - Value', 'E-mail 1 - Value', 'Organization Name']
                
                for part in parts:
                    if ':' in part:
                        key, val = part.split(':', 1)
                        key, val = key.strip(), val.strip()
                        
                        if val and key in priority_fields:
                            print(f"   {key}: {val}")
            
            print()
    
    def interactive(self):
        """Interactive chatbot loop"""
        
        print("\n" + "="*70)
        print("🧠 AI-FIRST SMART CHATBOT")
        print("   Uses OpenAI to understand your queries intelligently")
        print("="*70)
        
        stats = self.search_system.get_collection_stats()
        print(f"\n📊 Database: {stats['document_count']} contacts")
        
        print("\n💡 Try any of these:")
        print("   - 'phone of Noah'           (AI extracts 'Noah')")
        print("   - 'contact info for Moishi' (AI extracts 'Moishi')")
        print("   - 'אבי אתרוגים'             (Direct Hebrew search)")
        print("   - 'phone of אבי אתרוגים'    (AI extracts Hebrew term)")
        print("   - 'that window guy'         (AI extracts 'window')")
        
        print("\n📝 Commands: 'stats' | 'quit'\n")
        
        while True:
            try:
                user_input = input("🔍 Search: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Goodbye!")
                    break
                
                elif user_input.lower() == 'stats':
                    s = self.search_system.get_collection_stats()
                    print(f"\n📊 Statistics:")
                    print(f"   Documents: {s['document_count']}")
                    print(f"   Model: {s['embedding_model']}")
                    continue
                
                # AI-first search
                self.search(user_input, n_results=5)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                import traceback
                traceback.print_exc()

def main():
    try:
        chatbot = AIFirstChatbot()
        chatbot.interactive()
    except RuntimeError as e:
        print(f"\n❌ {e}")
        print("\nTo use AI-first chatbot, set OPENAI_API_KEY in .env file")
        sys.exit(1)

if __name__ == "__main__":
    main()
