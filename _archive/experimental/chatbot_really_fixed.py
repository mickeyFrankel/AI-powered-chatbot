#!/usr/bin/env python3
"""
Smart Unified Chatbot - WITH PROPER TERM EXTRACTION
Now extracts search terms from natural language FIRST!
"""

import os
import sys
from pathlib import Path

os.environ['ANONYMIZED_TELEMETRY'] = 'False'
import warnings
warnings.filterwarnings('ignore', message='.*telemetry.*')
warnings.filterwarnings('ignore', message='.*capture.*')

sys.path.insert(0, str(Path(__file__).parent))

from vectoric_search import VectorDBQASystem, AdvancedVectorDBQASystem
import re

# ============================================================================
# CRITICAL FIX: Extract search terms from natural language
# ============================================================================

def extract_search_terms(query: str) -> str:
    """
    Extract actual search term from natural language.
    'phone number of Noam' → 'Noam'
    """
    import re
    
    # Remove common filler phrases
    filler_patterns = [
        r'\b(phone|telephone|tel|mobile|cell)\s+(number|no\.?|#)?\s+(of|for)?\s*',
        r'\b(contact|info|information|details)\s+(of|for|about)?\s*',
        r'\be-?mail\s+(of|for|address)?\s*',
        r'\baddress\s+(of|for)?\s*',
        r'\b(find|get|show|give\s+me|tell\s+me)\s+(the)?\s*',
        r'\b(what|what\'s|whats)\s+(is|are)?\s+(the)?\s*',
        r'\bdo\s+you\s+have\s+(the)?\s*',
    ]
    
    cleaned = query
    for pattern in filler_patterns:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
    
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    cleaned = re.sub(r'\s+(of|for|the|a|an)$', '', cleaned, flags=re.IGNORECASE)
    
    return cleaned if cleaned else query

# ============================================================================

def patch_hierarchical_search():
    """Apply hierarchical search patch"""
    
    def _exact_substring_match(self, query: str):
        """Level 2: substring matching with multi-word support"""
        import unicodedata
        import re
        
        def normalize(s):
            s = unicodedata.normalize("NFKC", s or "")
            return re.sub(r"\s+", " ", s).strip()
        
        q_norm = normalize(query).lower()
        if not q_norm or len(q_norm) < 2:
            return []
        
        query_words = [w for w in q_norm.split() if len(w) > 1]
        
        all_data = self.collection.get(include=["documents", "metadatas"])
        documents = all_data.get("documents", [])
        metadatas = all_data.get("metadatas", [])
        ids = all_data.get("ids", [])
        
        matches = []
        for doc, meta, doc_id in zip(documents, metadatas, ids):
            doc_norm = normalize(doc).lower()
            
            # Exact substring
            if q_norm in doc_norm:
                coverage = len(q_norm) / len(doc_norm) if len(doc_norm) > 0 else 0
                position = doc_norm.find(q_norm)
                position_score = 1 - (position / len(doc_norm)) if len(doc_norm) > 0 else 1
                score = min(0.90 + (coverage * 0.05) + (position_score * 0.05), 0.99)
                
                matches.append({
                    'id': doc_id,
                    'document': doc,
                    'metadata': meta,
                    'match_type': 'exact_substring',
                    'similarity_score': score,
                    'relevance': score
                })
            
            # Multi-word (all words present)
            elif len(query_words) > 1:
                all_words_present = all(word in doc_norm for word in query_words)
                
                if all_words_present:
                    total_word_length = sum(len(w) for w in query_words)
                    coverage = total_word_length / len(doc_norm) if len(doc_norm) > 0 else 0
                    positions = [doc_norm.find(w) for w in query_words]
                    avg_position = sum(positions) / len(positions) if positions else 0
                    position_score = 1 - (avg_position / len(doc_norm)) if len(doc_norm) > 0 else 1
                    score = min(0.85 + (coverage * 0.05) + (position_score * 0.05), 0.94)
                    
                    matches.append({
                        'id': doc_id,
                        'document': doc,
                        'metadata': meta,
                        'match_type': 'multi_word',
                        'similarity_score': score,
                        'relevance': score
                    })
        
        return sorted(matches, key=lambda x: x['similarity_score'], reverse=True)
    
    if not hasattr(VectorDBQASystem, 'search_original'):
        VectorDBQASystem.search_original = VectorDBQASystem.search
    
    VectorDBQASystem._exact_substring_match = _exact_substring_match
    
    def search_patched(self, query: str, n_results: int = 5, **kwargs):
        """PATCHED: Extract terms first, then search"""
        
        # CRITICAL: Extract search terms from natural language
        extracted = extract_search_terms(query)
        
        if extracted != query:
            print(f"   📝 Extracted search term: '{extracted}' from '{query}'")
        
        # Try substring match with extracted term
        substring_matches = self._exact_substring_match(extracted)
        if substring_matches:
            return {
                'query': extracted,
                'original_query': query,
                'search_method': 'exact_substring',
                'results': substring_matches[:n_results]
            }
        
        # Fall back to original search
        return self.search_original(extracted, n_results=n_results, **kwargs)
    
    VectorDBQASystem.search = search_patched
    print("✅ Hierarchical search WITH term extraction applied!")

patch_hierarchical_search()

# ============================================================================

class SmartChatbot:
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory
        self.simple_system = VectorDBQASystem(persist_directory=persist_directory)
        
        self.advanced_system = None
        self.has_openai = bool(os.getenv("OPENAI_API_KEY"))
        
        if self.has_openai:
            try:
                self.advanced_system = AdvancedVectorDBQASystem(persist_directory=persist_directory)
                print("✅ AI mode available")
            except Exception as e:
                print(f"⚠️  AI mode unavailable: {e}")
                self.has_openai = False
        else:
            print("💡 Running in free mode only")
    
    def should_use_ai(self, query: str) -> bool:
        if not self.has_openai:
            return False
        
        query_lower = query.lower().strip()
        
        # Simple patterns (use free search)
        simple_patterns = [
            r'phone.*(?:number|of)',
            r'contact.*(?:info|details|for)',
            r'email.*(?:of|for)',
            r'(?:find|show|get).*(?:phone|contact|email)',
            r'^[א-ת\s]+$',
            r'^[a-z]+\s+[a-z]+$',
            r'^\d{3}',
            r'@',
            r'^(?:load|stats|history|clear|quit|exit)',
        ]
        
        for pattern in simple_patterns:
            if re.search(pattern, query_lower):
                return False
        
        # Complex patterns (use AI)
        ai_patterns = [
            r'\b(?:who|what|when|where|why|how)\b',
            r'\b(?:explain|describe|tell me about|compare)\b',
            r'\b(?:analyze|summarize|recommend|suggest)\b',
        ]
        
        for pattern in ai_patterns:
            if re.search(pattern, query_lower):
                return True
        
        return len(query.split()) > 8 or query.endswith('?')
    
    def process_query(self, query: str):
        use_ai = self.should_use_ai(query)
        
        if use_ai:
            print("🤖 Using AI mode")
            return self.advanced_system.agent_answer(query)
        else:
            print("⚡ Using FREE mode (with term extraction + hierarchical search)")
            
            results = self.simple_system.search(query, n_results=5)
            
            if not results['results']:
                print(f"\n❌ No matches found")
                return None
            
            search_method = results.get('search_method', 'unknown')
            print(f"\n✅ Found {len(results['results'])} match(es)")
            print(f"   Method: {search_method}")
            
            if results['results']:
                top_score = results['results'][0].get('similarity_score', 0)
                if search_method == 'exact_substring':
                    print(f"   🎯 EXACT MATCH ({top_score:.0%})")
            
            print("\n" + "-"*60)
            
            for i, result in enumerate(results['results'], 1):
                doc = result['document']
                score = result.get('similarity_score', 0)
                match_type = result.get('match_type', 'unknown')
                
                quality = "🎯 EXACT" if score >= 0.90 else "✅" if score >= 0.70 else "🔶"
                print(f"\n{i}. {quality} ({score:.0%})")
                
                if '|' in doc:
                    for part in doc.split('|')[:5]:
                        if ':' in part:
                            key, val = part.split(':', 1)
                            key, val = key.strip(), val.strip()
                            if val and key in ['First Name', 'Last Name', 'Phone 1 - Value', 
                                              'Phone 2 - Value', 'E-mail 1 - Value']:
                                print(f"   {key}: {val}")
            
            return results
    
    def interactive_qa(self):
        print("\n" + "="*60)
        print("🧠 SMART CHATBOT (FIXED WITH TERM EXTRACTION)")
        print("   ✅ Extracts names from natural language!")
        print("="*60)
        
        stats = self.simple_system.get_collection_stats()
        print(f"📊 Documents: {stats['document_count']}")
        print(f"💰 AI mode: {'✅ Available' if self.has_openai else '❌ Disabled'}")
        
        print("\n📝 Commands: 'stats' | 'quit'")
        print("💡 Try: 'phone number of Noam' or just 'Noam'\n")
        
        while True:
            try:
                user_input = input("🔍 Query: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                elif user_input.lower() == 'stats':
                    s = self.simple_system.get_collection_stats()
                    print(f"\n📊 Stats: {s['document_count']} documents")
                    continue
                
                self.process_query(user_input)
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

def main():
    chatbot = SmartChatbot()
    chatbot.interactive_qa()

if __name__ == "__main__":
    main()
