#!/usr/bin/env python3
"""
Smart Unified Chatbot - Routes queries intelligently
Uses FREE simple search when possible, AI only when needed
WITH HIERARCHICAL SEARCH PATCH APPLIED
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

# ============================================================================
# HIERARCHICAL SEARCH PATCH - Applied at runtime
# ============================================================================

def extract_search_terms(query: str) -> str:
    """Extract actual search term from natural language"""
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
    cleaned = re.sub(r'\s+(of|for|the|a|an)
    
    def _exact_substring_match(self, query: str):
        """Level 2: IMPROVED substring matching - handles multi-word queries"""
        import unicodedata
        import re
        
        def normalize(s):
            s = unicodedata.normalize("NFKC", s or "")
            return re.sub(r"\s+", " ", s).strip()
        
        q_norm = normalize(query).lower()
        if not q_norm or len(q_norm) < 2:
            return []
        
        # Split query into words for multi-word matching
        query_words = [w for w in q_norm.split() if len(w) > 1]
        
        all_data = self.collection.get(include=["documents", "metadatas"])
        documents = all_data.get("documents", [])
        metadatas = all_data.get("metadatas", [])
        ids = all_data.get("ids", [])
        
        matches = []
        for doc, meta, doc_id in zip(documents, metadatas, ids):
            doc_norm = normalize(doc).lower()
            
            # Method 1: Exact substring (for single words or exact phrases)
            if q_norm in doc_norm:
                coverage = len(q_norm) / len(doc_norm) if len(doc_norm) > 0 else 0
                position = doc_norm.find(q_norm)
                position_score = 1 - (position / len(doc_norm)) if len(doc_norm) > 0 else 1
                
                score = 0.90 + (coverage * 0.05) + (position_score * 0.05)
                score = min(score, 0.99)
                
                matches.append({
                    'id': doc_id,
                    'document': doc,
                    'metadata': meta,
                    'match_type': 'exact_substring',
                    'similarity_score': score,
                    'relevance': score
                })
            
            # Method 2: Multi-word matching ("Moishi Chen" finds doc with both words)
            elif len(query_words) > 1:
                all_words_present = all(word in doc_norm for word in query_words)
                
                if all_words_present:
                    total_word_length = sum(len(w) for w in query_words)
                    coverage = total_word_length / len(doc_norm) if len(doc_norm) > 0 else 0
                    
                    positions = [doc_norm.find(w) for w in query_words]
                    avg_position = sum(positions) / len(positions) if positions else 0
                    position_score = 1 - (avg_position / len(doc_norm)) if len(doc_norm) > 0 else 1
                    
                    # Multi-word matches score slightly lower
                    score = 0.85 + (coverage * 0.05) + (position_score * 0.05)
                    score = min(score, 0.94)
                    
                    matches.append({
                        'id': doc_id,
                        'document': doc,
                        'metadata': meta,
                        'match_type': 'multi_word',
                        'similarity_score': score,
                        'relevance': score
                    })
        
        return sorted(matches, key=lambda x: x['similarity_score'], reverse=True)
    
    # Save original search
    if not hasattr(VectorDBQASystem, 'search_original'):
        VectorDBQASystem.search_original = VectorDBQASystem.search
    
    # Add new method
    VectorDBQASystem._exact_substring_match = _exact_substring_match
    
    # Create patched search that checks substring first
    def search_patched(self, query: str, n_results: int = 5, **kwargs):
        """Patched search - checks exact substring BEFORE semantic"""
        
        # Try substring match first
        substring_matches = self._exact_substring_match(query)
        if substring_matches:
            # Format results
            return {
                'query': query,
                'original_query': query,
                'search_method': 'exact_substring',
                'results': substring_matches[:n_results]
            }
        
        # Fall back to original search
        return self.search_original(query, n_results=n_results, **kwargs)
    
    VectorDBQASystem.search = search_patched
    print("✅ Hierarchical search patch applied (substring check enabled)")

# Apply the patch immediately
patch_hierarchical_search()

# ============================================================================

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
            print("⚡ Using simple mode (free, fast + HIERARCHICAL search)")
            
            # Use patched search with substring check
            results = self.simple_system.search(query, n_results=5)
            
            if not results['results']:
                print(f"\n❌ No matches found for '{query}'")
                print("\n💡 Try:")
                print("   - Fewer characters (e.g., just first name)")
                print("   - Different spelling")
                print("   - Part of phone number")
                return None
            
            # Show top results
            search_method = results.get('search_method', 'unknown')
            print(f"\n✅ Found {len(results['results'])} match(es)")
            print(f"   Method: {search_method}")
            
            if results['results']:
                top_score = results['results'][0].get('similarity_score', 0)
                if search_method == 'exact_substring':
                    print(f"   🎯 EXACT SUBSTRING MATCH ({top_score:.0%})")
                elif top_score < 0.7:
                    print(f"   ⚠️  Best match is {top_score:.0%} similar (not exact)")
            
            print("\n" + "-"*60)
            
            for i, result in enumerate(results['results'], 1):
                doc = result['document']
                score = result.get('similarity_score', 0)
                match_type = result.get('match_type', 'unknown')
                
                # Color code by quality
                if match_type == 'exact_substring' or score >= 0.90:
                    quality = "🎯 EXACT"
                elif score >= 0.85:
                    quality = "✅ Excellent"
                elif score >= 0.70:
                    quality = "⚠️  Good"
                else:
                    quality = "🔶 Possible"
                
                print(f"\n{i}. {quality} match ({score:.0%})")
                
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
        print("🧠 SMART CHATBOT (WITH HIERARCHICAL SEARCH)")
        print("   Automatically chooses: Free search OR AI reasoning")
        print("   ✅ Now checks EXACT SUBSTRING before semantic!")
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
, '', cleaned, flags=re.IGNORECASE)
    
    return cleaned if cleaned else query

def patch_hierarchical_search():
    """Apply hierarchical search patch to VectorDBQASystem"""
    
    def _exact_substring_match(self, query: str):
        """Level 2: IMPROVED substring matching - handles multi-word queries"""
        import unicodedata
        import re
        
        def normalize(s):
            s = unicodedata.normalize("NFKC", s or "")
            return re.sub(r"\s+", " ", s).strip()
        
        q_norm = normalize(query).lower()
        if not q_norm or len(q_norm) < 2:
            return []
        
        # Split query into words for multi-word matching
        query_words = [w for w in q_norm.split() if len(w) > 1]
        
        all_data = self.collection.get(include=["documents", "metadatas"])
        documents = all_data.get("documents", [])
        metadatas = all_data.get("metadatas", [])
        ids = all_data.get("ids", [])
        
        matches = []
        for doc, meta, doc_id in zip(documents, metadatas, ids):
            doc_norm = normalize(doc).lower()
            
            # Method 1: Exact substring (for single words or exact phrases)
            if q_norm in doc_norm:
                coverage = len(q_norm) / len(doc_norm) if len(doc_norm) > 0 else 0
                position = doc_norm.find(q_norm)
                position_score = 1 - (position / len(doc_norm)) if len(doc_norm) > 0 else 1
                
                score = 0.90 + (coverage * 0.05) + (position_score * 0.05)
                score = min(score, 0.99)
                
                matches.append({
                    'id': doc_id,
                    'document': doc,
                    'metadata': meta,
                    'match_type': 'exact_substring',
                    'similarity_score': score,
                    'relevance': score
                })
            
            # Method 2: Multi-word matching ("Moishi Chen" finds doc with both words)
            elif len(query_words) > 1:
                all_words_present = all(word in doc_norm for word in query_words)
                
                if all_words_present:
                    total_word_length = sum(len(w) for w in query_words)
                    coverage = total_word_length / len(doc_norm) if len(doc_norm) > 0 else 0
                    
                    positions = [doc_norm.find(w) for w in query_words]
                    avg_position = sum(positions) / len(positions) if positions else 0
                    position_score = 1 - (avg_position / len(doc_norm)) if len(doc_norm) > 0 else 1
                    
                    # Multi-word matches score slightly lower
                    score = 0.85 + (coverage * 0.05) + (position_score * 0.05)
                    score = min(score, 0.94)
                    
                    matches.append({
                        'id': doc_id,
                        'document': doc,
                        'metadata': meta,
                        'match_type': 'multi_word',
                        'similarity_score': score,
                        'relevance': score
                    })
        
        return sorted(matches, key=lambda x: x['similarity_score'], reverse=True)
    
    # Save original search
    if not hasattr(VectorDBQASystem, 'search_original'):
        VectorDBQASystem.search_original = VectorDBQASystem.search
    
    # Add new method
    VectorDBQASystem._exact_substring_match = _exact_substring_match
    
    # Create patched search that checks substring first
    def search_patched(self, query: str, n_results: int = 5, **kwargs):
        """Patched search - checks exact substring BEFORE semantic"""
        
        # Try substring match first
        substring_matches = self._exact_substring_match(query)
        if substring_matches:
            # Format results
            return {
                'query': query,
                'original_query': query,
                'search_method': 'exact_substring',
                'results': substring_matches[:n_results]
            }
        
        # Fall back to original search
        return self.search_original(query, n_results=n_results, **kwargs)
    
    VectorDBQASystem.search = search_patched
    print("✅ Hierarchical search patch applied (substring check enabled)")

# Apply the patch immediately
patch_hierarchical_search()

# ============================================================================

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
            print("⚡ Using simple mode (free, fast + HIERARCHICAL search)")
            
            # Use patched search with substring check
            results = self.simple_system.search(query, n_results=5)
            
            if not results['results']:
                print(f"\n❌ No matches found for '{query}'")
                print("\n💡 Try:")
                print("   - Fewer characters (e.g., just first name)")
                print("   - Different spelling")
                print("   - Part of phone number")
                return None
            
            # Show top results
            search_method = results.get('search_method', 'unknown')
            print(f"\n✅ Found {len(results['results'])} match(es)")
            print(f"   Method: {search_method}")
            
            if results['results']:
                top_score = results['results'][0].get('similarity_score', 0)
                if search_method == 'exact_substring':
                    print(f"   🎯 EXACT SUBSTRING MATCH ({top_score:.0%})")
                elif top_score < 0.7:
                    print(f"   ⚠️  Best match is {top_score:.0%} similar (not exact)")
            
            print("\n" + "-"*60)
            
            for i, result in enumerate(results['results'], 1):
                doc = result['document']
                score = result.get('similarity_score', 0)
                match_type = result.get('match_type', 'unknown')
                
                # Color code by quality
                if match_type == 'exact_substring' or score >= 0.90:
                    quality = "🎯 EXACT"
                elif score >= 0.85:
                    quality = "✅ Excellent"
                elif score >= 0.70:
                    quality = "⚠️  Good"
                else:
                    quality = "🔶 Possible"
                
                print(f"\n{i}. {quality} match ({score:.0%})")
                
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
        print("🧠 SMART CHATBOT (WITH HIERARCHICAL SEARCH)")
        print("   Automatically chooses: Free search OR AI reasoning")
        print("   ✅ Now checks EXACT SUBSTRING before semantic!")
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
