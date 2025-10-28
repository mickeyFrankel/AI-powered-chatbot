#!/usr/bin/env python3
"""
Tier-Based Multi-Method Search Implementation
Formal ranking: Exact > Substring > Fuzzy > Semantic
"""

import os
import sys
from pathlib import Path

os.environ['ANONYMIZED_TELEMETRY'] = 'False'
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))

from vectoric_search import VectorDBQASystem, AdvancedVectorDBQASystem
import re
import unicodedata

try:
    from rapidfuzz import fuzz
    HAS_FUZZY = True
except ImportError:
    HAS_FUZZY = False

# ============================================================================
# TERM EXTRACTION
# ============================================================================

def extract_search_terms(query: str) -> str:
    """Extract name from natural language"""
    filler_patterns = [
        r'\b(phone|telephone|tel|mobile|cell)\s+(number|no\.?|#)?\s+(of|for)?\s*',
        r'\b(contact|info|information|details)\s+(of|for|about)?\s*',
        r'\be-?mail\s+(of|for|address)?\s*',
        r'\b(find|get|show|give\s+me|tell\s+me)\s+(the)?\s*',
    ]
    
    cleaned = query
    for pattern in filler_patterns:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
    
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned if cleaned else query

# ============================================================================
# TIER-BASED SEARCH IMPLEMENTATION
# ============================================================================

def normalize(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    return re.sub(r"\s+", " ", s).strip()

def tier_based_search(qa_system, query: str, n_results: int = 5):
    """
    Multi-tier search with formal ranking
    Returns all relevant results across tiers
    """
    
    q_norm = normalize(query).lower()
    if not q_norm:
        return {'results': []}
    
    all_data = qa_system.collection.get(include=["documents", "metadatas"])
    documents = all_data.get("documents", [])
    metadatas = all_data.get("metadatas", [])
    ids = all_data.get("ids", [])
    
    results = []
    
    # ========================================================================
    # TIER 4: EXACT MATCH (4000-4100)
    # ========================================================================
    for doc, meta, doc_id in zip(documents, metadatas, ids):
        doc_norm = normalize(doc).lower()
        if q_norm == doc_norm:
            results.append({
                'id': doc_id,
                'document': doc,
                'metadata': meta,
                'tier': 4,
                'raw_score': 100,
                'final_score': 4100,
                'match_type': 'exact',
                'explanation': 'Exact match'
            })
    
    # ========================================================================
    # TIER 3: EXACT SUBSTRING (3000-3099)
    # ========================================================================
    query_words = [w for w in q_norm.split() if len(w) > 1]
    
    for doc, meta, doc_id in zip(documents, metadatas, ids):
        doc_norm = normalize(doc).lower()
        
        # Method 1: Exact substring
        if q_norm in doc_norm:
            position = doc_norm.find(q_norm)
            coverage = len(q_norm) / len(doc_norm) if len(doc_norm) > 0 else 0
            position_score = 1 - (position / len(doc_norm)) if len(doc_norm) > 0 else 1
            
            raw_score = 90 + (position_score * 5) + (coverage * 5)
            raw_score = min(raw_score, 99)
            
            results.append({
                'id': doc_id,
                'document': doc,
                'metadata': meta,
                'tier': 3,
                'raw_score': raw_score,
                'final_score': 3000 + raw_score,
                'match_type': 'exact_substring',
                'explanation': f'Contains "{query}"'
            })
        
        # Method 2: Multi-word (all words present)
        elif len(query_words) > 1:
            all_words_present = all(word in doc_norm for word in query_words)
            if all_words_present:
                total_len = sum(len(w) for w in query_words)
                coverage = total_len / len(doc_norm) if len(doc_norm) > 0 else 0
                positions = [doc_norm.find(w) for w in query_words]
                avg_position = sum(positions) / len(positions) if positions else 0
                position_score = 1 - (avg_position / len(doc_norm)) if len(doc_norm) > 0 else 1
                
                raw_score = 85 + (position_score * 5) + (coverage * 5)
                raw_score = min(raw_score, 94)
                
                results.append({
                    'id': doc_id,
                    'document': doc,
                    'metadata': meta,
                    'tier': 3,
                    'raw_score': raw_score,
                    'final_score': 3000 + raw_score,
                    'match_type': 'multi_word',
                    'explanation': f'Contains all words: {", ".join(query_words)}'
                })
    
    # ========================================================================
    # TIER 2: FUZZY MATCH (2000-2099)
    # ========================================================================
    if HAS_FUZZY:
        for doc, meta, doc_id in zip(documents, metadatas, ids):
            # Skip if already matched in higher tier
            if any(r['id'] == doc_id and r['tier'] >= 3 for r in results):
                continue
            
            doc_norm = normalize(doc).lower()
            
            # Fuzzy match on document
            fuzzy_score = fuzz.token_set_ratio(q_norm, doc_norm)
            
            # Also try fuzzy on name field if available
            name = meta.get('First Name', '') + ' ' + meta.get('Last Name', '')
            name_norm = normalize(name).lower()
            if name_norm:
                name_fuzzy = fuzz.ratio(q_norm, name_norm)
                fuzzy_score = max(fuzzy_score, name_fuzzy)
            
            if fuzzy_score >= 70:
                results.append({
                    'id': doc_id,
                    'document': doc,
                    'metadata': meta,
                    'tier': 2,
                    'raw_score': fuzzy_score,
                    'final_score': 2000 + fuzzy_score,
                    'match_type': 'fuzzy',
                    'explanation': f'Similar spelling ({fuzzy_score}% match)'
                })
    
    # ========================================================================
    # TIER 1: SEMANTIC MATCH (1000-1099)
    # ========================================================================
    # Only run semantic if we have few results from higher tiers
    high_tier_count = sum(1 for r in results if r['tier'] >= 2)
    
    if high_tier_count < 3:
        try:
            query_embedding = qa_system.embedding_model.encode([q_norm])
            semantic_results = qa_system.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=10
            )
            
            if semantic_results['documents'] and semantic_results['documents'][0]:
                for i in range(len(semantic_results['documents'][0])):
                    doc_id = semantic_results['ids'][0][i]
                    
                    # Skip if already in higher tier
                    if any(r['id'] == doc_id and r['tier'] >= 2 for r in results):
                        continue
                    
                    distance = semantic_results['distances'][0][i]
                    cosine_sim = 1 - (distance / 2)
                    
                    if cosine_sim >= 0.3:
                        raw_score = cosine_sim * 100
                        
                        results.append({
                            'id': doc_id,
                            'document': semantic_results['documents'][0][i],
                            'metadata': semantic_results['metadatas'][0][i],
                            'tier': 1,
                            'raw_score': raw_score,
                            'final_score': 1000 + raw_score,
                            'match_type': 'semantic',
                            'explanation': f'Semantically related (cosine: {cosine_sim:.2f})'
                        })
        except Exception as e:
            print(f"   ⚠️  Semantic search unavailable: {e}")
    
    # ========================================================================
    # SORT AND DEDUPLICATE
    # ========================================================================
    
    # Sort by final score (descending)
    results.sort(key=lambda x: x['final_score'], reverse=True)
    
    # Deduplicate (keep highest tier for each document)
    seen_ids = set()
    deduped = []
    for r in results:
        if r['id'] not in seen_ids:
            seen_ids.add(r['id'])
            deduped.append(r)
    
    # Return top N results
    return {
        'query': query,
        'results': deduped[:n_results]
    }

# ============================================================================
# DISPLAY FUNCTIONS
# ============================================================================

def display_tier_results(results_dict):
    """Display results grouped by tier with visual indicators"""
    
    query = results_dict['query']
    results = results_dict['results']
    
    if not results:
        print(f"\n❌ No matches found for '{query}'")
        return
    
    # Group by tier
    by_tier = {}
    for r in results:
        tier = r['tier']
        if tier not in by_tier:
            by_tier[tier] = []
        by_tier[tier].append(r)
    
    tier_info = {
        4: ("🎯 EXACT MATCH", "green"),
        3: ("⭐ CONTAINS", "blue"),
        2: ("🔤 SIMILAR / DID YOU MEAN", "yellow"),
        1: ("🔗 RELATED", "purple")
    }
    
    print(f"\n{'='*70}")
    print(f"Results for: '{query}' ({len(results)} found)")
    print(f"{'='*70}\n")
    
    overall_rank = 1
    
    # Display each tier
    for tier in sorted(by_tier.keys(), reverse=True):
        tier_results = by_tier[tier]
        tier_label, color = tier_info[tier]
        
        print(f"\n{tier_label} ({len(tier_results)} result{'s' if len(tier_results) > 1 else ''})")
        print("─" * 70)
        
        for r in tier_results:
            score_pct = (r['raw_score'] / 100) if r['tier'] <= 2 else (r['raw_score'])
            
            print(f"\n{overall_rank}. Score: {r['final_score']} ({score_pct:.0f}% confidence)")
            print(f"   {r['explanation']}")
            
            # Extract and show key fields
            doc = r['document']
            if '|' in doc:
                for part in doc.split('|')[:5]:
                    if ':' in part:
                        key, val = part.split(':', 1)
                        key, val = key.strip(), val.strip()
                        if val and key in ['First Name', 'Last Name', 'Phone 1 - Value', 
                                          'Phone 2 - Value', 'E-mail 1 - Value']:
                            print(f"   {key}: {val}")
            
            overall_rank += 1
    
    print(f"\n{'='*70}\n")

# ============================================================================
# MAIN CHATBOT
# ============================================================================

class TierBasedChatbot:
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.qa_system = VectorDBQASystem(persist_directory=persist_directory)
    
    def search(self, query: str, n_results: int = 5):
        # Extract search terms
        extracted = extract_search_terms(query)
        if extracted != query:
            print(f"📝 Extracted: '{extracted}' from '{query}'")
        
        # Run tier-based search
        results = tier_based_search(self.qa_system, extracted, n_results)
        
        # Display with tier formatting
        display_tier_results(results)
        
        return results
    
    def interactive(self):
        print("\n" + "="*70)
        print("🧠 TIER-BASED SEARCH CHATBOT")
        print("   Shows both exact matches AND possible corrections!")
        print("="*70)
        
        stats = self.qa_system.get_collection_stats()
        print(f"📊 Documents: {stats['document_count']}")
        
        print("\n💡 Try:")
        print("   - 'Moishi' (shows Moishi Chen + Moshavi)")
        print("   - 'phone number of Noam'")
        print("   - 'Moyshi' (typo)")
        print("\n📝 Commands: 'quit' to exit\n")
        
        while True:
            try:
                query = input("🔍 Search: ").strip()
                
                if not query:
                    continue
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                self.search(query, n_results=5)
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                import traceback
                traceback.print_exc()

def main():
    chatbot = TierBasedChatbot()
    chatbot.interactive()

if __name__ == "__main__":
    main()
