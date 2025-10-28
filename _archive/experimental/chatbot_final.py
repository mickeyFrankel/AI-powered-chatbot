#!/usr/bin/env python3
"""
AI-FIRST + HIERARCHICAL SEARCH
Combines AI query understanding WITH proper hierarchical search
"""

import os
import sys
import json
from pathlib import Path
import unicodedata
import re

os.environ['ANONYMIZED_TELEMETRY'] = 'False'
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))

from vectoric_search import VectorDBQASystem
from openai import OpenAI

try:
    from rapidfuzz import fuzz
    HAS_FUZZY = True
except ImportError:
    HAS_FUZZY = False

# ============================================================================
# AI QUERY ANALYZER
# ============================================================================

class AIQueryAnalyzer:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY required")
        self.llm = OpenAI(api_key=api_key)
    
    def analyze_query(self, user_query: str) -> dict:
        system_prompt = """Extract the core search term from queries in ANY language.

Rules:
- "phone of X" → extract "X"
- "number of X" → extract "X"
- "contact info for X" → extract "X"
- Just "X" → extract "X"

Return ONLY JSON:
{"search_term": "extracted term", "intent": "find_contact", "confidence": 0.95}"""

        try:
            response = self.llm.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f'Query: "{user_query}"\n\nExtract:'}
                ],
                temperature=0,
                max_tokens=150
            )
            
            content = response.choices[0].message.content.strip()
            if content.startswith('```'):
                content = content.split('```')[1]
                if content.startswith('json'):
                    content = content[4:]
                content = content.strip()
            
            result = json.loads(content)
            return result if 'search_term' in result else {
                "search_term": user_query,
                "intent": "find_contact",
                "confidence": 0.5
            }
        except:
            return {"search_term": user_query, "intent": "find_contact", "confidence": 0.5}

# ============================================================================
# HIERARCHICAL SEARCH (INTEGRATED)
# ============================================================================

def normalize(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    return re.sub(r"\s+", " ", s).strip()

def hierarchical_search(qa_system, query: str, n_results: int = 5) -> dict:
    """
    Proper hierarchical search:
    1. Exact substring
    2. Fuzzy match
    3. Semantic search
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
    # LEVEL 1: EXACT SUBSTRING
    # ========================================================================
    query_words = [w for w in q_norm.split() if len(w) > 1]
    
    for doc, meta, doc_id in zip(documents, metadatas, ids):
        doc_norm = normalize(doc).lower()
        
        # Method A: Exact substring
        if q_norm in doc_norm:
            position = doc_norm.find(q_norm)
            coverage = len(q_norm) / len(doc_norm) if len(doc_norm) > 0 else 0
            position_score = 1 - (position / len(doc_norm)) if len(doc_norm) > 0 else 1
            
            score = 0.90 + (position_score * 0.05) + (coverage * 0.05)
            score = min(score, 0.99)
            
            results.append({
                'id': doc_id,
                'document': doc,
                'metadata': meta,
                'tier': 3,
                'similarity_score': score,
                'match_type': 'exact_substring'
            })
        
        # Method B: All words present
        elif len(query_words) > 1:
            if all(word in doc_norm for word in query_words):
                total_len = sum(len(w) for w in query_words)
                coverage = total_len / len(doc_norm) if len(doc_norm) > 0 else 0
                positions = [doc_norm.find(w) for w in query_words]
                avg_pos = sum(positions) / len(positions) if positions else 0
                pos_score = 1 - (avg_pos / len(doc_norm)) if len(doc_norm) > 0 else 1
                
                score = 0.85 + (pos_score * 0.05) + (coverage * 0.05)
                score = min(score, 0.94)
                
                results.append({
                    'id': doc_id,
                    'document': doc,
                    'metadata': meta,
                    'tier': 3,
                    'similarity_score': score,
                    'match_type': 'multi_word'
                })
    
    # If we found substring matches, return them
    if results:
        results.sort(key=lambda x: x['similarity_score'], reverse=True)
        return {
            'query': query,
            'search_method': 'exact_substring',
            'results': results[:n_results]
        }
    
    # ========================================================================
    # LEVEL 2: FUZZY MATCH
    # ========================================================================
    if HAS_FUZZY:
        for doc, meta, doc_id in zip(documents, metadatas, ids):
            doc_norm = normalize(doc).lower()
            
            # Fuzzy on document
            fuzzy_score = fuzz.token_set_ratio(q_norm, doc_norm)
            
            # Also try on name fields
            name = (meta.get('First Name', '') + ' ' + meta.get('Last Name', '')).strip()
            if name:
                name_norm = normalize(name).lower()
                name_fuzzy = fuzz.ratio(q_norm, name_norm)
                fuzzy_score = max(fuzzy_score, name_fuzzy)
            
            if fuzzy_score >= 70:
                results.append({
                    'id': doc_id,
                    'document': doc,
                    'metadata': meta,
                    'tier': 2,
                    'similarity_score': fuzzy_score / 100,
                    'match_type': 'fuzzy',
                    'fuzzy_score': fuzzy_score
                })
    
    # If we found fuzzy matches, return them
    if results:
        results.sort(key=lambda x: x['similarity_score'], reverse=True)
        return {
            'query': query,
            'search_method': 'fuzzy',
            'results': results[:n_results]
        }
    
    # ========================================================================
    # LEVEL 3: SEMANTIC SEARCH
    # ========================================================================
    try:
        query_embedding = qa_system.embedding_model.encode([q_norm])
        semantic_results = qa_system.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=n_results
        )
        
        if semantic_results['documents'] and semantic_results['documents'][0]:
            for i in range(len(semantic_results['documents'][0])):
                distance = semantic_results['distances'][0][i]
                cosine_sim = 1 - (distance / 2)
                
                if cosine_sim >= 0.3:
                    results.append({
                        'id': semantic_results['ids'][0][i],
                        'document': semantic_results['documents'][0][i],
                        'metadata': semantic_results['metadatas'][0][i],
                        'tier': 1,
                        'similarity_score': cosine_sim,
                        'match_type': 'semantic'
                    })
    except Exception as e:
        print(f"   ⚠️  Semantic search failed: {e}")
    
    results.sort(key=lambda x: x['similarity_score'], reverse=True)
    return {
        'query': query,
        'search_method': 'semantic',
        'results': results[:n_results]
    }

# ============================================================================
# AI-FIRST CHATBOT WITH PROPER HIERARCHICAL SEARCH
# ============================================================================

class ProperAIChatbot:
    def __init__(self, persist_directory: str = "./chroma_db"):
        print("🤖 Initializing AI-First Chatbot with Hierarchical Search...")
        
        self.analyzer = AIQueryAnalyzer()
        self.search_system = VectorDBQASystem(persist_directory=persist_directory)
        
        print("✅ Ready!")
    
    def search(self, user_query: str, n_results: int = 5):
        print(f"\n{'='*70}")
        print(f"🔍 Query: '{user_query}'")
        print(f"{'='*70}")
        
        # Step 1: AI analyzes query
        print("\n🧠 AI Analysis...")
        analysis = self.analyzer.analyze_query(user_query)
        
        search_term = analysis['search_term']
        intent = analysis['intent']
        confidence = analysis.get('confidence', 0.5)
        
        if search_term != user_query:
            print(f"   📝 Extracted: '{search_term}' from '{user_query}'")
        else:
            print(f"   ✓ Direct search: '{search_term}'")
        
        print(f"   Intent: {intent} | Confidence: {confidence:.0%}")
        
        # Step 2: Hierarchical search
        print(f"\n⚡ Hierarchical Search for: '{search_term}'")
        
        results = hierarchical_search(self.search_system, search_term, n_results)
        
        # Step 3: Display
        self._display(results, intent)
        
        return results
    
    def _display(self, results: dict, intent: str):
        if not results.get('results'):
            print(f"\n❌ No matches found")
            return
        
        method = results.get('search_method', 'unknown')
        result_list = results['results']
        
        # Tier labels
        tier_labels = {
            3: "⭐ EXACT SUBSTRING",
            2: "🔤 FUZZY MATCH",
            1: "🔗 SEMANTIC"
        }
        
        # Group by tier
        by_tier = {}
        for r in result_list:
            tier = r.get('tier', 1)
            if tier not in by_tier:
                by_tier[tier] = []
            by_tier[tier].append(r)
        
        print(f"\n✅ Found {len(result_list)} match(es) using {method}")
        print(f"\n{'-'*70}\n")
        
        rank = 1
        for tier in sorted(by_tier.keys(), reverse=True):
            tier_results = by_tier[tier]
            label = tier_labels.get(tier, "MATCH")
            
            if len(by_tier) > 1:  # Only show tier label if multiple tiers
                print(f"{label} ({len(tier_results)} result{'s' if len(tier_results) > 1 else ''})")
                print("─" * 70 + "\n")
            
            for r in tier_results:
                score = r['similarity_score']
                
                emoji = "🎯" if score >= 0.90 else "✅" if score >= 0.70 else "🔶"
                
                print(f"{rank}. {emoji} {score:.0%} match")
                
                # Show key fields
                doc = r['document']
                if '|' in doc:
                    for part in doc.split('|')[:6]:
                        if ':' in part:
                            key, val = part.split(':', 1)
                            key, val = key.strip(), val.strip()
                            
                            if val and key in ['First Name', 'Last Name', 'Phone 1 - Value', 
                                              'Phone 2 - Value', 'E-mail 1 - Value', 'Organization Name']:
                                print(f"   {key}: {val}")
                
                print()
                rank += 1
    
    def interactive(self):
        print("\n" + "="*70)
        print("🧠 AI-FIRST + HIERARCHICAL SEARCH CHATBOT")
        print("   Smart AI query understanding + Proper search ranking")
        print("="*70)
        
        stats = self.search_system.get_collection_stats()
        print(f"\n📊 Database: {stats['document_count']} contacts")
        
        print("\n💡 Try:")
        print("   - 'אבי אתרוגים'")
        print("   - 'number of אבי אתרוגים'")
        print("   - 'phone of Noah'")
        print("   - 'Moishi'")
        
        print("\n📝 Commands: 'quit'\n")
        
        while True:
            try:
                query = input("🔍 Search: ").strip()
                
                if not query:
                    continue
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Goodbye!")
                    break
                
                self.search(query, n_results=5)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                import traceback
                traceback.print_exc()

def main():
    try:
        chatbot = ProperAIChatbot()
        chatbot.interactive()
    except RuntimeError as e:
        print(f"\n❌ {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
