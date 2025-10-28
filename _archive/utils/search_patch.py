"""
Surgical patch for vectoric_search.py - adds hierarchical search
This patches ONLY the search() method while preserving everything else
"""

def get_hierarchical_search_methods():
    """
    Returns the new hierarchical search methods to add to VectorDBQASystem
    """
    
    def _exact_match(self, query: str):
        """Level 1: Exact match (100% similarity)"""
        import unicodedata
        import re
        
        def normalize(s):
            s = unicodedata.normalize("NFKC", s or "")
            return re.sub(r"\s+", " ", s).strip()
        
        q_norm = normalize(query).lower()
        if not q_norm:
            return []
        
        all_data = self.collection.get(include=["documents", "metadatas"])
        documents = all_data.get("documents", [])
        metadatas = all_data.get("metadatas", [])
        ids = all_data.get("ids", [])
        
        matches = []
        for doc, meta, doc_id in zip(documents, metadatas, ids):
            doc_norm = normalize(doc).lower()
            if q_norm == doc_norm:
                matches.append({
                    'id': doc_id,
                    'document': doc,
                    'metadata': meta,
                    'match_type': 'exact',
                    'similarity_score': 1.0,
                    'relevance': 1.0
                })
        
        return matches
    
    def _exact_substring_match(self, query: str):
        """Level 2: Exact substring containment (90-99% similarity)"""
        import unicodedata
        import re
        
        def normalize(s):
            s = unicodedata.normalize("NFKC", s or "")
            return re.sub(r"\s+", " ", s).strip()
        
        q_norm = normalize(query).lower()
        if not q_norm or len(q_norm) < 2:
            return []
        
        all_data = self.collection.get(include=["documents", "metadatas"])
        documents = all_data.get("documents", [])
        metadatas = all_data.get("metadatas", [])
        ids = all_data.get("ids", [])
        
        matches = []
        for doc, meta, doc_id in zip(documents, metadatas, ids):
            doc_norm = normalize(doc).lower()
            
            # Check if query is substring
            if q_norm in doc_norm:
                # Calculate score based on coverage and position
                coverage = len(q_norm) / len(doc_norm) if len(doc_norm) > 0 else 0
                position = doc_norm.find(q_norm)
                position_score = 1 - (position / len(doc_norm)) if len(doc_norm) > 0 else 1
                
                # Score: 0.90 base + bonuses
                score = 0.90 + (coverage * 0.05) + (position_score * 0.05)
                score = min(score, 0.99)  # Cap at 0.99
                
                matches.append({
                    'id': doc_id,
                    'document': doc,
                    'metadata': meta,
                    'match_type': 'exact_substring',
                    'similarity_score': score,
                    'relevance': score
                })
        
        return sorted(matches, key=lambda x: x['similarity_score'], reverse=True)
    
    def search_hierarchical(self, query: str, n_results: int = 5, **kwargs):
        """
        NEW hierarchical search - checks exact substring BEFORE semantic
        This is the patched version that fixes the search logic
        """
        import unicodedata
        import re
        
        def normalize(s):
            s = unicodedata.normalize("NFKC", s or "")
            return re.sub(r"\s+", " ", s).strip()
        
        original_query = query
        query_normalized = normalize(query)
        
        print(f"\n🔍 Hierarchical Search: '{query_normalized}'")
        
        # Level 1: Exact Match
        exact_matches = self._exact_match(query_normalized)
        if exact_matches:
            print(f"✅ Level 1: Found {len(exact_matches)} EXACT match(es)")
            results = {
                'query': query_normalized,
                'original_query': original_query,
                'search_method': 'exact',
                'results': exact_matches[:n_results]
            }
            return results
        
        # Level 2: Exact Substring (THIS IS THE KEY FIX!)
        substring_matches = self._exact_substring_match(query_normalized)
        if substring_matches:
            print(f"✅ Level 2: Found {len(substring_matches)} EXACT SUBSTRING match(es)")
            results = {
                'query': query_normalized,
                'original_query': original_query,
                'search_method': 'exact_substring',
                'results': substring_matches[:n_results]
            }
            return results
        
        # Level 3: Fallback to original semantic search
        print(f"⚠️  No exact/substring matches, falling back to semantic search")
        return self.search_original(query, n_results=n_results, **kwargs)
    
    return {
        '_exact_match': _exact_match,
        '_exact_substring_match': _exact_substring_match,
        'search_hierarchical': search_hierarchical
    }


def apply_patch():
    """
    Apply the hierarchical search patch to vectoric_search.py
    """
    import sys
    from pathlib import Path
    
    # Import the module
    sys.path.insert(0, str(Path(__file__).parent))
    from vectoric_search import VectorDBQASystem
    
    # Get the new methods
    methods = get_hierarchical_search_methods()
    
    # Save original search method
    VectorDBQASystem.search_original = VectorDBQASystem.search
    
    # Add new methods to the class
    for method_name, method_func in methods.items():
        setattr(VectorDBQASystem, method_name, method_func)
    
    # Replace search with hierarchical version
    VectorDBQASystem.search = VectorDBQASystem.search_hierarchical
    
    print("✅ Hierarchical search patch applied!")
    print("   - Added: _exact_match()")
    print("   - Added: _exact_substring_match()")
    print("   - Patched: search() → search_hierarchical()")
    
    return VectorDBQASystem


if __name__ == "__main__":
    # Test the patch
    qa = apply_patch()(persist_directory="./chroma_db")
    
    print(f"\n📊 Database: {qa.get_collection_stats()['document_count']} documents")
    
    # Test search
    print("\n" + "="*60)
    results = qa.search("Moishi", n_results=3)
    
    print(f"\n🎯 Found {len(results['results'])} results:")
    for i, r in enumerate(results['results'], 1):
        print(f"\n{i}. {r['match_type'].upper()} - {r['similarity_score']:.0%}")
        print(f"   {r['document'][:100]}...")
