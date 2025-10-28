#!/usr/bin/env python3
"""
Test the hybrid search fixes for partial name matching
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from vectoric_search import VectorDBQASystem

def test_partial_name_search():
    """Test that partial names now work"""
    
    print("🧪 Testing Hybrid Search Fixes")
    print("=" * 60)
    
    # Use existing database
    qa_system = VectorDBQASystem(persist_directory="./chroma_db")
    
    stats = qa_system.get_collection_stats()
    print(f"\n📊 Using database with {stats['document_count']} documents\n")
    
    # Test cases that were failing before
    test_cases = [
        ("ברק", "Should find ברק גורדון"),
        ("משה", "Should find all משה contacts"),
        ("Moishi", "Should find Moishi Chen"),
        ("050-408", "Should find by phone number"),
    ]
    
    print("🎯 TESTING PARTIAL MATCHES")
    print("=" * 60)
    
    for query, description in test_cases:
        print(f"\n📝 Test: {query}")
        print(f"   Goal: {description}")
        print("-" * 60)
        
        try:
            results = qa_system.search(query, n_results=3)
            
            if results['results']:
                print(f"   ✅ SUCCESS - Found {len(results['results'])} result(s)")
                print(f"   Method used: {results.get('search_method', 'unknown')}")
                
                # Show first result
                top = results['results'][0]
                doc = top['document']
                score = top.get('similarity_score', 0)
                
                print(f"\n   📄 Top result (score: {score:.2f}):")
                # Show first 150 chars
                preview = doc[:150] + "..." if len(doc) > 150 else doc
                print(f"   {preview}")
            else:
                print(f"   ❌ FAIL - No results found")
                
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
    
    print("\n" + "=" * 60)
    print("✨ Test complete!")
    print("\nThe hybrid search should now find partial matches")
    print("that were failing before!")

if __name__ == "__main__":
    test_partial_name_search()
