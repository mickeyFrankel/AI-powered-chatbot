#!/usr/bin/env python3
"""
Test script to verify top 5 fuzzy search and Hebrew Unicode normalization
"""

from vectoric_search import AdvancedVectorDBQASystem

def test_fuzzy_search():
    print("="*60)
    print("Testing Fuzzy Search (Top 5 Default)")
    print("="*60)
    
    qa = AdvancedVectorDBQASystem()
    
    # Test 1: Hebrew search with Unicode variations
    print("\n1. Testing Hebrew Unicode: 'אהובתי'")
    result1 = qa.search('אהובתי', n_results=5)
    print(f"   Found: {len(result1['results'])} results")
    for i, r in enumerate(result1['results'][:3], 1):
        name = r['metadata'].get('name', 'Unknown')
        score = r.get('similarity_score', 0)
        print(f"   {i}. {name} (similarity: {score:.3f})")
    
    # Test 2: Hebrew search with different Unicode
    print("\n2. Testing Hebrew Unicode: 'אהובתיֿ' (with combining char)")
    result2 = qa.search('אהובתיֿ', n_results=5)
    print(f"   Found: {len(result2['results'])} results")
    for i, r in enumerate(result2['results'][:3], 1):
        name = r['metadata'].get('name', 'Unknown')
        score = r.get('similarity_score', 0)
        print(f"   {i}. {name} (similarity: {score:.3f})")
    
    # Test 3: Relationship query
    print("\n3. Testing relationship understanding: 'חמותי'")
    result3 = qa.search('חמותי', n_results=5)
    print(f"   Found: {len(result3['results'])} results")
    for i, r in enumerate(result3['results'][:3], 1):
        name = r['metadata'].get('name', 'Unknown')
        score = r.get('similarity_score', 0)
        print(f"   {i}. {name} (similarity: {score:.3f})")
    
    print("\n" + "="*60)
    print("✅ All tests completed!")
    print("="*60)
    
    # Verify both searches found the same top result
    if result1['results'] and result2['results']:
        top1 = result1['results'][0]['metadata'].get('name')
        top2 = result2['results'][0]['metadata'].get('name')
        if top1 == top2:
            print(f"\n🎯 SUCCESS: Unicode normalization working!")
            print(f"   Both 'אהובתי' and 'אהובתיֿ' found: {top1}")
        else:
            print(f"\n⚠️  Different top results:")
            print(f"   'אהובתי' → {top1}")
            print(f"   'אהובתיֿ' → {top2}")

if __name__ == "__main__":
    test_fuzzy_search()
