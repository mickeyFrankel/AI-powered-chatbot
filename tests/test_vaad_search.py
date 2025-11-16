#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')
from vectoric_search import AdvancedVectorDBQASystem

qa = AdvancedVectorDBQASystem()

print("Testing different search methods:\n")

# Test 1: Keyword search (should work)
print("1. Keyword search for 'ועד בית':")
results = qa.search_full_text('ועד בית', limit=10)
print(f"   Found: {len(results)} results")
for r in results:
    print(f"   - {r['name']}")
print()

# Test 2: Keyword search for 'ועד' only
print("2. Keyword search for 'ועד':")
results = qa.search_full_text('ועד', limit=10)
print(f"   Found: {len(results)} results")
for r in results[:5]:
    print(f"   - {r['name']}")
print()

# Test 3: Semantic search (what agent uses)
print("3. Semantic search for 'ועד בית':")
results = qa.search('ועד בית', n_results=5)
print(f"   Found: {len(results.get('results', []))} results")
for r in results.get('results', [])[:5]:
    print(f"   - {r['metadata'].get('name', 'Unknown')}")
print()

# Test 4: Semantic search for Hebrew phrase
print("4. Semantic search for 'חבר בוועד בית':")
results = qa.search('חבר בוועד בית', n_results=5)
print(f"   Found: {len(results.get('results', []))} results")
for r in results.get('results', [])[:5]:
    print(f"   - {r['metadata'].get('name', 'Unknown')}")
print()

# Test 5: What tool should agent use?
print("\n" + "="*80)
print("DIAGNOSIS:")
print("="*80)
print("User query: 'כל מי שחבר בוועד בית' (everyone in house committee)")
print("\nCorrect tool: search_keyword with keyword='ועד בית'")
print("Likely problem: Agent uses semantic search instead of keyword search")
print("\nSolution: Agent needs to recognize 'ועד בית' as a specific keyword/role")
print("          and use search_keyword tool, not semantic search tool")
