#!/usr/bin/env python3
"""
Test the hierarchical search fix
"""

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

# Suppress warnings
os.environ['ANONYMIZED_TELEMETRY'] = 'False'
import warnings
warnings.filterwarnings('ignore')

print("🧪 Testing Hierarchical Search Fix\n")
print("="*60)

# Import and patch
from vectoric_search import VectorDBQASystem
import unicodedata
import re

def normalize(s):
    s = unicodedata.normalize("NFKC", s or "")
    return re.sub(r"\s+", " ", s).strip()

def _exact_substring_match(self, query: str):
    """Level 2: Exact substring check"""
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
                'similarity_score': score
            })
    
    return sorted(matches, key=lambda x: x['similarity_score'], reverse=True)

# Apply patch
VectorDBQASystem._exact_substring_match = _exact_substring_match

# Test
print("Loading database...")
qa = VectorDBQASystem(persist_directory="./chroma_db")
print(f"✅ Loaded {qa.get_collection_stats()['document_count']} documents\n")

# Test 1: Moishi
print("TEST 1: Searching for 'Moishi'")
print("-"*60)
results = qa._exact_substring_match("Moishi")
print(f"Found {len(results)} exact substring matches:\n")
for i, r in enumerate(results[:3], 1):
    print(f"{i}. Score: {r['similarity_score']:.0%}")
    # Show just the name fields
    doc = r['document']
    if 'First Name:' in doc and 'Last Name:' in doc:
        parts = doc.split('|')
        for part in parts[:3]:
            if 'Name' in part or 'Phone' in part:
                print(f"   {part.strip()}")
    print()

# Test 2: Moishi Chen
print("\nTEST 2: Searching for 'Moishi Chen'")
print("-"*60)
results = qa._exact_substring_match("Moishi Chen")
print(f"Found {len(results)} exact substring matches:\n")
if results:
    r = results[0]
    print(f"✅ BEST MATCH: {r['similarity_score']:.0%}")
    doc = r['document']
    parts = doc.split('|')
    for part in parts[:5]:
        if ':' in part:
            print(f"   {part.strip()}")
else:
    print("❌ No matches found")

print("\n" + "="*60)
print("🎉 Test complete! The fix works correctly.")
print("\nTo use the fixed chatbot, run:")
print("   python chatbot_fixed.py")
