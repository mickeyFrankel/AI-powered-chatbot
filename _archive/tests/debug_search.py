#!/usr/bin/env python3
"""
Diagnostic tool - Show what's ACTUALLY in the database
Helps debug why searches fail
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from vectoric_search import VectorDBQASystem

def show_all_names_containing(substring: str):
    """Show ALL documents containing substring"""
    
    print(f"\n🔍 Searching database for documents containing: '{substring}'")
    print("="*60)
    
    qa_system = VectorDBQASystem(persist_directory="./chroma_db")
    
    # Get ALL documents
    all_data = qa_system.collection.get(include=["documents", "metadatas"])
    documents = all_data.get("documents", [])
    metadatas = all_data.get("metadatas", [])
    
    print(f"📊 Total documents in database: {len(documents)}")
    
    # Search for substring (case-insensitive)
    substring_lower = substring.lower()
    matches = []
    
    for i, (doc, meta) in enumerate(zip(documents, metadatas)):
        if substring_lower in doc.lower():
            matches.append((i, doc, meta))
    
    print(f"✅ Found {len(matches)} documents containing '{substring}'\n")
    
    if not matches:
        print("❌ NO MATCHES FOUND!")
        print("\n💡 This means:")
        print("   1. The name is spelled differently")
        print("   2. The name is not in the database")
        print("   3. Try searching for parts of the name")
        return
    
    # Show matches
    print("-"*60)
    for idx, (doc_idx, doc, meta) in enumerate(matches[:20], 1):  # Show first 20
        print(f"\n{idx}. Document #{doc_idx}")
        
        # Parse document
        if '|' in doc:
            parts = doc.split('|')
            shown = 0
            for part in parts:
                if ':' in part and shown < 10:  # Show first 10 fields
                    key, val = part.split(':', 1)
                    key = key.strip()
                    val = val.strip()
                    if val:
                        print(f"   {key}: {val}")
                        shown += 1
        else:
            preview = doc[:300]
            print(f"   {preview}...")
        
        if idx >= 20:
            print(f"\n... and {len(matches) - 20} more matches")
            break
    
    print("\n" + "="*60)

def search_by_parts():
    """Interactive search by parts of name"""
    print("\n🔍 Interactive Name Search")
    print("="*60)
    
    tests = [
        "אבי",
        "אתרוג",
        "תפוז",
        "שוורץ",
    ]
    
    print("\n📋 Testing common name parts:")
    for test in tests:
        qa_system = VectorDBQASystem(persist_directory="./chroma_db")
        all_data = qa_system.collection.get(include=["documents"])
        documents = all_data.get("documents", [])
        
        count = sum(1 for doc in documents if test in doc.lower())
        print(f"   '{test}': {count} matches")
    
    print("\n" + "="*60)
    
    while True:
        query = input("\n🔍 Search for substring (or 'quit'): ").strip()
        
        if not query or query.lower() in ['quit', 'exit', 'q']:
            break
        
        show_all_names_containing(query)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Command line search
        search_term = " ".join(sys.argv[1:])
        show_all_names_containing(search_term)
    else:
        # Interactive
        search_by_parts()
