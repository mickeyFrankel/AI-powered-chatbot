#!/usr/bin/env python3
"""
Simple chatbot with substring search - No OpenAI API key required
FIXED VERSION with proper partial name matching
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from vectoric_search import VectorDBQASystem

# Add substring search method
def search_by_substring(self, substring: str, n_results: int = 50):
    """Search for documents containing exact substring"""
    print(f"\n🔍 Substring search for: '{substring}'")
    
    all_docs = self.collection.get(include=["documents", "metadatas", "ids"])
    documents = all_docs.get("documents", [])
    metadatas = all_docs.get("metadatas", [])
    ids = all_docs.get("ids", [])
    
    substring_lower = substring.lower()
    matches = []
    
    for doc, metadata, doc_id in zip(documents, metadatas, ids):
        if substring_lower in doc.lower():
            matches.append({
                'id': doc_id,
                'document': doc,
                'metadata': metadata,
                'similarity_score': 1.0
            })
    
    matches = matches[:n_results]
    print(f"✅ Found {len(matches)} contacts containing '{substring}'")
    
    return {
        'query': substring,
        'results': matches
    }

VectorDBQASystem.search_by_substring = search_by_substring

# Enhanced interactive QA
def enhanced_interactive_qa(self):
    """Interactive QA with substring search capability"""
    print("\n" + "="*60)
    print("🤖 SIMPLE VECTOR DATABASE Q&A")
    print("   ✅ No API key needed")
    print("   ✅ Substring search enabled")
    print("   ✅ Searches all 1917 contacts")
    print("="*60)
    
    stats = self.get_collection_stats()
    print(f"📊 Documents in database: {stats['document_count']}")
    print(f"🧠 Embedding model: {stats['embedding_model']}")
    
    print("\n📝 Available commands:")
    print("   'load <file_path>' - Load a new file")
    print("   'stats' - Show database statistics")
    print("   'find <name>' - Find all contacts with name")
    print("   'quit' or 'exit' - Exit the system")
    print("\n💡 Tip: For names, use 'find ברק' to search all contacts with that name")
    
    while True:
        try:
            user_input = input("\n🔍 Enter your query or command: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            elif user_input.lower() == 'stats':
                s = self.get_collection_stats()
                print(f"\n📊 Database Statistics:")
                print(f"   Documents: {s['document_count']}")
                print(f"   Model: {s['embedding_model']}")
            
            elif user_input.lower().startswith('find '):
                # Substring search
                search_term = user_input[5:].strip()
                results = self.search_by_substring(search_term, n_results=20)
                
                if not results['results']:
                    print(f"❌ No contacts found containing '{search_term}'")
                    print("💡 Try:")
                    print(f"   - Just first name: 'find {search_term.split()[0] if search_term.split() else search_term}'")
                    print("   - Check spelling")
                    print("   - Try English/Hebrew alternative")
                else:
                    print(f"\n🎯 Found {len(results['results'])} contacts with '{search_term}':")
                    print("-" * 60)
                    for i, r in enumerate(results['results'][:10], 1):
                        doc = r['document']
                        print(f"\n{i}. {doc[:150]}{'...' if len(doc) > 150 else ''}")
                    
                    if len(results['results']) > 10:
                        print(f"\n... and {len(results['results']) - 10} more results")
            
            elif user_input.lower().startswith('load '):
                path = user_input[5:].strip()
                try:
                    docs = self.load_file(path)
                    self.add_documents(docs)
                except Exception as e:
                    print(f"❌ Error loading file: {e}")
            
            else:
                # Try both semantic and substring search
                print("🔍 Trying semantic search...")
                sem_results = self.search(user_input, n_results=3)
                
                # Also try substring for any words in query
                words = user_input.split()
                sub_results = []
                for word in words:
                    if len(word) > 2:  # Skip short words
                        sr = self.search_by_substring(word, n_results=5)
                        if sr['results']:
                            sub_results.extend(sr['results'])
                
                # Show best results
                if sem_results['results'] or sub_results:
                    print(f"\n🎯 Results:")
                    print("-" * 60)
                    
                    # Show semantic results first
                    if sem_results['results']:
                        print("\n📊 Semantic matches:")
                        for i, r in enumerate(sem_results['results'], 1):
                            print(f"\n{i}. (Score: {r['similarity_score']:.2f})")
                            print(f"   {r['document'][:150]}...")
                    
                    # Show substring matches
                    if sub_results:
                        print("\n🔍 Exact text matches:")
                        seen = set()
                        count = 0
                        for r in sub_results:
                            doc_id = r.get('id', r.get('document'))
                            if doc_id not in seen:
                                seen.add(doc_id)
                                count += 1
                                print(f"\n{count}. {r['document'][:150]}...")
                                if count >= 5:
                                    break
                else:
                    print("❌ No results found.")
                    print("💡 Try: 'find <name>' for exact name matching")
        
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

VectorDBQASystem.interactive_qa = enhanced_interactive_qa

def main():
    print("🚀 Simple VectorDB Q&A - FIXED VERSION")
    print("   No OpenAI API key required")
    print("=" * 60)
    
    qa_system = VectorDBQASystem(persist_directory="./chroma_db")
    qa_system.interactive_qa()

if __name__ == "__main__":
    main()
