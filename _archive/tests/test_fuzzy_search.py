#!/usr/bin/env python3
"""
Test script for fuzzy search / typo correction functionality
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from vectoric_search import VectorDBQASystem

def test_fuzzy_search():
    """Test fuzzy matching with various typos"""
    
    print("🧪 Testing Fuzzy Search & Typo Correction")
    print("=" * 60)
    
    # Initialize system
    qa_system = VectorDBQASystem(persist_directory="./test_fuzzy_db")
    
    # Load sample data
    sample_file = Path(__file__).parent / "sample_multilingual_data.csv"
    
    if not sample_file.exists():
        print("❌ Sample data not found. Creating it...")
        from vectoric_search import create_sample_multilingual_csv
        sample_file = Path(create_sample_multilingual_csv())
    
    print(f"\n📁 Loading: {sample_file}")
    documents = qa_system.load_file(str(sample_file))
    qa_system.add_documents(documents)
    
    # Test cases with typos
    test_queries = [
        ("machne learning", "machine learning"),
        ("deap lerning", "deep learning"),
        ("naturel languge", "natural language"),
        ("compter vision", "computer vision"),
        ("Gugle search", "Google search"),
        ("neurla network", "neural network"),
        ("artficial inteligence", "artificial intelligence"),
        ("data sience", "data science"),
        ("algrithm", "algorithm"),
    ]
    
    print("\n" + "=" * 60)
    print("🎯 FUZZY MATCHING TEST RESULTS")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for typo, expected in test_queries:
        print(f"\n📝 Testing: '{typo}'")
        print(f"   Expected: '{expected}'")
        
        try:
            results = qa_system.search(typo, n_results=1, auto_correct=True)
            
            if results.get('corrected'):
                corrections_str = ', '.join([f'{old}→{new}' for old, new in results['corrected']])
                print(f"   ✅ Corrected: {corrections_str}")
                
                # Check if any expected words were corrected
                found_corrections = [new for old, new in results['corrected']]
                if any(word in expected.split() for word in found_corrections):
                    passed += 1
                    print(f"   ✓ PASS")
                else:
                    failed += 1
                    print(f"   ✗ FAIL: Correction doesn't match expected")
            else:
                print(f"   ⚠️  No corrections made")
                # Check if query actually needed correction
                if typo.lower() != expected.lower():
                    failed += 1
                    print(f"   ✗ FAIL: Should have corrected")
                else:
                    passed += 1
                    print(f"   ✓ PASS: No correction needed")
            
            # Show search results
            if results['results']:
                top_result = results['results'][0]
                print(f"   📄 Top result (score: {top_result['similarity_score']:.2f}):")
                print(f"      {top_result['document'][:80]}...")
            else:
                print(f"   📄 No results found")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
            failed += 1
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    print(f"✅ Passed: {passed}/{len(test_queries)}")
    print(f"❌ Failed: {failed}/{len(test_queries)}")
    print(f"📈 Success rate: {(passed/len(test_queries)*100):.1f}%")
    
    # Cleanup
    print("\n🧹 Cleaning up test database...")
    import shutil
    shutil.rmtree("./test_fuzzy_db", ignore_errors=True)
    
    return passed == len(test_queries)

if __name__ == "__main__":
    try:
        success = test_fuzzy_search()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n👋 Test interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
