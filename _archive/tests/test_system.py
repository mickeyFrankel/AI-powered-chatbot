#!/usr/bin/env python3
"""
Test script to verify the chatbot system is working correctly
"""

import sys
import os
from pathlib import Path

def test_imports():
    """Test if all required packages can be imported"""
    print("🧪 Testing imports...")
    
    tests = [
        ("pandas", "pandas"),
        ("numpy", "numpy"), 
        ("chromadb", "chromadb"),
        ("sentence_transformers", "sentence_transformers"),
        ("sklearn", "scikit-learn"),
        ("dotenv", "python-dotenv"),
    ]
    
    failed = []
    
    for module, package in tests:
        try:
            __import__(module)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            failed.append(package)
    
    if failed:
        print(f"\n❌ Missing packages: {', '.join(failed)}")
        print("Run: python install_dependencies.py")
        return False
    else:
        print("✅ All core packages imported successfully!")
        return True

def test_sample_data():
    """Test if sample data exists and is readable"""
    print("\n🧪 Testing sample data...")
    
    project_dir = Path(__file__).parent
    sample_file = project_dir / "sample_multilingual_data.csv"
    
    if sample_file.exists():
        try:
            import pandas as pd
            df = pd.read_csv(sample_file)
            print(f"✅ Sample data loaded: {len(df)} rows, {len(df.columns)} columns")
            print(f"   Columns: {', '.join(df.columns)}")
            return True
        except Exception as e:
            print(f"❌ Failed to read sample data: {e}")
            return False
    else:
        print(f"❌ Sample data not found at {sample_file}")
        return False

def test_chromadb():
    """Test ChromaDB functionality"""
    print("\n🧪 Testing ChromaDB...")
    
    try:
        import chromadb
        from chromadb.config import Settings
        
        # Create temporary test database
        test_dir = Path(__file__).parent / "test_chroma"
        test_dir.mkdir(exist_ok=True)
        
        client = chromadb.PersistentClient(
            path=str(test_dir),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Create test collection
        collection = client.create_collection("test_collection")
        
        # Add test document
        collection.add(
            documents=["This is a test document"],
            metadatas=[{"source": "test"}],
            ids=["test_1"]
        )
        
        # Query test
        results = collection.query(
            query_texts=["test"],
            n_results=1
        )
        
        if results["documents"] and results["documents"][0]:
            print("✅ ChromaDB is working correctly")
            
            # Cleanup
            import shutil
            shutil.rmtree(test_dir, ignore_errors=True)
            return True
        else:
            print("❌ ChromaDB query returned no results")
            return False
            
    except Exception as e:
        print(f"❌ ChromaDB test failed: {e}")
        return False

def test_sentence_transformers():
    """Test SentenceTransformers functionality"""
    print("\n🧪 Testing SentenceTransformers...")
    
    try:
        from sentence_transformers import SentenceTransformer
        
        # Try to load a small model
        print("   Loading embedding model...")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Test encoding
        embeddings = model.encode(["Hello world", "Test sentence"])
        
        if embeddings is not None and len(embeddings) == 2:
            print(f"✅ SentenceTransformers working: {embeddings.shape}")
            return True
        else:
            print("❌ SentenceTransformers encoding failed")
            return False
            
    except Exception as e:
        print(f"❌ SentenceTransformers test failed: {e}")
        return False

def test_minimal_system():
    """Test the minimal chatbot system"""
    print("\n🧪 Testing minimal chatbot system...")
    
    try:
        # Import our system
        sys.path.insert(0, str(Path(__file__).parent))
        from minimal_chatbot import MinimalVectorDBQASystem
        
        # Create temporary test directory
        test_dir = Path(__file__).parent / "test_minimal_db"
        
        # Initialize system
        qa_system = MinimalVectorDBQASystem(persist_directory=str(test_dir))
        
        # Test with sample data if available
        sample_file = Path(__file__).parent / "sample_multilingual_data.csv"
        if sample_file.exists():
            qa_system.load_csv(str(sample_file))
            
            # Test search
            results = qa_system.search("machine learning", n_results=2)
            
            if results["results"]:
                print(f"✅ Minimal system working: found {len(results['results'])} results")
                print(f"   Top result: {results['results'][0]['document'][:50]}...")
                
                # Cleanup
                import shutil
                shutil.rmtree(test_dir, ignore_errors=True)
                return True
            else:
                print("❌ Minimal system search returned no results")
                return False
        else:
            print("✅ Minimal system initialized (no sample data to test)")
            return True
            
    except Exception as e:
        print(f"❌ Minimal system test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 VectorDB Chatbot System - Test Suite")
    print("=" * 50)
    
    tests = [
        ("Import Tests", test_imports),
        ("Sample Data", test_sample_data),
        ("ChromaDB", test_chromadb),
        ("SentenceTransformers", test_sentence_transformers),
        ("Minimal System", test_minimal_system),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "="*50)
    print("📊 TEST SUMMARY")
    print("="*50)
    
    passed = 0
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = "✅ PASS" if passed_test else "❌ FAIL"
        print(f"{status:<8} {test_name}")
        if passed_test:
            passed += 1
    
    print(f"\nResult: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your system is ready to use.")
        print("\nTo start the chatbot:")
        print("  python minimal_chatbot.py")
    else:
        print(f"\n⚠️  {total - passed} tests failed. Please fix the issues above.")
        print("\nTo install missing dependencies:")
        print("  python install_dependencies.py")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
