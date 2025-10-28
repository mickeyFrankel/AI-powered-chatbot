#!/usr/bin/env python3
"""
ChromaDB-Free VectorDB Q&A System
Works with any Python version 3.8+ without ChromaDB dependency issues
"""

import os
import sys
import json
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Environment setup
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("⚠️  python-dotenv not installed, skipping .env file loading")

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Try to import SentenceTransformers
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
    print("✅ SentenceTransformers available")
except ImportError:
    print("⚠️  SentenceTransformers not available, using TF-IDF")
    HAS_SENTENCE_TRANSFORMERS = False


class SimplePersistentVectorDB:
    """Simple vector database using scikit-learn with persistence"""
    
    def __init__(self, persist_directory: str = "./simple_db"):
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(exist_ok=True)
        
        self.documents = []
        self.metadatas = []
        self.embeddings = None
        self.vectorizer = None
        self.embedding_model = None
        self.use_transformers = False
        
        # Try to load embedding model
        if HAS_SENTENCE_TRANSFORMERS:
            try:
                print("📦 Loading SentenceTransformers model...")
                self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
                self.use_transformers = True
                print("✅ SentenceTransformers model loaded")
            except Exception as e:
                print(f"⚠️  Failed to load SentenceTransformers: {e}")
                self.use_transformers = False
        
        if not self.use_transformers:
            print("📦 Using TF-IDF vectorizer")
            self.vectorizer = TfidfVectorizer(
                max_features=5000, 
                stop_words='english',
                ngram_range=(1, 2),
                min_df=1
            )
        
        # Load existing data
        self.load_from_disk()
    
    def add_documents(self, documents, metadatas):
        """Add documents to the database"""
        if not documents:
            return
            
        print(f"📝 Adding {len(documents)} documents...")
        
        self.documents.extend(documents)
        self.metadatas.extend(metadatas)
        
        # Generate embeddings
        if self.use_transformers:
            try:
                new_embeddings = self.embedding_model.encode(documents)
                if self.embeddings is None:
                    self.embeddings = new_embeddings
                else:
                    self.embeddings = np.vstack([self.embeddings, new_embeddings])
            except Exception as e:
                print(f"⚠️  SentenceTransformers encoding failed: {e}")
                self.use_transformers = False
                self._rebuild_tfidf()
        
        if not self.use_transformers:
            self._rebuild_tfidf()
        
        print(f"✅ Added {len(documents)} documents (total: {len(self.documents)})")
        self.save_to_disk()
    
    def _rebuild_tfidf(self):
        """Rebuild TF-IDF matrix for all documents"""
        if len(self.documents) > 0:
            self.embeddings = self.vectorizer.fit_transform(self.documents)
    
    def query(self, query_text, n_results=5):
        """Query the database"""
        if len(self.documents) == 0:
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}
        
        try:
            # Generate query embedding
            if self.use_transformers:
                query_embedding = self.embedding_model.encode([query_text])
                similarities = cosine_similarity(query_embedding, self.embeddings)[0]
            else:
                query_vector = self.vectorizer.transform([query_text])
                similarities = cosine_similarity(query_vector, self.embeddings)[0]
            
            # Get top results
            top_indices = np.argsort(similarities)[::-1][:n_results]
            
            results_docs = [self.documents[i] for i in top_indices]
            results_meta = [self.metadatas[i] for i in top_indices]
            results_distances = [1.0 - similarities[i] for i in top_indices]  # Convert to distance
            
            return {
                "documents": [results_docs],
                "metadatas": [results_meta], 
                "distances": [results_distances]
            }
        except Exception as e:
            print(f"❌ Query failed: {e}")
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}
    
    def count(self):
        """Get document count"""
        return len(self.documents)
    
    def save_to_disk(self):
        """Save database to disk"""
        try:
            data = {
                "documents": self.documents,
                "metadatas": self.metadatas,
                "use_transformers": self.use_transformers
            }
            
            # Save embeddings
            if self.use_transformers and self.embeddings is not None:
                np.save(self.persist_directory / "embeddings.npy", self.embeddings)
            elif not self.use_transformers and self.embeddings is not None:
                # Save TF-IDF components
                with open(self.persist_directory / "vectorizer.pkl", "wb") as f:
                    pickle.dump(self.vectorizer, f)
                with open(self.persist_directory / "tfidf_matrix.pkl", "wb") as f:
                    pickle.dump(self.embeddings, f)
            
            # Save documents and metadata
            with open(self.persist_directory / "data.json", "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
        except Exception as e:
            print(f"⚠️  Failed to save to disk: {e}")
    
    def load_from_disk(self):
        """Load database from disk"""
        try:
            data_file = self.persist_directory / "data.json"
            if not data_file.exists():
                return
                
            with open(data_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            self.documents = data.get("documents", [])
            self.metadatas = data.get("metadatas", [])
            use_transformers_saved = data.get("use_transformers", False)
            
            if len(self.documents) > 0:
                if self.use_transformers and use_transformers_saved:
                    embeddings_file = self.persist_directory / "embeddings.npy"
                    if embeddings_file.exists():
                        self.embeddings = np.load(embeddings_file)
                        print(f"✅ Loaded {len(self.documents)} documents from disk (Transformers)")
                    else:
                        # Regenerate embeddings
                        print("🔄 Regenerating embeddings...")
                        self.embeddings = self.embedding_model.encode(self.documents)
                else:
                    # Load TF-IDF components
                    vectorizer_file = self.persist_directory / "vectorizer.pkl"
                    matrix_file = self.persist_directory / "tfidf_matrix.pkl"
                    
                    if vectorizer_file.exists() and matrix_file.exists():
                        with open(vectorizer_file, "rb") as f:
                            self.vectorizer = pickle.load(f)
                        with open(matrix_file, "rb") as f:
                            self.embeddings = pickle.load(f)
                        print(f"✅ Loaded {len(self.documents)} documents from disk (TF-IDF)")
                    else:
                        # Regenerate TF-IDF
                        print("🔄 Regenerating TF-IDF matrix...")
                        self._rebuild_tfidf()
                
        except Exception as e:
            print(f"⚠️  Failed to load from disk: {e}")


class NoChromaVectorDBQASystem:
    """VectorDB Q&A System without ChromaDB dependency"""
    
    def __init__(self, persist_directory: str = "./simple_db"):
        print("🔧 Initializing ChromaDB-free VectorDB system...")
        
        self.db = SimplePersistentVectorDB(persist_directory)
        print("✅ VectorDB system initialized successfully")
    
    def load_csv(self, file_path: str):
        """Load CSV file and add to database"""
        print(f"📁 Loading CSV file: {file_path}")
        
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
            print(f"✅ CSV loaded with {len(df)} rows, {len(df.columns)} columns")
            print(f"   Columns: {', '.join(df.columns)}")
        except Exception as e:
            print(f"❌ Failed to load CSV: {e}")
            return
        
        documents = []
        metadatas = []
        
        for idx, row in df.iterrows():
            # Combine all columns into text
            text_parts = []
            metadata = {"source": file_path, "row_id": idx}
            
            for col in df.columns:
                if pd.notna(row[col]):
                    text_parts.append(f"{col}: {row[col]}")
                    metadata[col] = str(row[col])
            
            text = " | ".join(text_parts)
            documents.append(text)
            metadatas.append(metadata)
        
        # Add to database
        self.db.add_documents(documents, metadatas)
    
    def search(self, query: str, n_results: int = 5):
        """Search the database"""
        print(f"🔍 Searching for: '{query}'")
        
        results = self.db.query(query, n_results)
        
        search_results = {
            "query": query,
            "results": []
        }
        
        if results["documents"] and results["documents"][0]:
            for i, (doc, metadata, distance) in enumerate(zip(
                results["documents"][0],
                results["metadatas"][0], 
                results["distances"][0]
            )):
                search_results["results"].append({
                    "document": doc,
                    "metadata": metadata,
                    "similarity_score": 1 - distance,  # Convert distance to similarity
                    "rank": i + 1
                })
        
        print(f"✅ Found {len(search_results['results'])} results")
        return search_results
    
    def interactive_qa(self):
        """Interactive Q&A session"""
        print("\n" + "="*60)
        print("🎮 Interactive Q&A Session Started")
        print("   ChromaDB-Free Vector Database System")
        print("   Type 'quit', 'exit', or 'q' to stop")
        print("="*60)
        
        while True:
            try:
                query = input("\n❓ Your question: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not query:
                    continue
                
                results = self.search(query, n_results=3)
                
                print(f"\n📋 Results for: '{query}'")
                print("-" * 50)
                
                if results["results"]:
                    for i, result in enumerate(results["results"], 1):
                        print(f"\n🔸 Result {i} (Score: {result['similarity_score']:.3f})")
                        print(f"   {result['document'][:300]}...")
                        if len(result['document']) > 300:
                            print("   [truncated]")
                else:
                    print("❌ No results found")
                    print("💡 Try different keywords or check if data is loaded")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")


def main():
    """Main function"""
    print("🚀 ChromaDB-Free VectorDB Q&A System")
    print("   Compatible with Python 3.8+ without ChromaDB issues")
    print("=" * 60)
    
    try:
        # Initialize system
        qa_system = NoChromaVectorDBQASystem()
        
        # Check for sample data
        project_dir = Path(__file__).parent
        sample_file = project_dir / "sample_multilingual_data.csv"
        
        if sample_file.exists():
            print(f"\n📁 Found sample data: {sample_file}")
            qa_system.load_csv(str(sample_file))
            
            # Quick test
            print("\n🧪 Quick test search...")
            results = qa_system.search("machine learning", n_results=2)
            if results["results"]:
                print("✅ System is working! Sample results:")
                for result in results["results"]:
                    print(f"   - Score: {result['similarity_score']:.3f}")
                    print(f"     {result['document'][:100]}...")
            else:
                print("⚠️  No results found in test")
        else:
            print(f"⚠️  Sample data not found at {sample_file}")
            print("   You can load your own CSV files manually")
        
        # Start interactive session
        qa_system.interactive_qa()
        
    except Exception as e:
        print(f"❌ Failed to start system: {e}")
        import traceback
        traceback.print_exc()
        print("\n💡 Make sure you have these packages installed:")
        print("   pip install pandas numpy scikit-learn sentence-transformers python-dotenv")


if __name__ == "__main__":
    main()
