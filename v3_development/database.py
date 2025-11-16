"""Database Operations - Single Responsibility"""
from pathlib import Path
from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from collections import Counter

from models import Document, NameFields, CollectionStats, IngestionReport
from config import DatabaseConfig


class VectorDatabase:
    """Handles all ChromaDB operations - Single Responsibility Principle"""
    
    def __init__(self):
        self.client = chromadb.PersistentClient(
            path=DatabaseConfig.PERSIST_DIRECTORY,
            settings=Settings(anonymized_telemetry=False, allow_reset=True)
        )
        self.embedding_model = SentenceTransformer(DatabaseConfig.EMBEDDING_MODEL)
        
        try:
            self.collection = self.client.get_collection(DatabaseConfig.COLLECTION_NAME)
        except:
            self.collection = self.client.create_collection(
                name=DatabaseConfig.COLLECTION_NAME,
                metadata={"hnsw:space": DatabaseConfig.DISTANCE_METRIC}
            )
    
    def add_documents(self, documents: List[Document]) -> IngestionReport:
        """Add documents with deduplication"""
        if not documents:
            return IngestionReport(0, 0, "empty", self.count())
        
        # Generate IDs
        abs_paths = [str(Path(d.source).resolve()) for d in documents]
        ids = [f"{p.lower()}::{d.row_id}" for p, d in zip(abs_paths, documents)]
        
        # Check existing
        existing = set(self.collection.get(ids=ids).get('ids', []) or [])
        new_idx = [i for i, _id in enumerate(ids) if _id not in existing]
        
        if not new_idx:
            return IngestionReport(0, len(ids), Path(documents[0].source).name, self.count())
        
        # Add new only
        new_docs = [documents[i] for i in new_idx]
        texts = [d.text for d in new_docs]
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True).tolist()
        
        metadatas = []
        for doc in new_docs:
            md = doc.metadata.copy()
            md.update({
                'source': str(Path(doc.source).resolve()),
                'source_name': Path(doc.source).name,
                'source_key': str(Path(doc.source).resolve()).lower(),
                'row_id': doc.row_id,
            })
            metadatas.append(md)
        
        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=[ids[i] for i in new_idx]
        )
        
        return IngestionReport(
            len(new_idx), len(existing), 
            Path(documents[0].source).name, 
            self.count()
        )
    
    def query(self, embedding: List[float], n_results: int = 5) -> Dict:
        """Semantic search"""
        return self.collection.query(
            query_embeddings=[embedding],
            n_results=n_results
        )
    
    def get_all(self, include: List[str] = None) -> Dict:
        """Get all documents"""
        return self.collection.get(include=include or ["documents", "metadatas"])
    
    def count(self) -> int:
        """Total documents"""
        return self.collection.count()
    
    def reset(self):
        """Delete and recreate collection"""
        self.client.delete_collection(DatabaseConfig.COLLECTION_NAME)
        self.collection = self.client.create_collection(
            name=DatabaseConfig.COLLECTION_NAME,
            metadata={"hnsw:space": DatabaseConfig.DISTANCE_METRIC}
        )
    
    def get_stats(self) -> CollectionStats:
        """Get statistics"""
        data = self.get_all(include=["metadatas"])
        metas = data.get("metadatas", []) or []
        
        sources = Counter()
        for md in metas:
            name = md.get('source_name', 'UNKNOWN')
            sources[name] += 1
        
        return CollectionStats(
            total_documents=self.count(),
            collection_name=DatabaseConfig.COLLECTION_NAME,
            embedding_model=DatabaseConfig.EMBEDDING_MODEL,
            sources=dict(sources)
        )
