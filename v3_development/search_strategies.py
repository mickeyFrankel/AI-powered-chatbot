"""Search Strategies - Strategy Pattern + Open/Closed Principle"""
from abc import ABC, abstractmethod
from typing import List
import re
from rapidfuzz import fuzz

from models import SearchResult, SearchResponse, SearchMethod
from database import VectorDatabase
from config import SearchConfig, ProfessionSynonyms


class SearchStrategy(ABC):
    """Abstract search strategy - Open/Closed Principle"""
    
    @abstractmethod
    def search(self, query: str, limit: int) -> SearchResponse:
        pass


class SemanticSearch(SearchStrategy):
    """Semantic vector search"""
    
    def __init__(self, db: VectorDatabase):
        self.db = db
    
    def search(self, query: str, limit: int) -> SearchResponse:
        embedding = self.db.embedding_model.encode([query])[0].tolist()
        results = self.db.query(embedding, limit)
        
        search_results = []
        if results['documents'] and results['documents'][0]:
            for i in range(len(results['documents'][0])):
                search_results.append(SearchResult(
                    id=results['ids'][0][i],
                    document=results['documents'][0][i],
                    metadata=results['metadatas'][0][i],
                    similarity_score=1 - results['distances'][0][i],
                    match_type='semantic'
                ))
        
        return SearchResponse(
            query=query,
            results=search_results,
            search_method=SearchMethod.SEMANTIC,
            total_found=len(search_results)
        )


class KeywordSearch(SearchStrategy):
    """Exact keyword matching with synonyms"""
    
    def __init__(self, db: VectorDatabase):
        self.db = db
    
    def search(self, query: str, limit: int) -> SearchResponse:
        # Check synonyms
        search_terms = ProfessionSynonyms.SYNONYMS.get(query.strip(), [query])
        
        all_data = self.db.get_all()
        documents = all_data.get("documents", [])
        metadatas = all_data.get("metadatas", [])
        ids = all_data.get("ids", [])
        
        matches = []
        seen_names = set()
        
        for term in search_terms:
            term_upper = self._normalize(term).upper()
            
            for doc, meta, doc_id in zip(documents, metadatas, ids):
                name = meta.get('name', 'Unknown')
                if name in seen_names:
                    continue
                
                if term_upper in self._normalize(doc).upper():
                    seen_names.add(name)
                    context = self._extract_context(doc, term)
                    
                    matches.append(SearchResult(
                        id=doc_id,
                        document=doc,
                        metadata=meta,
                        similarity_score=1.0,
                        match_type='keyword',
                        context=context
                    ))
                    
                    if len(matches) >= limit:
                        return self._build_response(query, matches)
        
        return self._build_response(query, matches)
    
    def _normalize(self, s: str) -> str:
        """Remove diacritics and normalize"""
        import unicodedata
        s = unicodedata.normalize("NFKC", s)
        s = re.sub(r'[\u0591-\u05C7]', '', s)
        return s.strip()
    
    def _extract_context(self, text: str, keyword: str, window: int = 100) -> str:
        """Extract snippet around keyword"""
        pos = text.upper().find(keyword.upper())
        if pos == -1:
            return text[:200]
        
        start = max(0, pos - window)
        end = min(len(text), pos + len(keyword) + window)
        snippet = text[start:end].strip()
        
        if start > 0:
            snippet = "..." + snippet
        if end < len(text):
            snippet += "..."
        
        return snippet
    
    def _build_response(self, query: str, matches: List[SearchResult]) -> SearchResponse:
        return SearchResponse(
            query=query,
            results=matches,
            search_method=SearchMethod.KEYWORD,
            total_found=len(matches)
        )


class FuzzySearch(SearchStrategy):
    """Fuzzy text matching"""
    
    def __init__(self, db: VectorDatabase):
        self.db = db
    
    def search(self, query: str, limit: int, threshold: int = 60) -> SearchResponse:
        all_data = self.db.get_all()
        documents = all_data.get("documents", [])
        metadatas = all_data.get("metadatas", [])
        ids = all_data.get("ids", [])
        
        matches = []
        for doc, meta, doc_id in zip(documents, metadatas, ids):
            score = fuzz.partial_ratio(query.lower(), doc.lower())
            if score >= threshold:
                matches.append(SearchResult(
                    id=doc_id,
                    document=doc,
                    metadata=meta,
                    similarity_score=score / 100.0,
                    match_type='fuzzy'
                ))
        
        matches.sort(key=lambda x: x.similarity_score, reverse=True)
        
        return SearchResponse(
            query=query,
            results=matches[:limit],
            search_method=SearchMethod.FUZZY_TEXT,
            total_found=len(matches)
        )


class HybridSearch(SearchStrategy):
    """Try keyword first, fallback to semantic - Strategy pattern"""
    
    def __init__(self, db: VectorDatabase):
        self.keyword = KeywordSearch(db)
        self.semantic = SemanticSearch(db)
    
    def search(self, query: str, limit: int) -> SearchResponse:
        # Try keyword first
        result = self.keyword.search(query, limit)
        if result.total_found > 0:
            return result
        
        # Fallback to semantic
        return self.semantic.search(query, limit)
