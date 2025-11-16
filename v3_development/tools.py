"""LangChain Tools - Interface Segregation Principle"""
from langchain.tools import tool
from typing import Dict, List

from database import VectorDatabase
from search_strategies import KeywordSearch, SemanticSearch


class ChatbotTools:
    """Factory for creating LangChain tools - Dependency Inversion"""
    
    def __init__(self, db: VectorDatabase):
        self.db = db
        self.keyword_search = KeywordSearch(db)
        self.semantic_search = SemanticSearch(db)
    
    def create_tools(self) -> List:
        """Create all tools"""
        
        db = self.db
        keyword_search = self.keyword_search
        semantic_search = self.semantic_search
        
        @tool
        def count_documents() -> Dict:
            """Total contacts in database"""
            total = db.count()
            return {"total_contacts": total, "message": f"{total:,} contacts"}
        
        @tool
        def search_keyword(keyword: str, limit: int = 50) -> Dict:
            """Search for exact keyword (professions, companies, roles)"""
            result = keyword_search.search(keyword, limit)
            
            formatted = []
            for r in result.results:
                phone = r.phone or 'N/A'
                line = f"{r.name} | Phone: {phone}"
                if r.context:
                    line += f" | Found: {r.context[:100]}"
                formatted.append(line)
            
            return {
                "found": result.total_found,
                "keyword": keyword,
                "results": formatted
            }
        
        @tool
        def search_semantic(query: str, n_results: int = 5) -> Dict:
            """Semantic search for names, relationships, fuzzy matches"""
            result = semantic_search.search(query, n_results)
            
            formatted = []
            for r in result.results:
                formatted.append({
                    "name": r.name,
                    "phone": r.phone,
                    "score": round(r.similarity_score, 2)
                })
            
            return {"query": query, "results": formatted}
        
        @tool
        def list_by_prefix(letter: str) -> Dict:
            """Names starting with letter"""
            all_data = db.get_all(include=["metadatas"])
            names = []
            for md in all_data.get("metadatas", []):
                name = md.get("name", "").strip()
                if name and name[0].upper() == letter.upper():
                    names.append(name)
            return {"letter": letter, "count": len(names), "names": sorted(set(names))[:50]}
        
        @tool
        def count_by_language() -> Dict:
            """Count Hebrew vs English names"""
            all_data = db.get_all(include=["metadatas"])
            hebrew = english = other = 0
            
            for md in all_data.get("metadatas", []):
                name = md.get("name", "").strip()
                if not name:
                    other += 1
                    continue
                
                first_char = next((c for c in name if c.isalpha()), None)
                if not first_char:
                    other += 1
                elif '\u0590' <= first_char <= '\u05FF':
                    hebrew += 1
                elif first_char.isascii():
                    english += 1
                else:
                    other += 1
            
            return {"hebrew": hebrew, "english": english, "other": other, "total": hebrew + english + other}
        
        return [count_documents, search_keyword, search_semantic, list_by_prefix, count_by_language]
