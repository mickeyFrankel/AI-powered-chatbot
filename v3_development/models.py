"""
Data Models Module for VectorDB Q&A System

This module defines all data structures, types, and models used throughout
the application. Following the Single Responsibility Principle, this module
is solely responsible for data structure definitions.

Author: Mickey Frankel
Date: 2025-10-30
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Protocol
from enum import Enum
from pathlib import Path


class SearchMethod(Enum):
    """Enumeration of available search methods."""
    
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    FUZZY_TEXT = "fuzzy_text"
    HYBRID = "hybrid"


class QueryType(Enum):
    """Enumeration of query types for tool selection."""
    
    PERSON_NAME = "person_name"
    PROFESSION = "profession"
    COMPANY = "company"
    RELATIONSHIP = "relationship"
    ALPHABETICAL = "alphabetical"
    COUNTING = "counting"


@dataclass
class Document:
    """
    Represents a single document in the database.
    
    Attributes:
        text: Full text content of the document
        metadata: Key-value pairs of metadata
        source: Path to the source file
        row_id: Unique identifier within the source
    """
    
    text: str
    metadata: Dict[str, Any]
    source: str
    row_id: Any
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert document to dictionary format."""
        return {
            'text': self.text,
            'metadata': self.metadata,
            'source': self.source,
            'row_id': self.row_id
        }


@dataclass
class SearchResult:
    """
    Represents a single search result.
    
    Attributes:
        id: Document ID in the database
        document: Full document text
        metadata: Document metadata
        similarity_score: Similarity score (0-1)
        match_type: Type of match (semantic/keyword/fuzzy)
        context: Optional context snippet for keyword matches
    """
    
    id: str
    document: str
    metadata: Dict[str, Any]
    similarity_score: float
    match_type: str
    context: Optional[str] = None
    
    @property
    def name(self) -> str:
        """Extract name from metadata."""
        return self.metadata.get('name', 'Unknown')
    
    @property
    def phone(self) -> Optional[str]:
        """Extract phone number from metadata."""
        for key, value in self.metadata.items():
            if 'phone' in key.lower() and value:
                return str(value)
        return None


@dataclass
class SearchResponse:
    """
    Aggregated search response with multiple results.
    
    Attributes:
        query: Original query string
        results: List of search results
        search_method: Method used for search
        total_found: Total number of results found
        corrected_query: Optional corrected query if typos were fixed
    """
    
    query: str
    results: List[SearchResult]
    search_method: SearchMethod
    total_found: int
    corrected_query: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'query': self.query,
            'results': [
                {
                    'id': r.id,
                    'name': r.name,
                    'phone': r.phone,
                    'similarity_score': r.similarity_score,
                    'match_type': r.match_type,
                    'context': r.context
                }
                for r in self.results
            ],
            'search_method': self.search_method.value,
            'total_found': self.total_found,
            'corrected_query': self.corrected_query
        }


@dataclass
class NameFields:
    """
    Extracted name-related fields from a document.
    
    Attributes:
        name: Full extracted name
        first_char: First alphabetic character (for indexing)
        name_len: Length of the name
    """
    
    name: str
    first_char: str
    name_len: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'name': self.name,
            'first_char': self.first_char,
            'name_len': self.name_len
        }


@dataclass
class CollectionStats:
    """
    Statistics about the document collection.
    
    Attributes:
        total_documents: Total number of documents
        collection_name: Name of the collection
        embedding_model: Name of the embedding model used
        sources: Dictionary of source files and their document counts
    """
    
    total_documents: int
    collection_name: str
    embedding_model: str
    sources: Dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'total_documents': self.total_documents,
            'collection_name': self.collection_name,
            'embedding_model': self.embedding_model,
            'sources': self.sources
        }


@dataclass
class LanguageDistribution:
    """
    Distribution of names by language/script.
    
    Attributes:
        hebrew: Count of Hebrew names
        english: Count of English/Latin names
        other: Count of other/unknown script names
        total: Total count
    """
    
    hebrew: int
    english: int
    other: int
    
    @property
    def total(self) -> int:
        """Calculate total count."""
        return self.hebrew + self.english + self.other
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'hebrew': self.hebrew,
            'english': self.english,
            'other': self.other,
            'total': self.total
        }


@dataclass
class IngestionReport:
    """
    Report from document ingestion operation.
    
    Attributes:
        documents_added: Number of new documents added
        documents_skipped: Number of duplicate documents skipped
        file_name: Name of the ingested file
        total_in_db: Total documents in database after ingestion
        errors: List of errors encountered
    """
    
    documents_added: int
    documents_skipped: int
    file_name: str
    total_in_db: int
    errors: List[str] = field(default_factory=list)
    
    @property
    def success(self) -> bool:
        """Check if ingestion was successful."""
        return self.documents_added > 0 and len(self.errors) == 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'documents_added': self.documents_added,
            'documents_skipped': self.documents_skipped,
            'file_name': self.file_name,
            'total_in_db': self.total_in_db,
            'errors': self.errors,
            'success': self.success
        }


# Protocol definitions for dependency inversion


class FileLoader(Protocol):
    """Protocol for file loading operations."""
    
    def load(self, file_path: Path) -> List[Document]:
        """
        Load documents from a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            List of Document objects
        """
        ...


class SearchStrategy(Protocol):
    """Protocol for search strategy implementations."""
    
    def search(
        self,
        query: str,
        limit: int
    ) -> SearchResponse:
        """
        Execute search with given query.
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            SearchResponse object
        """
        ...


class EmbeddingModel(Protocol):
    """Protocol for embedding model operations."""
    
    def encode(
        self,
        texts: List[str],
        show_progress_bar: bool = False
    ) -> Any:
        """
        Generate embeddings for texts.
        
        Args:
            texts: List of text strings
            show_progress_bar: Whether to show progress
            
        Returns:
            Array of embeddings
        """
        ...


# Type aliases for clarity
Embedding = List[float]
EmbeddingList = List[Embedding]
Metadata = Dict[str, Any]
DocumentID = str
