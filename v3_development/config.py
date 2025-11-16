"""
Configuration Module for VectorDB Q&A System

This module contains all configuration constants, settings, and environment variables
used throughout the application. Following the Single Responsibility Principle,
this module is solely responsible for configuration management.

Author: Mickey Frankel
Date: 2025-10-30
"""

import os
from pathlib import Path
from typing import Final
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Suppress third-party logging
os.environ['ANONYMIZED_TELEMETRY'] = 'False'


class DatabaseConfig:
    """Database configuration settings."""
    
    PERSIST_DIRECTORY: Final[str] = "./chroma_db"
    COLLECTION_NAME: Final[str] = "documents"
    EMBEDDING_MODEL: Final[str] = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    DISTANCE_METRIC: Final[str] = "cosine"


class SearchConfig:
    """Search configuration settings."""
    
    DEFAULT_SEARCH_LIMIT: Final[int] = 5
    DEFAULT_KEYWORD_LIMIT: Final[int] = 50
    MAX_KEYWORD_LIMIT: Final[int] = 200
    FUZZY_THRESHOLD: Final[float] = 80.0
    TEXT_MATCH_THRESHOLD: Final[int] = 60
    CONTEXT_WINDOW_SIZE: Final[int] = 100


class DataProcessingConfig:
    """Data processing configuration."""
    
    # DataFrame preprocessing thresholds
    SPARSE_COLUMN_THRESHOLD: Final[float] = 0.95
    EMPTY_ROW_THRESHOLD: Final[float] = 0.9
    
    # Phone number processing
    ISRAELI_PHONE_DIGITS: Final[tuple] = (9, 10)
    
    # Priority fields for extraction
    NAME_PRIORITY_FIELDS: Final[list] = [
        'First Name', 'Last Name', 'Organization Name',
        'name', 'title', 'industry_name'
    ]
    
    IMPORTANT_FIELD_PATTERNS: Final[list] = [
        'phone', 'mobile', 'tel', 'email', 'mail', 'address', 'addr'
    ]


class AgentConfig:
    """Agent configuration settings."""
    
    LLM_MODEL: Final[str] = "gpt-4"
    LLM_TEMPERATURE: Final[float] = 0.0
    MAX_HISTORY_TURNS: Final[int] = 5
    MAX_ITERATIONS: Final[int] = 3
    VERBOSE: Final[bool] = False


class ProfessionSynonyms:
    """Hebrew profession synonyms for keyword search."""
    
    SYNONYMS: Final[dict] = {
        'עורך דין': ['עורך דין', 'עו"ד', 'עוד', 'lawyer', 'attorney'],
        'עו"ד': ['עורך דין', 'עו"ד', 'עוד', 'lawyer', 'attorney'],
        'רופא': ['רופא', 'ד"ר', 'דר', 'doctor', 'dr'],
        'דוקטור': ['רופא', 'ד"ר', 'דר', 'doctor', 'dr', 'דוקטור'],
    }


class HebrewRelationships:
    """Hebrew family relationship translations."""
    
    RELATIONSHIPS: Final[dict] = {
        'חמותי': 'mother-in-law',
        'חמי': 'father-in-law',
        'גיסה': 'sister-in-law',
        'גיס': 'brother-in-law',
        'כלה': 'daughter-in-law',
        'חתן': 'son-in-law',
        'דודה': 'aunt',
        'דוד': 'uncle',
        'אחיין': 'nephew',
        'אחיינית': 'niece',
        'סבתא': 'grandmother',
        'סבא': 'grandfather',
        'נכד': 'grandson',
        'נכדה': 'granddaughter',
    }


class FileExtensions:
    """Supported file extensions."""
    
    CSV: Final[str] = '.csv'
    EXCEL: Final[tuple] = ('.xlsx', '.xls')
    DOCX: Final[str] = '.docx'
    PDF: Final[str] = '.pdf'
    TXT: Final[str] = '.txt'
    
    @classmethod
    def all_supported(cls) -> set:
        """Return all supported extensions."""
        return {cls.CSV, cls.DOCX, cls.PDF, cls.TXT, *cls.EXCEL}


class APIKeys:
    """API key management."""
    
    @staticmethod
    def get_openai_key() -> str:
        """
        Get OpenAI API key from environment.
        
        Returns:
            str: API key
            
        Raises:
            RuntimeError: If API key is not found
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is missing. "
                "Set it in your environment or .env file."
            )
        return api_key


class UIMessages:
    """User interface messages in Hebrew and English."""
    
    # Exit messages
    EXIT_PHRASES: Final[list] = [
        'quit', 'exit', 'q', 'bye', 'goodbye',
        'חלאס', 'סיום', 'להתראות', 'יציאה', 'סגור', 'ביי',
        'תודה ולהתראות', 'די', 'די תודה', 'זהו'
    ]
    
    GOODBYE_MESSAGE: Final[str] = "להתראות! Goodbye!"
    
    # Database messages
    NO_DOCUMENTS: Final[str] = (
        "אין מסמכים במאגר. טען קובץ תחילה. / "
        "No documents in database. Load a file first."
    )
    
    HISTORY_CLEARED: Final[str] = (
        "Conversation history cleared. Starting fresh!"
    )


# Logging configuration
import logging

def configure_logging(level: int = logging.ERROR) -> None:
    """
    Configure logging for the application.
    
    Args:
        level: Logging level (default: ERROR)
    """
    logging.getLogger('chromadb').setLevel(level)
    logging.getLogger('chromadb.telemetry').setLevel(logging.CRITICAL)
    
    # Suppress ChromaDB telemetry
    try:
        import chromadb.telemetry.posthog as posthog
        posthog.Posthog.capture = lambda *args, **kwargs: None
    except ImportError:
        pass


# Initialize logging on import
configure_logging()
