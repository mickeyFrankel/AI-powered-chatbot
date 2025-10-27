#!/usr/bin/env python3
"""
AI-First Unified Chatbot
Single intelligent agent system with GPT-4 function calling
"""

import os
import sys
from pathlib import Path

# Suppress warnings
os.environ['ANONYMIZED_TELEMETRY'] = 'False'
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))

from vectoric_search import AdvancedVectorDBQASystem

def main():
    """Main entry point"""
    
    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("\nERROR: OPENAI_API_KEY not found")
        print("\nSetup:")
        print("  1. Create .env file")
        print("  2. Add: OPENAI_API_KEY=your-key")
        print("  3. Get key at: https://platform.openai.com/api-keys\n")
        return 1
    
    try:
        # Initialize
        print("Loading AI agent...")
        qa_system = AdvancedVectorDBQASystem(persist_directory="./chroma_db")
        
        # Show stats
        stats = qa_system.get_collection_stats()
        print(f"Ready. Database: {stats['document_count']:,} documents\n")
        
        # Start
        qa_system.interactive_qa()
        return 0
        
    except KeyboardInterrupt:
        print("\nGoodbye!")
        return 0
        
    except Exception as e:
        print(f"\nError: {e}")
        print("Run: pip install -r requirements.txt\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())
