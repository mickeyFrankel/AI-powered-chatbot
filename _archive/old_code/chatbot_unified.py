#!/usr/bin/env python3
"""
AI-First Unified Chatbot
Single intelligent agent - no routing layer needed

The agent has 7 tools including search(), and intelligently
decides which tool to use for each query. No manual routing!

Usage:
    python chatbot_unified.py
"""

import os
import sys
from pathlib import Path

# Suppress warnings
os.environ['ANONYMIZED_TELEMETRY'] = 'False'
import warnings
warnings.filterwarnings('ignore', message='.*telemetry.*')
warnings.filterwarnings('ignore', message='.*capture.*')

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from vectoric_search import AdvancedVectorDBQASystem

def print_banner():
    """Print startup banner"""
    print("\n" + "=" * 70)
    print("🤖 AI-FIRST CONTACT SEARCH CHATBOT")
    print("=" * 70)
    print()
    print("✨ Powered by GPT-4 agent with 7 specialized tools:")
    print("   • search() - Semantic vector search")
    print("   • list_by_prefix() - Names by first letter")
    print("   • names_containing() - Substring search")
    print("   • names_by_length() - Filter by name length")
    print("   • names_by_prefix_and_length() - Combined filtering")
    print("   • letter_histogram() - Statistics by first letter")
    print("   • length_histogram() - Statistics by name length")
    print()
    print("💡 The agent intelligently chooses the right tool for each query")
    print("   No manual routing needed - it understands your intent!")
    print()
    print("🌍 Supports Hebrew and English seamlessly")
    print("💬 Maintains conversation context across queries")
    print()
    print("=" * 70)

def main():
    """Main entry point"""
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("\n❌ ERROR: OPENAI_API_KEY not found!")
        print()
        print("Please set your OpenAI API key:")
        print("  1. Create a .env file in this directory")
        print("  2. Add: OPENAI_API_KEY=your-key-here")
        print()
        print("Or set it in your environment:")
        print("  export OPENAI_API_KEY=your-key-here")
        print()
        return 1
    
    # Print banner
    print_banner()
    
    try:
        # Initialize the AI agent
        print("🔧 Initializing AI agent...")
        qa_system = AdvancedVectorDBQASystem(persist_directory="./chroma_db")
        print("✅ Agent ready!")
        print()
        
        # Get stats
        stats = qa_system.get_collection_stats()
        print(f"📊 Database: {stats['document_count']} documents")
        print(f"🧠 Model: {stats['embedding_model']}")
        print()
        
        if stats['document_count'] == 0:
            print("⚠️  Database is empty! Load data with: load <file_path>")
            print()
        
        # Start interactive session
        qa_system.interactive_qa()
        
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
        return 0
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
