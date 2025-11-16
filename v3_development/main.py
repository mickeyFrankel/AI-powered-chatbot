"""
VectorDB Q&A System - Clean Architecture Entry Point

SOLID Principles:
- Single Responsibility: Each module has one job
- Open/Closed: Easy to add new search strategies, file loaders
- Liskov Substitution: All strategies/loaders are interchangeable
- Interface Segregation: Clean protocols for each component
- Dependency Inversion: Agent depends on abstractions, not concrete classes

Author: AI Assistant
Date: 2025-10-30
"""

from pathlib import Path
from database import VectorDatabase
from file_loaders import FileLoaderFactory
from agent import ChatAgent
from config import UIMessages


class ChatbotApp:
    """Main application - Facade Pattern"""
    
    def __init__(self):
        print("\n🚀 Initializing AI Chatbot...")
        self.db = VectorDatabase()
        self.loader_factory = FileLoaderFactory()
        self.agent = ChatAgent(self.db)
        print(f"✅ Loaded {self.db.count():,} contacts\n")
    
    def run(self):
        """Interactive chat loop"""
        print("="*70)
        print("  AI Chatbot - מערכת שאלות ותשובות")
        print("="*70)
        print("\nCommands:")
        print("  • Type your question in Hebrew or English")
        print("  • 'load <file>' - Load contacts file")
        print("  • 'stats' - Show database statistics")
        print("  • 'quit' - Exit\n")
        
        while True:
            try:
                user_input = input("💬 You: ").strip()
                
                if not user_input:
                    continue
                
                # Exit
                if user_input.lower() in UIMessages.EXIT_PHRASES:
                    print(f"\n{UIMessages.GOODBYE_MESSAGE}")
                    break
                
                # Stats
                if user_input.lower() in ['stats', 'מידע', 'סטטיסטיקה']:
                    self._show_stats()
                    continue
                
                # Load file
                if user_input.lower().startswith('load '):
                    filepath = user_input[5:].strip()
                    self._load_file(filepath)
                    continue
                
                # Check DB has data
                if self.db.count() == 0:
                    print(f"⚠️  {UIMessages.NO_DOCUMENTS}")
                    continue
                
                # Answer
                answer = self.agent.answer(user_input)
                print(f"\n🤖 Bot: {answer}\n")
            
            except KeyboardInterrupt:
                print(f"\n\n{UIMessages.GOODBYE_MESSAGE}")
                break
            except Exception as e:
                print(f"❌ Error: {e}\n")
    
    def _show_stats(self):
        """Display statistics"""
        stats = self.db.get_stats()
        print(f"\n📊 Database Statistics:")
        print(f"   Total contacts: {stats.total_documents:,}")
        print(f"   Sources: {len(stats.sources)}")
        for source, count in stats.sources.items():
            print(f"     • {source}: {count:,} contacts")
        print()
    
    def _load_file(self, filepath: str):
        """Load file into database"""
        try:
            path = Path(filepath)
            print(f"📂 Loading {path.name}...")
            
            documents = self.loader_factory.load_file(path)
            report = self.db.add_documents(documents)
            
            print(f"✅ Added {report.documents_added:,} new contacts")
            if report.documents_skipped:
                print(f"   (Skipped {report.documents_skipped:,} duplicates)")
            print(f"   Total in DB: {report.total_in_db:,}\n")
        
        except Exception as e:
            print(f"❌ Failed to load file: {e}\n")


def main():
    """Entry point"""
    app = ChatbotApp()
    app.run()


if __name__ == "__main__":
    main()
