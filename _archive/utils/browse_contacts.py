#!/usr/bin/env python3
"""
Browse contacts starting with a letter - see what's actually in database
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import os
os.environ['ANONYMIZED_TELEMETRY'] = 'False'

from vectoric_search import VectorDBQASystem

def browse_contacts_by_letter(letter: str):
    """Show all contacts starting with a letter"""
    
    print(f"\n🔍 Browsing contacts starting with '{letter}'")
    print("="*60)
    
    qa_system = VectorDBQASystem(persist_directory="./chroma_db")
    
    # Get all documents
    all_data = qa_system.collection.get(include=["documents"])
    documents = all_data.get("documents", [])
    
    matches = []
    letter_upper = letter.upper()
    
    for doc in documents:
        # Check if any name field starts with the letter
        if letter in doc or letter_upper in doc:
            # Extract name
            parts = doc.split('|')
            first_name = ""
            last_name = ""
            phone = ""
            
            for part in parts:
                if 'First Name:' in part:
                    first_name = part.split(':', 1)[1].strip()
                elif 'Last Name:' in part:
                    last_name = part.split(':', 1)[1].strip()
                elif 'Phone 1 - Value:' in part:
                    phone = part.split(':', 1)[1].strip()
            
            full_name = f"{first_name} {last_name}".strip()
            
            # Check if name starts with letter
            if full_name.startswith(letter) or full_name.startswith(letter_upper):
                matches.append({
                    'name': full_name,
                    'phone': phone,
                    'first': first_name,
                    'last': last_name
                })
    
    if not matches:
        print(f"❌ No contacts found starting with '{letter}'")
        return
    
    # Sort by name
    matches.sort(key=lambda x: x['name'])
    
    print(f"\n✅ Found {len(matches)} contacts:\n")
    
    for i, contact in enumerate(matches, 1):
        print(f"{i:3}. {contact['name']}")
        if contact['phone']:
            print(f"     📞 {contact['phone']}")
        print()
    
    print("="*60)

def main():
    """Interactive browse"""
    
    print("📖 Contact Browser")
    print("See what's actually in your database")
    print("="*60)
    
    while True:
        letter = input("\n🔍 Browse contacts starting with (or 'quit'): ").strip()
        
        if not letter:
            continue
        
        if letter.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        
        if len(letter) > 1:
            print("⚠️  Please enter just one letter")
            continue
        
        browse_contacts_by_letter(letter)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        browse_contacts_by_letter(sys.argv[1])
    else:
        main()
