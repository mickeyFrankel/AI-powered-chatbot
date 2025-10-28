#!/usr/bin/env python3
"""
Contact Search Tool - Shows ALL results without AI filtering
Perfect for finding contacts by partial name
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from vectoric_search import VectorDBQASystem

def format_contact(doc: str, score: float) -> str:
    """Extract and format contact info from document"""
    lines = []
    
    # Parse document text
    parts = doc.split(" | ")
    info = {}
    for part in parts:
        if ": " in part:
            key, value = part.split(": ", 1)
            info[key.strip()] = value.strip()
    
    # Extract key fields
    name_parts = []
    for key in ['First Name', 'Middle Name', 'Last Name']:
        if key in info and info[key]:
            name_parts.append(info[key])
    
    full_name = " ".join(name_parts) if name_parts else info.get('File As', 'Unknown')
    
    # Find phone number
    phone = None
    for i in range(1, 5):
        phone_key = f'Phone {i} - Value'
        if phone_key in info and info[phone_key]:
            phone = info[phone_key]
            break
    
    # Find email
    email = None
    for i in range(1, 3):
        email_key = f'E-mail {i} - Value'
        if email_key in info and info[email_key]:
            email = info[email_key]
            break
    
    # Format output
    lines.append(f"👤 **{full_name}**")
    if phone:
        lines.append(f"   📞 {phone}")
    if email:
        lines.append(f"   📧 {email}")
    lines.append(f"   🎯 Match score: {score:.2f}")
    
    return "\n".join(lines)

def search_contacts(query: str, limit: int = 10):
    """Search for contacts and display ALL results"""
    
    print("\n" + "="*60)
    print(f"🔍 Searching contacts for: '{query}'")
    print("="*60)
    
    # Initialize system
    qa_system = VectorDBQASystem(persist_directory="./chroma_db")
    
    # Search
    results = qa_system.search(query, n_results=limit)
    
    if not results['results']:
        print(f"\n❌ No contacts found matching '{query}'")
        print("\n💡 Try:")
        print(f"   - Fewer characters: '{query[:max(1, len(query)-1)]}'")
        print(f"   - Different spelling")
        print(f"   - Just first name")
        return
    
    # Display results
    print(f"\n✅ Found {len(results['results'])} contacts:")
    print(f"   Search method: {results.get('search_method', 'unknown')}")
    print("\n" + "-"*60)
    
    for i, result in enumerate(results['results'], 1):
        print(f"\n{i}.")
        formatted = format_contact(
            result['document'],
            result['similarity_score']
        )
        print(formatted)
    
    print("\n" + "="*60)

def main():
    """Interactive contact search"""
    print("🔍 Contact Search Tool")
    print("Shows ALL matches without AI filtering")
    print("="*60)
    
    while True:
        try:
            query = input("\n🔍 Search for contact (or 'quit'): ").strip()
            
            if not query:
                continue
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            search_contacts(query, limit=20)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Command line search
        query = " ".join(sys.argv[1:])
        search_contacts(query)
    else:
        # Interactive mode
        main()
