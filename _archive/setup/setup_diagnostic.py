#!/usr/bin/env python3
"""
Quick setup and diagnostic for the chatbot
Fixes common issues
"""

import subprocess
import sys
import os

def check_rapidfuzz():
    """Check if rapidfuzz is installed"""
    try:
        import rapidfuzz
        print("✅ rapidfuzz installed")
        return True
    except ImportError:
        print("❌ rapidfuzz NOT installed")
        return False

def install_rapidfuzz():
    """Install rapidfuzz"""
    print("\n📦 Installing rapidfuzz...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "rapidfuzz"])
        print("✅ rapidfuzz installed successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to install: {e}")
        return False

def main():
    print("🔧 Chatbot Setup & Diagnostic")
    print("="*60)
    
    # Check rapidfuzz
    has_rapidfuzz = check_rapidfuzz()
    
    if not has_rapidfuzz:
        response = input("\n❓ Install rapidfuzz for fuzzy search? (y/n): ").lower()
        if response == 'y':
            install_rapidfuzz()
    
    print("\n" + "="*60)
    print("📋 Known Issues & Solutions:")
    print("="*60)
    
    print("\n1️⃣ Telemetry Warnings (Harmless)")
    print("   Message: 'Failed to send telemetry event'")
    print("   Fix: Already suppressed in smart_chatbot.py")
    print("   Impact: None - just noise")
    
    print("\n2️⃣ Wrong Search Results")
    print("   Problem: Semantic search gives unrelated results")
    print("   Example: Search 'שוורץ' returns 'תמיר סמילנסקי'")
    print("   Cause: Name not in database OR Hebrew semantic search weak")
    print("   Solution: Smart chatbot now rejects bad semantic results")
    
    print("\n3️⃣ Cosine Search Limitations")
    print("   Q: Is cosine search accurate?")
    print("   A: For TEXT MATCHING: Yes! For MEANING: Hit or miss")
    print("   ")
    print("   Text search: 'שוורץ' finds 'משה שוורץ' ✅")
    print("   Semantic search: 'שוורץ' finds random names ❌")
    print("   ")
    print("   Smart chatbot uses TEXT FIRST, semantic as fallback")
    
    print("\n4️⃣ Timeout Errors")
    print("   Message: 'Read timed out' from huggingface.co")
    print("   Cause: Slow internet or HuggingFace down")
    print("   Fix: Wait and retry - model already cached locally")
    
    print("\n" + "="*60)
    print("✅ Setup complete!")
    print("\n🚀 Run the chatbot:")
    print("   python chatbot.py")
    print("\n💡 For contact searches:")
    print("   python search_contacts.py")

if __name__ == "__main__":
    main()
