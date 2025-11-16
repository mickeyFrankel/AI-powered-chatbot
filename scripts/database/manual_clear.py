#!/usr/bin/env python3
"""
Manual database clear script - use if UI clear fails
Run this with: python3 manual_clear.py
"""
import os
import shutil

def clear_database():
    """Manually clear all database files"""
    print("\n" + "="*60)
    print("MANUAL DATABASE CLEAR")
    print("="*60 + "\n")
    
    # Stop if server is running
    print("⚠️  WARNING: Make sure the server is STOPPED before running this!")
    response = input("Have you stopped the server? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("❌ Aborted. Stop the server first with Ctrl+C")
        return
    
    # Clear ChromaDB
    if os.path.exists("./chroma_db"):
        print("Deleting ./chroma_db...")
        shutil.rmtree("./chroma_db")
        print("   ✅ Deleted")
    else:
        print("   ⚠️  ./chroma_db not found")
    
    # Clear contacts_db
    if os.path.exists("./contacts_db"):
        print("Deleting ./contacts_db...")
        shutil.rmtree("./contacts_db")
        print("   ✅ Deleted")
    else:
        print("   ⚠️  ./contacts_db not found")
    
    # Clear temp files
    import glob
    temp_files = glob.glob("./temp_upload_*")
    if temp_files:
        print(f"Deleting {len(temp_files)} temp files...")
        for f in temp_files:
            os.remove(f)
        print("   ✅ Deleted")
    
    print("\n" + "="*60)
    print("✅ CLEAR COMPLETE")
    print("="*60)
    print("\nYou can now start the server with: ./start.sh\n")

if __name__ == "__main__":
    clear_database()
