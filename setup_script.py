#!/usr/bin/env python3
"""
Setup and Environment Check Script for VectorDB Q&A System
This script will check your Python environment and install missing packages
"""

import sys
import subprocess
import importlib.util

def check_python_version():
    """Check Python version"""
    print(f"🐍 Python version: {sys.version}")
    print(f"🐍 Python executable: {sys.executable}")
    
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required!")
        return False
    else:
        print("✅ Python version OK")
        return True

def check_package(package_name, import_name=None):
    """Check if a package is installed"""
    if import_name is None:
        import_name = package_name
    
    spec = importlib.util.find_spec(import_name)
    if spec is not None:
        print(f"✅ {package_name} is installed")
        return True
    else:
        print(f"❌ {package_name} is NOT installed")
        return False

def install_package(package_name):
    """Install a package using pip"""
    try:
        print(f"📦 Installing {package_name}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        print(f"✅ {package_name} installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {package_name}: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 VectorDB Q&A System - Environment Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return
    
    print("\n📋 Checking required packages...")
    
    # Core required packages
    core_packages = [
        ("chromadb", "chromadb"),
        ("sentence-transformers", "sentence_transformers"),
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("scikit-learn", "sklearn")
    ]
    
    # Optional packages
    optional_packages = [
        ("openpyxl", "openpyxl"),
        ("python-docx", "docx"),
        ("PyPDF2", "PyPDF2")
    ]
    
    # Check and install core packages
    missing_core = []
    for pkg_name, import_name in core_packages:
        if not check_package(pkg_name, import_name):
            missing_core.append(pkg_name)
    
    if missing_core:
        print(f"\n📦 Installing missing core packages: {', '.join(missing_core)}")
        for pkg in missing_core:
            install_package(pkg)
    
    # Check optional packages
    print(f"\n📋 Checking optional packages...")
    missing_optional = []
    for pkg_name, import_name in optional_packages:
        if not check_package(pkg_name, import_name):
            missing_optional.append(pkg_name)
    
    if missing_optional:
        print(f"\n⚠️  Optional packages missing: {', '.join(missing_optional)}")
        print("These packages enable support for additional file formats:")
        print("  - openpyxl: Excel files (.xlsx, .xls)")
        print("  - python-docx: Word documents (.docx)")
        print("  - PyPDF2: PDF files (.pdf)")
        
        install_choice = input("\nInstall optional packages? (y/n): ").lower().strip()
        if install_choice == 'y':
            for pkg in missing_optional:
                install_package(pkg)
    
    # Final check
    print("\n🔍 Final package check...")
    all_packages = core_packages + optional_packages
    
    available_packages = []
    for pkg_name, import_name in all_packages:
        if check_package(pkg_name, import_name):
            available_packages.append(pkg_name)
    
    print(f"\n✅ Available packages: {', '.join(available_packages)}")
    
    # Test import of our system
    print("\n🧪 Testing VectorDB system import...")
    try:
        # Try to create a minimal test
        import chromadb
        from sentence_transformers import SentenceTransformer
        import pandas as pd
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity
        
        print("✅ Core system components can be imported successfully!")
        
        # Test ChromaDB
        client = chromadb.Client()
        print("✅ ChromaDB is working!")
        
        print("\n🎉 Setup complete! You can now run the VectorDB Q&A system.")
        print("\nTo run the system:")
        print(f"  {sys.executable} vectordb_system.py")
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        print("Please check the error messages above and try installing missing packages.")

if __name__ == "__main__":
    main()
