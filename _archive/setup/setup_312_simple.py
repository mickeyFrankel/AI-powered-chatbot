#!/usr/bin/env python3
"""
Simple script to set up Python 3.12 environment and install ChromaDB
"""

import subprocess
import sys
import shutil
from pathlib import Path

def main():
    print("🚀 Setting up Python 3.12 environment with ChromaDB")
    print("=" * 60)
    
    project_dir = Path.cwd()
    print(f"📁 Project directory: {project_dir}")
    
    # Remove old virtual environment
    old_venv = project_dir / ".venv"
    if old_venv.exists():
        print("\n🗑️  Removing old Python 3.13 virtual environment...")
        try:
            shutil.rmtree(old_venv)
            print("✅ Old virtual environment removed")
        except Exception as e:
            print(f"❌ Error removing old venv: {e}")
            return False
    
    # Create new virtual environment with Python 3.12
    print("\n🏗️  Creating new virtual environment with Python 3.12...")
    try:
        result = subprocess.run(
            ["/opt/homebrew/bin/python3.12", "-m", "venv", ".venv"],
            capture_output=True,
            text=True,
            timeout=60
        )
        if result.returncode == 0:
            print("✅ Virtual environment created")
        else:
            print(f"❌ Failed to create venv: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    # Get pip path
    pip_cmd = str(project_dir / ".venv" / "bin" / "pip")
    python_cmd = str(project_dir / ".venv" / "bin" / "python")
    
    # Verify Python version
    print("\n🔍 Verifying Python version...")
    result = subprocess.run([python_cmd, "--version"], capture_output=True, text=True)
    print(f"✅ {result.stdout.strip()}")
    
    # Upgrade pip
    print("\n📦 Upgrading pip...")
    subprocess.run([pip_cmd, "install", "--upgrade", "pip"], capture_output=True)
    print("✅ Pip upgraded")
    
    # Install dependencies
    print("\n📦 Installing dependencies...")
    
    dependencies = [
        "python-dotenv",
        "numpy",
        "pandas",
        "scikit-learn",
        "sentence-transformers",
        "openai",
        "openpyxl",
        "python-docx",
        "PyPDF2",
    ]
    
    for dep in dependencies:
        print(f"   Installing {dep}...")
        result = subprocess.run(
            [pip_cmd, "install", dep],
            capture_output=True,
            text=True,
            timeout=300
        )
        if result.returncode == 0:
            print(f"   ✅ {dep}")
        else:
            print(f"   ⚠️  {dep} - continuing anyway")
    
    # Install ChromaDB
    print("\n🎯 Installing ChromaDB (the main goal!)...")
    result = subprocess.run(
        [pip_cmd, "install", "chromadb==0.4.24"],
        capture_output=True,
        text=True,
        timeout=300
    )
    
    if result.returncode == 0:
        print("✅ ChromaDB installed successfully!")
    else:
        print("❌ ChromaDB installation failed")
        print(f"Error: {result.stderr}")
        return False
    
    # Test imports
    print("\n🧪 Testing imports...")
    test_script = '''
import sys
try:
    import chromadb
    print("✅ ChromaDB")
    import pandas
    print("✅ pandas")
    import numpy
    print("✅ numpy")
    import sentence_transformers
    print("✅ sentence-transformers")
    import sklearn
    print("✅ scikit-learn")
    print("\\n🎉 All packages installed successfully!")
except Exception as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)
'''
    
    test_file = project_dir / "test_312_imports.py"
    with open(test_file, 'w') as f:
        f.write(test_script)
    
    result = subprocess.run([python_cmd, str(test_file)], capture_output=True, text=True)
    print(result.stdout)
    
    if result.returncode == 0:
        test_file.unlink()
        print("\n🎉 SUCCESS! Python 3.12 + ChromaDB setup complete!")
        print("\n🚀 To run your chatbot:")
        print("   source .venv/bin/activate")
        print("   python vectoric_search.py")
        return True
    else:
        print("\n❌ Import test failed")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ Setup failed")
        sys.exit(1)
