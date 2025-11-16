#!/usr/bin/env python3
"""
Setup and Validation Script

Installs dependencies, validates installation, and runs first test.
"""

import subprocess
import sys
from pathlib import Path


def print_header(text):
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)


def install_dependencies():
    """Install required packages"""
    print_header("INSTALLING DEPENDENCIES")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("✅ Dependencies installed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Installation failed: {e}")
        return False


def run_validation():
    """Run validation tests"""
    print_header("RUNNING VALIDATION TESTS")
    
    try:
        result = subprocess.run([sys.executable, "validate.py"])
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return False


def show_next_steps():
    """Display next steps"""
    print_header("🎉 SETUP COMPLETE")
    
    print("\n📋 What to do next:\n")
    print("1️⃣  Start the chatbot:")
    print("   python3 main.py")
    
    print("\n2️⃣  Load your data:")
    print("   > load /path/to/contacts.csv")
    
    print("\n3️⃣  Ask questions:")
    print("   > אנשים שעובדים ב-AI")
    print("   > כמה אנשים יש במאגר?")
    
    print("\n4️⃣  Optional - Start API server:")
    print("   python3 api_server.py")
    
    print("\n📖 Documentation:")
    print("   • README.md - Architecture overview")
    print("   • QUICKSTART.md - Usage examples")
    print("   • ARCHITECTURE.md - Detailed design")
    
    print("\n" + "="*70)


def main():
    """Main setup flow"""
    print_header("CHATBOT SETUP & VALIDATION")
    
    # Check if we're in the right directory
    if not Path("requirements.txt").exists():
        print("❌ Error: Run this from the refactored/ directory")
        print("   cd refactored && python3 setup.py")
        sys.exit(1)
    
    # Install
    if not install_dependencies():
        print("\n❌ Setup failed at dependency installation")
        sys.exit(1)
    
    # Validate
    if not run_validation():
        print("\n⚠️  Some validation tests failed")
        print("   Check error messages above")
        sys.exit(1)
    
    # Success
    show_next_steps()


if __name__ == "__main__":
    main()
