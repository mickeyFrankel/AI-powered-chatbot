#!/usr/bin/env python3
"""
Setup MCP Server for Claude Desktop
This will install MCP and configure Claude to use your chatbot
"""

import subprocess
import sys
import json
import os
from pathlib import Path

def main():
    print("🚀 Setting up MCP Server for Claude Desktop")
    print("=" * 60)
    
    project_dir = Path("/Users/miryamstessman/Downloads/chatbot")
    pip_cmd = str(project_dir / ".venv" / "bin" / "pip")
    python_cmd = str(project_dir / ".venv" / "bin" / "python")
    
    # Step 1: Install MCP
    print("\n📦 Step 1: Installing MCP package...")
    result = subprocess.run(
        [pip_cmd, "install", "mcp"],
        capture_output=True,
        text=True,
        timeout=120
    )
    
    if result.returncode == 0:
        print("✅ MCP installed successfully")
    else:
        print(f"⚠️  MCP installation issue: {result.stderr}")
        print("Continuing anyway...")
    
    # Step 2: Create Claude Desktop config
    print("\n📝 Step 2: Creating Claude Desktop configuration...")
    
    # Claude Desktop config location on macOS
    claude_config_dir = Path.home() / "Library" / "Application Support" / "Claude"
    claude_config_file = claude_config_dir / "claude_desktop_config.json"
    
    # Create directory if it doesn't exist
    claude_config_dir.mkdir(parents=True, exist_ok=True)
    
    # Load existing config or create new one
    if claude_config_file.exists():
        with open(claude_config_file, 'r') as f:
            config = json.load(f)
        print("✅ Found existing Claude config")
    else:
        config = {"mcpServers": {}}
        print("✅ Creating new Claude config")
    
    # Add our vectordb server
    config["mcpServers"]["vectordb"] = {
        "command": str(python_cmd),
        "args": [str(project_dir / "vectordb_MCP_server.py")],
        "env": {
            "PYTHONPATH": str(project_dir),
            "CHROMA_DB_DIR": str(project_dir / "chroma_db")
        }
    }
    
    # Save config
    with open(claude_config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Configuration saved to:")
    print(f"   {claude_config_file}")
    
    # Step 3: Test the server
    print("\n🧪 Step 3: Testing MCP server...")
    
    test_script = '''
import sys
sys.path.insert(0, "/Users/miryamstessman/Downloads/chatbot")

try:
    from vectordb_MCP_server import mcp, ping, stats, ask, search
    print("✅ MCP server imports successfully")
    
    # Test ping
    result = ping()
    print(f"✅ Ping successful: {result['ok']}")
    
    # Test stats
    stats_result = stats()
    print(f"✅ Stats: {stats_result['document_count']} documents")
    
    print("\\n🎉 MCP server is working!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
'''
    
    test_file = project_dir / "test_mcp_server.py"
    with open(test_file, 'w') as f:
        f.write(test_script)
    
    result = subprocess.run([python_cmd, str(test_file)], capture_output=True, text=True)
    print(result.stdout)
    
    if result.returncode == 0:
        test_file.unlink()
        print("\n" + "="*60)
        print("🎉 SUCCESS! MCP Server is configured!")
        print("="*60)
        print("\n📋 Next steps:")
        print("1. ✅ MCP package installed")
        print("2. ✅ Claude Desktop configured")
        print("3. ✅ Server tested and working")
        print("\n🔄 TO ACTIVATE:")
        print("1. Quit Claude Desktop completely (Cmd+Q)")
        print("2. Restart Claude Desktop")
        print("3. Look for 🔌 icon in Claude - your 'vectordb' server should appear")
        print("\n💡 USAGE IN CLAUDE:")
        print("Ask Claude things like:")
        print('  - "Search my contacts for machine learning experts"')
        print('  - "Find documents about deep learning"')
        print('  - "What contacts do I have in the AI industry?"')
        print("\n📄 Configuration file:")
        print(f"   {claude_config_file}")
        return True
    else:
        print(result.stderr)
        print("\n❌ Server test failed")
        print("Check the errors above and try again")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
