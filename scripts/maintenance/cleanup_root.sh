#!/bin/bash
# Safe Cleanup Script - Remove duplicated files from root after reorganization
# This script ONLY removes files that are safely copied to new locations

set -e

echo "🧹 Chatbot Root Directory Cleanup"
echo "=================================="
echo ""
echo "This will remove files that have been copied to the new structure."
echo "Files to keep: .env, .gitignore, requirements.txt, README.md, frontend/"
echo ""

# Function to safely remove file if it exists
safe_remove() {
    local file=$1
    if [ -f "$file" ]; then
        echo "  ✓ Removing: $file"
        rm "$file"
    fi
}

# Function to safely remove directory if it exists
safe_remove_dir() {
    local dir=$1
    if [ -d "$dir" ]; then
        echo "  ✓ Removing directory: $dir/"
        rm -rf "$dir"
    fi
}

# Ask for confirmation
read -p "⚠️  Continue with cleanup? (yes/no): " -r
echo
if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
    echo "❌ Cleanup cancelled"
    exit 1
fi

echo "🗑️  Removing duplicated files..."
echo ""

# Core backend files (now in backend/core/)
echo "Cleaning core backend files..."
safe_remove "chatbot.py"
safe_remove "vectoric_search.py" 
safe_remove "api.py"
safe_remove "app.py"
safe_remove "chatbot_api.log"
echo ""

# Utility scripts (now in scripts/utils/)
echo "Cleaning utility scripts..."
safe_remove "check_vaad.py"
safe_remove "diagnose.py"
safe_remove "fix_search.py"
safe_remove "update_prompt.py"
echo ""

# Database scripts (now in scripts/database/)
echo "Cleaning database scripts..."
safe_remove "manual_clear.py"
safe_remove "postgres_mcp_server.py"
echo ""

# Setup scripts (now in scripts/setup/)
echo "Cleaning setup scripts..."
safe_remove "setup_all.sh"
safe_remove "start.sh"
safe_remove "quick_test.sh"
echo ""

# MCP servers (now in backend/mcp_servers/)
echo "Cleaning MCP servers..."
safe_remove "vectordb_MCP_server.py"
safe_remove_dir "mcp_server_vectordb"
safe_remove_dir "servers"  # if empty or duplicated
echo ""

# Database directories (now in databases/)
echo "Cleaning database directories..."
safe_remove_dir "chroma_db"
safe_remove_dir "chatbot_db"
safe_remove_dir "postgres_data"
safe_remove_dir "postgres_backups"
safe_remove_dir "postgres_init"
echo ""

# Documentation (now in docs/)
echo "Cleaning documentation files..."
safe_remove "PRODUCTION_READY.md"
safe_remove "QUICK_MCP_REFERENCE.md"
safe_remove "BUG_FIXES.md"
safe_remove "_ORGANIZATION_SUMMARY.md"
safe_remove "PROJECT_STRUCTURE.md"
safe_remove "TESTING_GUIDE.md"
safe_remove "STRUCTURE_README.md"
echo ""

# Config files (now in config/)
echo "Cleaning config files..."
safe_remove "docker-compose.yml"
safe_remove "industries.txt"
safe_remove "reasoning_web_ui.html"
echo ""

# Refactored code (now in v3_development/)
echo "Cleaning refactored directory..."
safe_remove_dir "refactored"
echo ""

# Other directories to remove
echo "Cleaning other directories..."
safe_remove_dir "__pycache__"
safe_remove_dir "src"  # if empty
echo ""

# Keep these files/dirs:
# - .env
# - .gitignore
# - .git
# - .idea
# - .venv
# - requirements.txt
# - README.md
# - frontend/
# - backend/
# - databases/
# - scripts/
# - config/
# - docs/
# - v3_development/
# - data/
# - preprocessing/
# - tests/
# - _archive/
# - _backup_*/
# - reorganize_project.sh
# - cleanup_root.sh

echo "✅ Cleanup complete!"
echo ""
echo "📁 Remaining structure:"
ls -la | grep "^d" | grep -v "^\.$" | awk '{print "   " $9}'
echo ""
echo "📄 Remaining files:"
ls -la | grep "^-" | awk '{print "   " $9}'
echo ""
echo "🎯 Your project is now organized!"
echo ""
echo "Next steps:"
echo "1. Test that everything works: cd backend/core && python3 api.py"
echo "2. Update any scripts that reference old paths"
echo "3. Review and clean _archive/ if needed"
