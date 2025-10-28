#!/bin/bash
# Automated Cleanup Script for Chatbot Folder
# Moves redundant files to archive folders for safe keeping

set -e  # Exit on error

CHATBOT_DIR="/Users/miryamstessman/Downloads/chatbot"
cd "$CHATBOT_DIR"

echo "🧹 Chatbot Folder Cleanup Script"
echo "================================="
echo ""
echo "This will move redundant files to archive folders"
echo "Nothing will be permanently deleted (safe operation)"
echo ""
read -p "Continue? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Cancelled"
    exit 1
fi

# Create archive directories
echo ""
echo "📁 Creating archive directories..."
mkdir -p _archive_experimental
mkdir -p _archive_tests
mkdir -p _archive_utils
mkdir -p _archive_docs
mkdir -p _archive_setup

# Counter
MOVED=0

# Phase 1: Archive Experimental Files
echo ""
echo "📦 Phase 1: Archiving experimental versions..."
FILES=(
    "chatbot_final.py"
    "chatbot_fixed.py"
    "chatbot_really_fixed.py"
    "chatbot_no_chroma.py"
    "chatbot_tiered.py"
    "fixed_chatbot.py"
    "simple_chatbot.py"
    "vectoric_search_BACKUP.py"
    "vectoric_search_fixed.py"
    "smart_chatbot_patched.py"
)

for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        mv "$file" _archive_experimental/
        echo "  ✓ Moved $file"
        ((MOVED++))
    fi
done

# Phase 2: Archive Test Scripts
echo ""
echo "🧪 Phase 2: Archiving test scripts..."
FILES=(
    "test_extraction.py"
    "test_fix.py"
    "test_fix_improved.py"
    "test_fuzzy_search.py"
    "test_hybrid_search.py"
    "test_routing.py"
    "test_system.py"
    "debug_search.py"
    "check_csv.py"
)

for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        mv "$file" _archive_tests/
        echo "  ✓ Moved $file"
        ((MOVED++))
    fi
done

# Phase 3: Archive Utility Scripts
echo ""
echo "🔧 Phase 3: Archiving utility scripts..."
FILES=(
    "browse_contacts.py"
    "search_contacts.py"
    "search_patch.py"
    "fix_phone_numbers.py"
    "extract_terms.py"
)

for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        mv "$file" _archive_utils/
        echo "  ✓ Moved $file"
        ((MOVED++))
    fi
done

# Phase 4: Archive Documentation
echo ""
echo "📚 Phase 4: Archiving duplicate documentation..."
FILES=(
    "DISTANCE_METRICS_COMPLETE_GUIDE.md"
    "DISTANCE_METRICS_EXPLAINED.md"
    "DISTANCE_METRICS_INDEX.md"
    "DISTANCE_METRICS_QUICK_REFERENCE.md"
    "DISTANCE_METRICS_SUMMARY.md"
    "CODE_FLOW_DISTANCE_METRICS.md"
    "FOUR_METHODS_QUICK_REFERENCE.md"
)

for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        mv "$file" _archive_docs/
        echo "  ✓ Moved $file"
        ((MOVED++))
    fi
done

# Phase 5: Archive Setup Scripts
echo ""
echo "⚙️  Phase 5: Archiving setup scripts..."
FILES=(
    "setup_312_simple.py"
    "setup_diagnostic.py"
    "setup_mcp_server.py"
    "postgres_data_loader.py"
    "install_rapidfuzz.sh"
)

for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        mv "$file" _archive_setup/
        echo "  ✓ Moved $file"
        ((MOVED++))
    fi
done

# Summary
echo ""
echo "✅ Cleanup Complete!"
echo "==================="
echo ""
echo "📊 Summary:"
echo "   Files moved: $MOVED"
echo ""
echo "📁 Archive locations:"
echo "   _archive_experimental/ - Experimental versions"
echo "   _archive_tests/        - Test scripts"
echo "   _archive_utils/        - Utility scripts"
echo "   _archive_docs/         - Duplicate documentation"
echo "   _archive_setup/        - Setup scripts"
echo ""
echo "🎯 Core files remaining:"
echo "   chatbot.py"
echo "   chatbot_ai_first.py"
echo "   smart_chatbot.py"
echo "   vectoric_search.py"
echo "   vectordb_MCP_server.py"
echo "   postgres_mcp_server.py"
echo ""
echo "💡 All files are safely archived, not deleted"
echo "   You can restore them anytime from archive folders"
echo ""
