#!/bin/bash

# Chatbot Project Organization Script
# Moves files to appropriate directories for better project structure

set -e  # Exit on error

echo "🧹 Organizing Chatbot Project..."
echo ""

# Create necessary directories
echo "📁 Creating directory structure..."
mkdir -p scripts
mkdir -p tests
mkdir -p _archive/old_code
mkdir -p _archive/backup_files

# Move old/backup code to archive
echo "📦 Archiving old code..."
if [ -f "chatbot_OLD.py" ]; then
    mv chatbot_OLD.py _archive/old_code/
    echo "  ✓ Moved chatbot_OLD.py"
fi

if [ -f "chatbot_unified.py" ]; then
    mv chatbot_unified.py _archive/old_code/
    echo "  ✓ Moved chatbot_unified.py"
fi

if [ -f "vectoric_search_BACKUP.py" ]; then
    mv vectoric_search_BACKUP.py _archive/backup_files/
    echo "  ✓ Moved vectoric_search_BACKUP.py"
fi

if [ -f "vectoric_search_BACKUP2.py" ]; then
    mv vectoric_search_BACKUP2.py _archive/backup_files/
    echo "  ✓ Moved vectoric_search_BACKUP2.py"
fi

if [ -f "vectoric_search_BACKUP3.py" ]; then
    mv vectoric_search_BACKUP3.py _archive/backup_files/
    echo "  ✓ Moved vectoric_search_BACKUP3.py"
fi

if [ -f "archive_old_routing.sh" ]; then
    mv archive_old_routing.sh _archive/old_code/
    echo "  ✓ Moved archive_old_routing.sh"
fi

if [ -f "replace_db.sh" ]; then
    mv replace_db.sh _archive/old_code/
    echo "  ✓ Moved replace_db.sh"
fi

# Move utility scripts to scripts/
echo ""
echo "🔧 Organizing utility scripts..."
for script in preprocess_contacts.py preprocess_csv.py fix_names.py fix_phone_csv.py \
              fix_search.py inspect_csv.py check_vaad.py diagnose.py update_prompt.py \
              manual_clear.py clear_db.sh install.sh; do
    if [ -f "$script" ]; then
        mv "$script" scripts/
        echo "  ✓ Moved $script"
    fi
done

# Move test files to tests/
echo ""
echo "🧪 Organizing test files..."
for test in test_comprehensive.py test_fuzzy_search.py test_real_data.py \
            test_refactor.py test_vaad_search.py; do
    if [ -f "$test" ]; then
        mv "$test" tests/
        echo "  ✓ Moved $test"
    fi
done

# Handle refactored directory - move to archive if it's old code
echo ""
echo "📂 Handling refactored directory..."
if [ -d "refactored" ]; then
    if [ -d "_archive/refactored_version" ]; then
        echo "  ⚠️  Refactored archive already exists, skipping..."
    else
        mv refactored _archive/refactored_version
        echo "  ✓ Moved refactored/ to _archive/refactored_version/"
    fi
fi

# Remove empty directories
echo ""
echo "🗑️  Removing empty directories..."
for dir in src servers preprocessing; do
    if [ -d "$dir" ] && [ -z "$(ls -A $dir)" ]; then
        rmdir "$dir"
        echo "  ✓ Removed empty $dir/"
    fi
done

# Handle data directory - keep it for potential data files
if [ -d "data" ] && [ -z "$(ls -A data)" ]; then
    echo "  ℹ️  Keeping empty data/ for future data files"
fi

# Archive PostgreSQL files if not actively used
echo ""
echo "🗄️  PostgreSQL files (keeping in place for now):"
echo "  - postgres_data/"
echo "  - postgres_backups/"
echo "  - postgres_init/"
echo "  - postgres_mcp_server.py"
echo "  ℹ️  If not needed, manually move these to _archive/postgres/"

echo ""
echo "✅ Organization complete!"
echo ""
echo "📊 Project structure:"
echo "Root (core files only):"
echo "  - chatbot.py, api.py, vectoric_search.py"
echo "  - Configuration files"
echo "  - Start scripts"
echo ""
echo "scripts/       → Utility scripts"
echo "tests/         → Test files"
echo "frontend/      → React app"
echo "docs/          → Documentation"
echo "_archive/      → Old code and backups"
echo ""
echo "🎉 Ready for clean development!"
