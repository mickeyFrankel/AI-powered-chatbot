#!/bin/bash
# Simple cleanup script - run this to organize files

echo "🧹 Organizing root directory..."

# Create directories
mkdir -p scripts/maintenance
mkdir -p scripts/diagnostics  
mkdir -p docs/setup-guides

# Move maintenance scripts
[ -f cleanup_root.sh ] && mv cleanup_root.sh scripts/maintenance/
[ -f reorganize_project.sh ] && mv reorganize_project.sh scripts/maintenance/
[ -f show_status.sh ] && mv show_status.sh scripts/maintenance/

# Move diagnostic scripts
[ -f diagnose_nonetype.py ] && mv diagnose_nonetype.py scripts/diagnostics/
[ -f test_agent_import.py ] && mv test_agent_import.py scripts/diagnostics/

# Move preview and final cleanup scripts
[ -f preview_cleanup.sh ] && mv preview_cleanup.sh scripts/maintenance/
[ -f final_cleanup.sh ] && mv final_cleanup.sh scripts/maintenance/

echo "✅ Done! Files organized."
echo ""
echo "📊 Remaining in root:"
ls -1 | grep -v "^\."

echo ""
echo "Want to delete the old backup? Run:"
echo "  rm -rf _backup_20251111_181249"
