#!/bin/bash
# Preview what will be moved/deleted in final cleanup

cd "$(dirname "$0")"

echo "🔍 Final Cleanup Preview"
echo "=" * 50
echo ""

echo "📋 Files that will be MOVED:"
echo ""

echo "To scripts/maintenance/:"
for file in cleanup_root.sh reorganize_project.sh show_status.sh; do
    [ -f "$file" ] && echo "   ✓ $file"
done
echo ""

echo "To scripts/diagnostics/:"
for file in diagnose_nonetype.py test_agent_import.py; do
    [ -f "$file" ] && echo "   ✓ $file"
done
echo ""

echo "To docs/setup-guides/:"
for file in BACKEND_FIX_SUMMARY.md CLEANUP_REFERENCE.md FIX_NONETYPE_ERROR.md STRUCTURE_README.md requirements_updated.txt requirements_py313.txt; do
    [ -f "$file" ] && echo "   ✓ $file"
done
echo ""

echo "🗑️  Optional DELETE:"
[ -d "_backup_20251111_181249" ] && echo "   ⚠️  _backup_20251111_181249/ (old backup, ~$(du -sh _backup_20251111_181249 2>/dev/null | cut -f1))"
echo ""

echo "✅ Files that will STAY in root:"
echo "   • .env (API keys)"
echo "   • .gitignore"
echo "   • README.md"
echo "   • requirements.txt"
echo "   • start_backend.sh (main startup)"
echo ""

echo "📁 Organized directories (stay as-is):"
echo "   • backend/"
echo "   • frontend/"
echo "   • databases/"
echo "   • scripts/"
echo "   • config/"
echo "   • docs/"
echo "   • data/"
echo "   • preprocessing/"
echo "   • tests/"
echo "   • v3_development/"
echo ""

echo "To proceed with cleanup, run:"
echo "   chmod +x final_cleanup.sh"
echo "   ./final_cleanup.sh"
