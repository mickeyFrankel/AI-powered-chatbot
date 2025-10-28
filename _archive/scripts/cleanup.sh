#!/bin/bash
# Cleanup script - Organize chatbot files

cd /Users/miryamstessman/Downloads/chatbot

echo "🧹 Chatbot Project Cleanup"
echo "=========================="
echo ""

# Create organization folders
mkdir -p _archive/experiments
mkdir -p _archive/tests

echo "📦 Moving experimental files to _archive/..."

# Move my experimental chatbot versions
mv chatbot_fixed.py _archive/experiments/ 2>/dev/null
mv chatbot_really_fixed.py _archive/experiments/ 2>/dev/null
mv smart_chatbot_patched.py _archive/experiments/ 2>/dev/null

# Move test files
mv test_fix.py _archive/tests/ 2>/dev/null
mv test_fix_improved.py _archive/tests/ 2>/dev/null
mv test_extraction.py _archive/tests/ 2>/dev/null
mv test_fuzzy_search.py _archive/tests/ 2>/dev/null
mv test_hybrid_search.py _archive/tests/ 2>/dev/null
mv test_routing.py _archive/tests/ 2>/dev/null
mv test_system.py _archive/tests/ 2>/dev/null

# Move helper files
mv search_patch.py _archive/experiments/ 2>/dev/null
mv extract_terms.py _archive/experiments/ 2>/dev/null
mv vectoric_search_fixed.py _archive/experiments/ 2>/dev/null

# Move shell scripts
mv restore_and_test.sh _archive/ 2>/dev/null
mv apply_fix.sh _archive/ 2>/dev/null

echo "✅ Files organized!"
echo ""
echo "📂 Project structure:"
echo ""
echo "CORE FILES (keep):"
echo "  ✅ chatbot.py              - Original entry point"
echo "  ✅ smart_chatbot.py        - Original smart routing"
echo "  ✅ vectoric_search.py      - Core search engine"
echo ""
echo "RECOMMENDED TO USE:"
echo "  ⭐ chatbot_ai_first.py     - AI-powered (BEST)"
echo "  💰 chatbot_tiered.py       - Free alternative"
echo ""
echo "ARCHIVED:"
echo "  📦 _archive/experiments/   - My experimental versions"
echo "  📦 _archive/tests/         - Test files"
echo ""
echo "🎯 Quick Start:"
echo "  python chatbot_ai_first.py    # Recommended (needs OpenAI key)"
echo "  python chatbot_tiered.py      # Free alternative"
echo "  python chatbot.py             # Original (stable)"
echo ""
