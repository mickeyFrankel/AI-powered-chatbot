#!/bin/bash
# Script to safely apply the search fix

cd /Users/miryamstessman/Downloads/chatbot

# Backup the original
if [ ! -f "vectoric_search_BACKUP.py" ]; then
    echo "📦 Creating backup..."
    cp vectoric_search.py vectoric_search_BACKUP.py
    echo "✅ Backup created: vectoric_search_BACKUP.py"
else
    echo "⚠️  Backup already exists, skipping..."
fi

# Replace with fixed version
echo "🔧 Replacing with fixed version..."
cp vectoric_search_fixed.py vectoric_search.py
echo "✅ Replaced vectoric_search.py with fixed version"

echo ""
echo "🎉 Done! Now test with: python chatbot.py"
echo ""
echo "To revert: cp vectoric_search_BACKUP.py vectoric_search.py"
