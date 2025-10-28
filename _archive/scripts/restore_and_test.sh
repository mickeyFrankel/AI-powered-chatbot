#!/bin/bash
# Restore the original file and apply targeted fix

cd /Users/miryamstessman/Downloads/chatbot

echo "🔄 Restoring original vectoric_search.py from backup..."
cp vectoric_search_BACKUP.py vectoric_search.py

echo "✅ Restored! Now run the chatbot:"
echo "   python chatbot.py"
echo ""
echo "Note: The hierarchical search fix requires code changes."
echo "For now, using the working original version."
