#!/usr/bin/env python3
"""
Auto-fix script for clean response formatting
Run this to update vectoric_search.py automatically
"""

import re
import sys
from pathlib import Path

# File to modify
FILE_PATH = Path(__file__).parent / "backend" / "core" / "vectoric_search.py"

print("🔧 Fixing Response Format in vectoric_search.py")
print("=" * 50)

if not FILE_PATH.exists():
    print(f"❌ File not found: {FILE_PATH}")
    sys.exit(1)

# Read the file
print(f"📖 Reading {FILE_PATH}...")
with open(FILE_PATH, 'r', encoding='utf-8') as f:
    content = f.read()

# Backup
backup_path = FILE_PATH.with_suffix('.py.backup')
with open(backup_path, 'w', encoding='utf-8') as f:
    f.write(content)
print(f"💾 Backup created: {backup_path}")

# ============================================
# FIX 1: Update System Prompt
# ============================================
print("\n🔍 Finding system prompt...")

# Pattern to find the system message in ChatPromptTemplate
system_pattern = r'(\("system",\s*)(""".*?"""|\'\'\'.*?\'\'\'|".*?"|\'.*?\')(\s*\))'

new_system_prompt = '''"""You are a contact search assistant for Hebrew/English queries.

RESPONSE FORMAT - CRITICAL RULES:
1. Format: **שם:** [name]  **טלפון:** [phone]
2. Multiple matches: Show ALL (up to 5), each on new line
3. NO extra text: No "Found in:", "מצאתי", "I found", etc.
4. Just name and phone - nothing else
5. No results: "לא נמצאו תוצאות"

SEARCH RULES:
- Return ALL relevant matches
- For "ועד הבית": show ALL committee members
- Show top 5 most relevant

CORRECT FORMAT:
**שם:** דוד טופורוביץ' ועד בית אבא חלקיה
**טלפון:** 972+ 52-255-4290

**שם:** אסנת ועד הבית חיבנר
**טלפון:** 972+ 54-227-7884

WRONG (Never do this):
❌ "Found in: phone: +972..."
❌ "מצאתי את המידע:"
❌ "אם יש צורך במידע נוסף..."

Be concise. Facts only."""'''

def replace_system_prompt(match):
    return f'{match.group(1)}{new_system_prompt}{match.group(3)}'

content_new = re.sub(system_pattern, replace_system_prompt, content, count=1, flags=re.DOTALL)

if content_new != content:
    print("   ✅ System prompt updated")
else:
    print("   ⚠️  System prompt pattern not found - manual edit needed")

# ============================================
# FIX 2: Increase search result limit
# ============================================
print("\n🔍 Fixing search result limits...")

# Pattern: search_full_text(query, limit=1) → limit=5
content_new = re.sub(
    r'(search_full_text\([^)]*limit\s*=\s*)1(\s*\))',
    r'\g<1>5\2',
    content_new
)

# Pattern: collection.query(..., n_results=1) → n_results=5  
content_new = re.sub(
    r'(collection\.query\([^)]*n_results\s*=\s*)1(\s*[,)])',
    r'\g<1>5\2',
    content_new
)

print("   ✅ Search limits increased to 5")

# ============================================
# Write changes
# ============================================
print("\n💾 Writing changes...")
with open(FILE_PATH, 'w', encoding='utf-8') as f:
    f.write(content_new)

print(f"   ✅ File updated: {FILE_PATH}")
print(f"\n✅ Done! Restart backend to see changes:")
print(f"   ./start_backend.sh")
print(f"\n📝 If anything breaks, restore from backup:")
print(f"   cp {backup_path} {FILE_PATH}")
