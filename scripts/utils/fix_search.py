#!/usr/bin/env python3
"""Fix search functions - no name extraction changes"""

def fix_search_functions():
    with open("vectoric_search.py", 'r') as f:
        content = f.read()
    
    # Backup
    with open("vectoric_search_BACKUP.py", 'w') as f:
        f.write(content)
    
    # Fix 1: list_by_prefix - handle first char extraction properly
    old1 = """    def list_by_prefix(self, letter: str) -> list[str]:
        letter = letter.upper().strip()
        all_docs = self.collection.get(include=["metadatas"])
        names = []
        for md in all_docs.get("metadatas", []):
            name = (md.get("name") or "").strip()
            if name.upper().startswith(letter):
                names.append(name)
        return sorted(set(names), key=str.upper)"""
    
    new1 = """    def list_by_prefix(self, letter: str) -> list[str]:
        letter = letter.upper().strip()
        all_docs = self.collection.get(include=["metadatas"])
        names = []
        for md in all_docs.get("metadatas", []):
            name = (md.get("name") or "").strip()
            if not name:
                continue
            # Get first alphabetic character
            first_char = None
            for ch in name:
                if ch.isalpha():
                    first_char = ch.upper()
                    break
            if first_char and first_char == letter:
                names.append(name)
        return sorted(set(names), key=str.upper)"""
    
    content = content.replace(old1, new1)
    
    # Fix 2: count_by_language - calculate on-the-fly
    old2 = """            for meta in all_metas:
                first_char = meta.get('first_char', '')
                name = meta.get('name', '')
                
                if not first_char and name:
                    # Fallback: check first actual character
                    for ch in name:
                        if ch.isalpha():
                            first_char = ch
                            break
                
                # Check if Hebrew (U+0590 to U+05FF)
                if first_char and '\\u0590' <= first_char <= '\\u05FF':
                    hebrew_count += 1
                # Check if English/Latin (A-Z, a-z)
                elif first_char and first_char.isascii() and first_char.isalpha():
                    english_count += 1
                else:
                    other_count += 1"""
    
    new2 = """            for meta in all_metas:
                name = meta.get('name', '').strip()
                if not name:
                    other_count += 1
                    continue
                
                # Get first alphabetic character
                first_char = None
                for ch in name:
                    if ch.isalpha():
                        first_char = ch
                        break
                
                if not first_char:
                    other_count += 1
                    continue
                
                # Check if Hebrew (U+0590 to U+05FF)
                if '\\u0590' <= first_char <= '\\u05FF':
                    hebrew_count += 1
                # Check if English/Latin (A-Z, a-z)
                elif first_char.isascii() and first_char.isalpha():
                    english_count += 1
                else:
                    other_count += 1"""
    
    content = content.replace(old2, new2)
    
    with open("vectoric_search.py", 'w') as f:
        f.write(content)
    
    print("✅ Fixed search functions")
    print("Now run: python vectoric_search.py, then test again")

if __name__ == "__main__":
    fix_search_functions()
