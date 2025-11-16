#!/usr/bin/env python3
import re

with open("vectoric_search.py", 'r') as f:
    content = f.read()

with open("vectoric_search_BACKUP2.py", 'w') as f:
    f.write(content)

# Find and replace _derive_name_fields function
pattern = r'(    def _derive_name_fields\(self, text: str, metadata: Optional\[dict\] = None\) -> dict:.*?(?=\n    def |\n\nclass |\Z))'

new_function = '''    def _derive_name_fields(self, text: str, metadata: Optional[dict] = None) -> dict:
        name = ""
        md = metadata or {}
        
        # Try First + Last Name (Google Contacts CSV)
        first = md.get('First Name', '').strip()
        last = md.get('Last Name', '').strip()
        
        if first or last:
            name = ' '.join([p for p in [first, last] if p])
        
        # Fallback to Organization Name
        if not name:
            name = md.get('Organization Name', '').strip()
        
        # Last resort: extract from text
        if not name:
            preferred_keys = ['name', 'title', 'industry_name']
            for k in preferred_keys:
                v = md.get(k)
                if v and str(v).strip():
                    name = str(v).strip()
                    break
        
        if not name:
            t = self._normalize(text)
            first_chunk = t.split(" | ")[0]
            if ":" in first_chunk:
                maybe = first_chunk.split(":", 1)[1].strip()
                if maybe:
                    name = maybe
            if not name:
                name = first_chunk.strip()
        
        first_char = ""
        for ch in name:
            if ch.isalpha():
                first_char = ch.upper() if ch.isascii() else ch
                break
        
        return {
            "name": name,
            "first_char": first_char,
            "name_len": len(name),
        }'''

content = re.sub(pattern, new_function, content, flags=re.DOTALL)

with open("vectoric_search.py", 'w') as f:
    f.write(content)

print("✅ Fixed _derive_name_fields")
print("Now: reset database and reload contacts.csv")
