#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')
from vectoric_search import VectorDBQASystem

qa = VectorDBQASystem()

# Search for "ועד בית"
keyword = "ועד בית"
print(f"Searching for: '{keyword}'\n")

# Method 1: Full text search
results = qa.search_full_text(keyword, limit=50)
print(f"Full text search found: {len(results)} results")
if results:
    for r in results[:5]:
        print(f"  - {r['name']}")
        print(f"    Context: {r['keyword_context'][:100]}")
        print()

# Method 2: Check all names
print("\nChecking all names for 'ועד':")
all_metas = qa._get_all_metadatas()
found = []
for meta in all_metas:
    name = meta.get('name', '')
    if 'ועד' in name:
        found.append(name)

if found:
    print(f"Found {len(found)} contacts with 'ועד' in name:")
    for name in found[:10]:
        print(f"  - {name}")
else:
    print("  None found in names")

# Method 3: Search in all metadata
print("\nSearching all metadata for 'ועד בית':")
found_in_meta = []
for meta in all_metas:
    for key, val in meta.items():
        if val and 'ועד בית' in str(val):
            found_in_meta.append({
                'name': meta.get('name'),
                'field': key,
                'value': str(val)[:100]
            })
            break

if found_in_meta:
    print(f"Found {len(found_in_meta)} contacts:")
    for item in found_in_meta[:10]:
        print(f"  - {item['name']}")
        print(f"    Field: {item['field']}")
        print(f"    Value: {item['value']}")
        print()
else:
    print("  Not found in any metadata")

# Specific names mentioned
print("\nLooking for specific names:")
names_to_find = ['אסנת', 'טפורוב', 'דוד']
for search_name in names_to_find:
    found = [m.get('name', '') for m in all_metas if search_name in m.get('name', '')]
    if found:
        print(f"  '{search_name}': {found[:3]}")
    else:
        print(f"  '{search_name}': Not found")
