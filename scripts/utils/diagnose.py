#!/usr/bin/env python3
"""Show what names are actually stored in the database"""

import sys
sys.path.insert(0, '.')
from vectoric_search import VectorDBQASystem

qa = VectorDBQASystem()
all_metas = qa._get_all_metadatas()

print(f"Total contacts: {len(all_metas)}\n")
print("First 20 contacts:")
print("="*80)

for i, meta in enumerate(all_metas[:20]):
    name = meta.get('name', 'NO NAME')
    first_name = meta.get('First Name', '')
    last_name = meta.get('Last Name', '')
    phone = meta.get('phone', '')
    
    print(f"{i+1}. Stored name: {name}")
    if first_name or last_name:
        print(f"   CSV had: First='{first_name}' Last='{last_name}'")
    if phone:
        print(f"   Phone: {phone}")
    print()

# Check for David
print("\n" + "="*80)
print("Looking for 'David' in database:")
david_found = False
for meta in all_metas:
    name = meta.get('name', '')
    if 'david' in name.lower():
        print(f"  Found: {name}")
        david_found = True
        
    # Check if David is in First Name field but not extracted
    first = meta.get('First Name', '')
    if 'david' in first.lower() and 'david' not in name.lower():
        print(f"  ❌ BUG: First Name has 'David' but stored name is: {name}")
        david_found = True

if not david_found:
    print("  ❌ NOT FOUND - David is not searchable")

print("\n" + "="*80)
print("Diagnosis:")
print("If you see stored names as phone numbers (+972...),")
print("then name extraction during CSV loading is broken.")
