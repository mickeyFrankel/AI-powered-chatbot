#!/usr/bin/env python3
"""
Check CSV directly for name patterns
Bypasses the vector database to see raw data
"""

import pandas as pd
import sys

def search_csv_for_name(search_term: str, csv_file: str = "contacts.csv"):
    """Search CSV directly"""
    
    print(f"\n🔍 Searching CSV for: '{search_term}'")
    print("="*60)
    
    try:
        df = pd.read_csv(csv_file, dtype=str, encoding='utf-8')
        print(f"📊 CSV has {len(df)} rows, {len(df.columns)} columns")
        
        # Search all text columns
        search_lower = search_term.lower()
        matches = []
        
        for idx, row in df.iterrows():
            row_text = " ".join([str(val) for val in row.values if pd.notna(val)])
            if search_lower in row_text.lower():
                matches.append((idx, row))
        
        print(f"✅ Found {len(matches)} rows containing '{search_term}'\n")
        
        if not matches:
            print("❌ NO MATCHES in CSV!")
            print("\n💡 Try:")
            print("   - Different spelling")
            print("   - Just first name")
            print("   - Just last name")
            return
        
        # Show matches
        print("-"*60)
        for i, (idx, row) in enumerate(matches[:10], 1):
            print(f"\n{i}. Row #{idx}")
            
            # Show key fields
            for col in ['First Name', 'Last Name', 'Phone 1 - Value', 'E-mail 1 - Value']:
                if col in df.columns and pd.notna(row[col]) and row[col]:
                    print(f"   {col}: {row[col]}")
            
            # Show full name if available
            full_name_parts = []
            for col in ['First Name', 'Middle Name', 'Last Name']:
                if col in df.columns and pd.notna(row[col]) and row[col]:
                    full_name_parts.append(str(row[col]))
            
            if full_name_parts:
                full_name = " ".join(full_name_parts)
                print(f"   Full Name: {full_name}")
        
        if len(matches) > 10:
            print(f"\n... and {len(matches) - 10} more matches")
        
        print("\n" + "="*60)
        
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")

def find_hebrew_names_starting_with(prefix: str, csv_file: str = "contacts.csv"):
    """Find all names starting with Hebrew prefix"""
    
    print(f"\n🔍 Finding names starting with: '{prefix}'")
    print("="*60)
    
    try:
        df = pd.read_csv(csv_file, dtype=str, encoding='utf-8')
        
        matches = []
        prefix_lower = prefix.lower()
        
        for idx, row in df.iterrows():
            # Check first name
            first_name = str(row.get('First Name', ''))
            last_name = str(row.get('Last Name', ''))
            
            if first_name.lower().startswith(prefix_lower) or last_name.lower().startswith(prefix_lower):
                matches.append((idx, first_name, last_name, row))
        
        print(f"✅ Found {len(matches)} names starting with '{prefix}'\n")
        
        if not matches:
            print("❌ No names found!")
            return
        
        # Show unique names
        seen_names = set()
        print("-"*60)
        for idx, first, last, row in matches[:20]:
            full = f"{first} {last}".strip()
            if full and full not in seen_names:
                seen_names.add(full)
                phone = row.get('Phone 1 - Value', '')
                print(f"   {full}")
                if phone and pd.notna(phone):
                    print(f"      📞 {phone}")
        
        if len(matches) > 20:
            print(f"\n... and {len(matches) - 20} more")
        
        print("\n" + "="*60)
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        search_term = " ".join(sys.argv[1:])
        search_csv_for_name(search_term)
    else:
        print("\n🔍 CSV Direct Search Tool")
        print("="*60)
        
        # Test common patterns
        tests = ["אבי", "אתרוג", "שוורץ", "תפוז"]
        
        for test in tests:
            search_csv_for_name(test)
            print("\n")
