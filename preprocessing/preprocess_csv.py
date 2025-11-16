#!/usr/bin/env python3
"""
CSV Preprocessing Tool - Clean messy contact files before import

Features:
- Remove empty columns
- Remove duplicate rows
- Fix phone numbers (scientific notation)
- Standardize column names
- Remove empty rows
- Trim whitespace
- Fill missing values intelligently
- Optional: merge duplicate contacts by name

Usage: python preprocess_csv.py input.csv output_clean.csv
"""
import sys
import pandas as pd
import re
from collections import Counter

def fix_phone_number(value):
    """Convert scientific notation to proper phone format"""
    if pd.isna(value) or value == '':
        return ''
    
    value_str = str(value).strip()
    
    # Remove common phone formatting
    value_str = value_str.replace('-', '').replace(' ', '').replace('(', '').replace(')', '')
    
    # Detect scientific notation
    if 'e+' in value_str.lower() or 'e-' in value_str.lower():
        try:
            num = int(float(value_str))
            result = str(num)
            
            # Israeli phone: add leading 0 if missing
            if len(result) in [9, 10]:
                if not result.startswith('0'):
                    result = '0' + result
            
            return result
        except (ValueError, OverflowError):
            return value_str
    
    # Remove non-digits
    digits_only = re.sub(r'\D', '', value_str)
    
    # Israeli phone validation
    if len(digits_only) == 9:
        return '0' + digits_only
    elif len(digits_only) == 10 and digits_only[0] == '0':
        return digits_only
    
    return value_str

def standardize_column_name(col):
    """Standardize column names"""
    col = col.strip().lower()
    
    # Common variations
    mappings = {
        'full name': 'name',
        'contact name': 'name',
        'first name': 'first_name',
        'last name': 'last_name',
        'phone number': 'phone',
        'phone 1': 'phone_1',
        'phone 2': 'phone_2',
        'mobile': 'phone',
        'cell': 'phone',
        'telephone': 'phone',
        'e-mail': 'email',
        'e-mail address': 'email',
        'email address': 'email',
        'email 1': 'email_1',
        'email 2': 'email_2',
    }
    
    if col in mappings:
        return mappings[col]
    
    # Remove special characters, keep underscores
    col = re.sub(r'[^\w\s]', '', col)
    col = re.sub(r'\s+', '_', col)
    
    return col

def preprocess_csv(input_file, output_file, merge_duplicates=False, min_column_fill=0.05):
    """
    Clean and preprocess CSV file
    
    Args:
        input_file: Input CSV path
        output_file: Output CSV path
        merge_duplicates: If True, merge contacts with same name
        min_column_fill: Minimum % of non-empty values to keep column (default 5%)
    """
    print(f"📖 Reading: {input_file}")
    df = pd.read_csv(input_file, encoding='utf-8')
    
    original_rows = len(df)
    original_cols = len(df.columns)
    
    print(f"   Original: {original_rows} rows, {original_cols} columns")
    print()
    
    # Step 1: Remove completely empty rows
    df = df.dropna(how='all')
    removed_empty_rows = original_rows - len(df)
    if removed_empty_rows > 0:
        print(f"✅ Removed {removed_empty_rows} completely empty rows")
    
    # Step 2: Remove empty/mostly empty columns
    threshold = int(len(df) * min_column_fill)
    cols_before = len(df.columns)
    
    non_empty_counts = df.count()
    columns_to_keep = non_empty_counts[non_empty_counts >= threshold].index.tolist()
    
    removed_cols = [col for col in df.columns if col not in columns_to_keep]
    if removed_cols:
        print(f"✅ Removed {len(removed_cols)} mostly empty columns:")
        for col in removed_cols[:5]:  # Show first 5
            fill_pct = (df[col].count() / len(df)) * 100
            print(f"   - '{col}' ({fill_pct:.1f}% filled)")
        if len(removed_cols) > 5:
            print(f"   ... and {len(removed_cols) - 5} more")
    
    df = df[columns_to_keep]
    
    # Step 3: Standardize column names
    print(f"\n✅ Standardizing column names...")
    name_mapping = {col: standardize_column_name(col) for col in df.columns}
    changed_names = {old: new for old, new in name_mapping.items() if old != new}
    if changed_names:
        for old, new in list(changed_names.items())[:5]:
            print(f"   '{old}' → '{new}'")
        if len(changed_names) > 5:
            print(f"   ... and {len(changed_names) - 5} more")
    
    df.rename(columns=name_mapping, inplace=True)
    
    # Step 4: Fix phone numbers
    print(f"\n✅ Fixing phone numbers...")
    phone_cols = [col for col in df.columns if 'phone' in col or 'mobile' in col or 'tel' in col]
    fixed_count = 0
    
    for col in phone_cols:
        before = df[col].astype(str).str.contains('e\\+|e-', case=False, na=False).sum()
        df[col] = df[col].apply(fix_phone_number)
        after = df[col].astype(str).str.contains('e\\+|e-', case=False, na=False).sum()
        fixed = before - after
        if fixed > 0:
            print(f"   Fixed {fixed} numbers in '{col}'")
            fixed_count += fixed
    
    if fixed_count == 0:
        print(f"   No scientific notation found ✓")
    
    # Step 5: Trim whitespace from all text fields
    print(f"\n✅ Trimming whitespace...")
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.strip() if df[col].dtype == 'object' else df[col]
    
    # Step 6: Remove duplicate rows
    duplicates_before = df.duplicated().sum()
    if duplicates_before > 0:
        df = df.drop_duplicates()
        print(f"\n✅ Removed {duplicates_before} duplicate rows")
    
    # Step 7: Optional - Merge duplicate contacts by name
    if merge_duplicates and 'name' in df.columns:
        print(f"\n✅ Merging contacts with duplicate names...")
        
        # Group by name and merge
        grouped = df.groupby('name', as_index=False).agg(
            lambda x: x.dropna().iloc[0] if len(x.dropna()) > 0 else None
        )
        
        merged_count = len(df) - len(grouped)
        if merged_count > 0:
            print(f"   Merged {merged_count} duplicate contacts")
            df = grouped
        else:
            print(f"   No duplicate names found")
    
    # Step 8: Replace NaN with empty strings
    df = df.fillna('')
    
    # Final stats
    print(f"\n" + "="*50)
    print(f"📊 SUMMARY:")
    print(f"   Rows: {original_rows} → {len(df)} ({original_rows - len(df)} removed)")
    print(f"   Columns: {original_cols} → {len(df.columns)} ({original_cols - len(df.columns)} removed)")
    print(f"   Final size: {len(df)} contacts, {len(df.columns)} fields")
    print(f"="*50)
    
    # Save cleaned file
    df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"\n✅ Saved clean file: {output_file}")
    
    # Show sample of cleaned data
    print(f"\n📋 First 3 rows preview:")
    print(df.head(3).to_string(max_colwidth=30))
    
    return df

def main():
    if len(sys.argv) < 3:
        print("Usage: python preprocess_csv.py input.csv output.csv [--merge-duplicates]")
        print()
        print("Options:")
        print("  --merge-duplicates    Merge contacts with same name")
        print("  --strict              Remove columns with <20% data (default: 5%)")
        print()
        print("Example:")
        print("  python preprocess_csv.py messy_contacts.csv clean_contacts.csv")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    merge_duplicates = '--merge-duplicates' in sys.argv
    strict = '--strict' in sys.argv
    
    min_fill = 0.20 if strict else 0.05
    
    try:
        preprocess_csv(input_file, output_file, merge_duplicates, min_fill)
        print(f"\n🎉 Done! Upload '{output_file}' to your chatbot.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
