#!/usr/bin/env python3
"""
Contacts CSV Preprocessor - Clean and optimize contacts data for efficient search
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from collections import Counter

def analyze_csv(df):
    """Analyze the CSV to show what will be cleaned"""
    print("\n" + "="*70)
    print("📊 ANALYSIS REPORT")
    print("="*70)
    
    total_rows = len(df)
    total_cols = len(df.columns)
    
    print(f"\n📋 Original: {total_rows:,} rows × {total_cols} columns")
    print(f"📦 File size impact: ~{df.memory_usage(deep=True).sum() / 1024:.1f} KB")
    
    # Analyze columns
    empty_cols = []
    sparse_cols = []
    duplicate_cols = []
    
    print("\n🔍 Column Analysis:")
    print("-" * 70)
    
    for col in df.columns:
        non_null = df[col].notna().sum()
        fill_rate = (non_null / total_rows) * 100
        
        if non_null == 0:
            empty_cols.append(col)
            print(f"  ❌ {col:30s} - EMPTY (0%)")
        elif fill_rate < 5:
            sparse_cols.append((col, fill_rate))
            print(f"  ⚠️  {col:30s} - SPARSE ({fill_rate:.1f}%)")
        elif fill_rate < 50:
            print(f"  📉 {col:30s} - LOW ({fill_rate:.1f}%)")
        else:
            print(f"  ✓  {col:30s} - OK ({fill_rate:.1f}%)")
    
    # Check for duplicate columns
    for i, col1 in enumerate(df.columns):
        for col2 in df.columns[i+1:]:
            if df[col1].equals(df[col2]):
                duplicate_cols.append((col1, col2))
    
    # Check for duplicate rows
    duplicate_rows = df.duplicated().sum()
    
    # Check for empty rows
    empty_rows = df.isna().all(axis=1).sum()
    
    print("\n" + "="*70)
    print("🧹 CLEANUP RECOMMENDATIONS:")
    print("="*70)
    
    if empty_cols:
        print(f"\n❌ Remove {len(empty_cols)} completely empty columns:")
        for col in empty_cols[:5]:
            print(f"   • {col}")
        if len(empty_cols) > 5:
            print(f"   ... and {len(empty_cols) - 5} more")
    
    if sparse_cols:
        print(f"\n⚠️  Consider removing {len(sparse_cols)} sparse columns (<5% filled):")
        for col, rate in sparse_cols[:5]:
            print(f"   • {col} ({rate:.1f}%)")
        if len(sparse_cols) > 5:
            print(f"   ... and {len(sparse_cols) - 5} more")
    
    if duplicate_cols:
        print(f"\n🔁 Remove {len(duplicate_cols)} duplicate columns:")
        for col1, col2 in duplicate_cols[:3]:
            print(f"   • {col1} = {col2}")
        if len(duplicate_cols) > 3:
            print(f"   ... and {len(duplicate_cols) - 3} more")
    
    if duplicate_rows > 0:
        print(f"\n🔁 Remove {duplicate_rows:,} duplicate rows")
    
    if empty_rows > 0:
        print(f"\n❌ Remove {empty_rows:,} completely empty rows")
    
    return {
        'empty_cols': empty_cols,
        'sparse_cols': [col for col, _ in sparse_cols],
        'duplicate_cols': [col2 for _, col2 in duplicate_cols],
        'duplicate_rows': duplicate_rows,
        'empty_rows': empty_rows
    }

def clean_csv(df, remove_sparse=True, sparse_threshold=5.0):
    """Clean the CSV by removing empty/sparse columns and duplicate rows"""
    
    print("\n" + "="*70)
    print("🧹 CLEANING IN PROGRESS...")
    print("="*70)
    
    original_rows = len(df)
    original_cols = len(df.columns)
    
    # Step 1: Remove completely empty columns
    empty_cols = df.columns[df.isna().all()].tolist()
    if empty_cols:
        df = df.drop(columns=empty_cols)
        print(f"\n✓ Removed {len(empty_cols)} empty columns")
    
    # Step 2: Remove sparse columns (optional)
    if remove_sparse:
        sparse_cols = []
        for col in df.columns:
            fill_rate = (df[col].notna().sum() / len(df)) * 100
            if fill_rate < sparse_threshold:
                sparse_cols.append(col)
        
        if sparse_cols:
            df = df.drop(columns=sparse_cols)
            print(f"✓ Removed {len(sparse_cols)} sparse columns (<{sparse_threshold}% filled)")
    
    # Step 3: Remove duplicate columns
    duplicate_cols = []
    cols_to_check = df.columns.tolist()
    for i, col1 in enumerate(cols_to_check):
        if col1 not in df.columns:  # Already removed
            continue
        for col2 in cols_to_check[i+1:]:
            if col2 in df.columns and df[col1].equals(df[col2]):
                duplicate_cols.append(col2)
                df = df.drop(columns=[col2])
    
    if duplicate_cols:
        print(f"✓ Removed {len(duplicate_cols)} duplicate columns")
    
    # Step 4: Remove completely empty rows
    empty_rows = df.isna().all(axis=1).sum()
    if empty_rows > 0:
        df = df.dropna(how='all')
        print(f"✓ Removed {empty_rows} completely empty rows")
    
    # Step 5: Remove duplicate rows
    duplicate_rows = df.duplicated().sum()
    if duplicate_rows > 0:
        df = df.drop_duplicates()
        print(f"✓ Removed {duplicate_rows} duplicate rows")
    
    # Step 6: Strip whitespace from all string columns
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
    print(f"✓ Stripped whitespace from text fields")
    
    # Step 7: Replace empty strings with NaN for consistency
    df = df.replace('', np.nan)
    
    print("\n" + "="*70)
    print("✅ CLEANUP COMPLETE")
    print("="*70)
    print(f"\nBefore: {original_rows:,} rows × {original_cols} columns")
    print(f"After:  {len(df):,} rows × {len(df.columns)} columns")
    print(f"\nRemoved: {original_rows - len(df):,} rows, {original_cols - len(df.columns)} columns")
    print(f"Reduction: {((1 - len(df)*len(df.columns)/(original_rows*original_cols)) * 100):.1f}% smaller")
    
    return df

def smart_consolidate(df):
    """Intelligently consolidate similar columns (phone, email, etc.)"""
    print("\n" + "="*70)
    print("🔄 SMART CONSOLIDATION")
    print("="*70)
    
    # Find phone columns
    phone_cols = [col for col in df.columns if any(term in col.lower() for term in ['phone', 'mobile', 'tel', 'cellular'])]
    if len(phone_cols) > 1:
        print(f"\n📱 Found {len(phone_cols)} phone columns: {', '.join(phone_cols)}")
        print("   → Consolidating into 'phone'")
        
        # Create consolidated phone column (take first non-null value)
        df['phone'] = df[phone_cols].bfill(axis=1).iloc[:, 0]
        df = df.drop(columns=[col for col in phone_cols if col != 'phone'])
    
    # Find email columns
    email_cols = [col for col in df.columns if 'mail' in col.lower()]
    if len(email_cols) > 1:
        print(f"\n📧 Found {len(email_cols)} email columns: {', '.join(email_cols)}")
        print("   → Consolidating into 'email'")
        
        df['email'] = df[email_cols].bfill(axis=1).iloc[:, 0]
        df = df.drop(columns=[col for col in email_cols if col != 'email'])
    
    # Find address columns
    address_cols = [col for col in df.columns if any(term in col.lower() for term in ['address', 'street', 'city', 'כתובת'])]
    if len(address_cols) > 1:
        print(f"\n🏠 Found {len(address_cols)} address columns: {', '.join(address_cols)}")
        print("   → Consolidating into 'address'")
        
        # Combine address fields into one
        df['address'] = df[address_cols].apply(
            lambda row: ' '.join([str(x) for x in row if pd.notna(x)]), axis=1
        )
        df['address'] = df['address'].replace('', np.nan)
        df = df.drop(columns=address_cols)
    
    return df

def main():
    """Main preprocessing workflow"""
    print("\n" + "="*70)
    print("🧹 CONTACTS CSV PREPROCESSOR")
    print("="*70)
    
    # Get input file
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        input_file = input("\nEnter path to contacts CSV file: ").strip()
    
    input_path = Path(input_file)
    
    if not input_path.exists():
        print(f"\n❌ Error: File not found: {input_file}")
        sys.exit(1)
    
    # Read CSV
    print(f"\n📖 Reading: {input_path.name}")
    try:
        df = pd.read_csv(input_path, encoding='utf-8')
    except Exception as e:
        print(f"\n❌ Error reading CSV: {e}")
        sys.exit(1)
    
    # Analyze
    analysis = analyze_csv(df)
    
    # Ask user for confirmation
    print("\n" + "="*70)
    response = input("\n🚀 Proceed with cleanup? [Y/n]: ").strip().lower()
    
    if response and response not in ['y', 'yes']:
        print("\n❌ Cancelled.")
        sys.exit(0)
    
    # Clean
    df_clean = clean_csv(df, remove_sparse=True, sparse_threshold=5.0)
    
    # Consolidate
    consolidate = input("\n🔄 Apply smart consolidation (merge phone/email columns)? [Y/n]: ").strip().lower()
    if not consolidate or consolidate in ['y', 'yes']:
        df_clean = smart_consolidate(df_clean)
    
    # Save
    output_file = input_path.parent / f"{input_path.stem}_cleaned{input_path.suffix}"
    df_clean.to_csv(output_file, index=False, encoding='utf-8')
    
    print("\n" + "="*70)
    print("✅ SUCCESS!")
    print("="*70)
    print(f"\n💾 Saved to: {output_file}")
    print(f"📊 Final: {len(df_clean):,} rows × {len(df_clean.columns)} columns")
    
    # Show column summary
    print(f"\n📋 Remaining columns ({len(df_clean.columns)}):")
    for col in df_clean.columns:
        fill_rate = (df_clean[col].notna().sum() / len(df_clean)) * 100
        print(f"   • {col:30s} ({fill_rate:.1f}% filled)")
    
    print("\n✨ Ready to ingest into your chatbot!")
    print(f"   Use: Upload CSV → Select '{output_file.name}'")

if __name__ == "__main__":
    main()
