#!/usr/bin/env python3
"""
CSV Column Inspector - See what columns are in your CSV
"""
import pandas as pd
import sys

if len(sys.argv) < 2:
    print("Usage: python3 inspect_csv.py <path_to_csv>")
    print("Example: python3 inspect_csv.py 'contacts (3).csv'")
    sys.exit(1)

csv_path = sys.argv[1]

print("\n" + "="*60)
print(f"CSV COLUMN INSPECTOR: {csv_path}")
print("="*60 + "\n")

df = pd.read_csv(csv_path)

print(f"Total columns: {len(df.columns)}")
print(f"Total rows: {len(df)}\n")

print("ALL COLUMNS:")
print("-" * 60)
for i, col in enumerate(df.columns, 1):
    print(f"{i:3}. {col}")

print("\n" + "="*60)
print("PHONE-RELATED COLUMNS:")
print("="*60)

phone_cols = [col for col in df.columns if 'phone' in col.lower() or 'mobile' in col.lower() or 'tel' in col.lower()]

if phone_cols:
    for col in phone_cols:
        non_null = df[col].notna().sum()
        sample_values = df[col].dropna().head(3).tolist()
        print(f"\n📱 {col}")
        print(f"   Non-null values: {non_null}/{len(df)}")
        print(f"   Sample values: {sample_values}")
else:
    print("❌ No phone-related columns found!")

print("\n" + "="*60)
