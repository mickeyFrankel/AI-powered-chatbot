#!/usr/bin/env python3
"""
Fix phone numbers in contacts.csv that are stored in scientific notation
Converts: 9.73E+11 → 972-xxx-xxxx (or keeps as is if not scientific)
"""

import pandas as pd
import re
from pathlib import Path

def fix_scientific_notation(value):
    """Convert scientific notation to proper phone number format"""
    if pd.isna(value):
        return value
    
    # Convert to string
    value_str = str(value)
    
    # Check if it's in scientific notation (e.g., 9.73E+11)
    if 'E+' in value_str or 'e+' in value_str:
        try:
            # Convert to integer (removes scientific notation)
            number = int(float(value_str))
            
            # Convert to string without decimals
            phone_str = str(number)
            
            # Format as phone number if it looks like Israeli format
            if phone_str.startswith('972') and len(phone_str) >= 11:
                # Format: 972-xx-xxx-xxxx or similar
                # Remove leading 972 and format the rest
                local = phone_str[3:]
                if len(local) == 9:
                    # Format as xxx-xxx-xxxx
                    formatted = f"{local[0:3]}-{local[3:6]}-{local[6:9]}"
                    return formatted
                elif len(local) == 8:
                    # Format as xx-xxx-xxxx
                    formatted = f"{local[0:2]}-{local[2:5]}-{local[5:8]}"
                    return formatted
                else:
                    return phone_str
            else:
                return phone_str
                
        except (ValueError, OverflowError):
            return value_str
    
    return value_str

def fix_contacts_csv(input_file='contacts.csv', output_file='contacts_fixed.csv'):
    """Fix all phone numbers in the CSV file"""
    
    print("🔧 Fixing Phone Numbers in CSV")
    print("=" * 60)
    
    # Read CSV
    print(f"📖 Reading: {input_file}")
    df = pd.read_csv(input_file, dtype=str)  # Read as strings to preserve formatting
    
    print(f"   Found {len(df)} contacts")
    
    # Find all phone number columns
    phone_columns = [col for col in df.columns if 'Phone' in col and 'Value' in col]
    print(f"   Phone columns: {len(phone_columns)}")
    
    # Count how many fixes needed
    fixes_made = 0
    
    # Fix each phone column
    for col in phone_columns:
        if col in df.columns:
            print(f"\n🔍 Checking column: {col}")
            
            # Count scientific notation entries
            scientific_count = df[col].astype(str).str.contains('E\+|e\+', na=False).sum()
            
            if scientific_count > 0:
                print(f"   Found {scientific_count} numbers in scientific notation")
                
                # Apply fix
                df[col] = df[col].apply(fix_scientific_notation)
                fixes_made += scientific_count
                print(f"   ✅ Fixed {scientific_count} entries")
            else:
                print(f"   ✓ No fixes needed")
    
    # Save fixed CSV
    print(f"\n💾 Saving fixed CSV to: {output_file}")
    df.to_csv(output_file, index=False)
    
    print("\n" + "=" * 60)
    print(f"✅ Complete! Fixed {fixes_made} phone numbers")
    print(f"📄 Original file: {input_file}")
    print(f"📄 Fixed file: {output_file}")
    print("\n💡 Next steps:")
    print("   1. Review the fixed file")
    print("   2. If good, replace original: mv contacts_fixed.csv contacts.csv")
    print("   3. Reload in chatbot: load contacts.csv")
    
    return fixes_made

if __name__ == "__main__":
    import sys
    
    # Get file paths from command line or use defaults
    input_file = sys.argv[1] if len(sys.argv) > 1 else 'contacts.csv'
    output_file = sys.argv[2] if len(sys.argv) > 2 else 'contacts_fixed.csv'
    
    try:
        fixes = fix_contacts_csv(input_file, output_file)
        
        if fixes > 0:
            print("\n🎉 Success! Your phone numbers are fixed!")
        else:
            print("\n✓ No fixes needed - all phone numbers look good!")
            
    except FileNotFoundError:
        print(f"❌ Error: Could not find {input_file}")
        print("   Make sure you're in the right directory")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
