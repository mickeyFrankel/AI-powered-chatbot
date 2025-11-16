#!/usr/bin/env python3
"""
Fix phone numbers in CSV that Excel converted to scientific notation
Usage: python fix_phone_csv.py input.csv output.csv
"""
import sys
import csv
import re

def fix_scientific_notation(value):
    """Convert scientific notation back to phone number"""
    if not value or value == '':
        return value
    
    # Check if it's scientific notation (e.g., 9.73E+11)
    if 'E+' in str(value).upper() or 'e+' in str(value):
        try:
            # Convert to integer
            num = float(value)
            # Format as integer string (removes scientific notation)
            result = f"{int(num)}"
            
            # Israeli phone format: should be 9-10 digits starting with 0
            if len(result) == 9 or len(result) == 10:
                if not result.startswith('0'):
                    result = '0' + result
            
            return result
        except (ValueError, OverflowError):
            return value
    
    return value

def fix_csv(input_file, output_file):
    """Process CSV and fix phone numbers"""
    with open(input_file, 'r', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames
        
        rows = []
        for row in reader:
            # Fix all phone/mobile fields
            for field in row:
                if 'phone' in field.lower() or 'mobile' in field.lower() or 'tel' in field.lower():
                    row[field] = fix_scientific_notation(row[field])
            rows.append(row)
    
    # Write fixed CSV
    with open(output_file, 'w', encoding='utf-8', newline='') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"✅ Fixed {len(rows)} rows")
    print(f"✅ Saved to: {output_file}")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python fix_phone_csv.py input.csv output.csv")
        sys.exit(1)
    
    fix_csv(sys.argv[1], sys.argv[2])
