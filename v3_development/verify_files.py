#!/usr/bin/env python3
"""
Verify all 3 files exist and are complete
"""

from pathlib import Path

files = {
    'config.py': 190,
    'models.py': 323,
    'file_loaders.py': 481
}

print("\n" + "="*60)
print("FILE VERIFICATION")
print("="*60)

all_good = True

for filename, expected_lines in files.items():
    path = Path(filename)
    
    if not path.exists():
        print(f"❌ {filename} - NOT FOUND")
        all_good = False
        continue
    
    actual_lines = len(path.read_text().splitlines())
    size = path.stat().st_size
    
    if actual_lines >= expected_lines - 5:  # Allow small variance
        print(f"✅ {filename} - {actual_lines} lines, {size:,} bytes")
    else:
        print(f"⚠️  {filename} - Only {actual_lines} lines (expected ~{expected_lines})")
        all_good = False

print("="*60)

if all_good:
    print("✅ ALL FILES PRESENT AND COMPLETE!")
    print("\nTo test imports:")
    print("  python3 -c 'import config, models, file_loaders; print(\"✅ Imports work!\")'")
else:
    print("⚠️  Some files missing or incomplete")

print("="*60 + "\n")
