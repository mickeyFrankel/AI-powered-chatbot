# How to Replace Contacts Database

## Problem: Phone Numbers in Scientific Notation

Excel converts phone numbers like `0542227884` to `9.73E+11`

## Solution Options

### Option A: Export Properly from Excel (BEST)

**Before exporting:**
1. Select all phone number columns
2. Right-click → Format Cells → Text
3. Save As → CSV UTF-8

**Or add apostrophe:**
- Type: `'0542227884` (starts with apostrophe)
- Excel will treat it as text

---

### Option B: Fix Existing CSV

```bash
# Fix phone numbers in your CSV
python fix_phone_csv.py your_contacts.csv fixed_contacts.csv
```

This converts `9.73E+11` → `0542227884`

---

## Replace Database

### Method 1: One Command (Recommended)

```bash
# Make scripts executable
chmod +x clear_db.sh replace_db.sh

# Fix CSV first (if needed)
python fix_phone_csv.py contacts.csv fixed_contacts.csv

# Replace database in one step
./replace_db.sh fixed_contacts.csv

# Restart servers
./start.sh
```

### Method 2: Manual Steps

```bash
# 1. Clear old database
./clear_db.sh

# 2. Load new CSV
python3 << 'EOF'
from vectoric_search import AdvancedVectorDBQASystem

qa = AdvancedVectorDBQASystem(persist_directory="./chroma_db")
qa.ingest_file("your_contacts.csv")

stats = qa.get_collection_stats()
print(f"Loaded {stats['document_count']} contacts")
EOF

# 3. Restart servers
./start.sh
```

---

## Verify Phone Numbers

After loading, test:
```
Query: "phone number of [contact name]"
Expected: 054-xxx-xxxx (not 9.73E+11)
```

---

## Files Created

- `clear_db.sh` - Delete current database
- `fix_phone_csv.py` - Fix phone format in CSV
- `replace_db.sh` - Complete replacement workflow
- `REPLACE_DATABASE.md` - This guide

---

## Troubleshooting

**Phone still showing as 9.73E+11?**
- CSV wasn't fixed before loading
- Run: `python fix_phone_csv.py` first

**Database not clearing?**
- Manually delete: `rm -rf chroma_db/ contacts_db/`

**CSV not loading?**
- Check encoding: Must be UTF-8
- Check file path: Use absolute path if needed
