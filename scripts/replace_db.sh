#!/bin/bash
# Complete workflow to replace contacts database

echo "========================================="
echo "Replace Contacts Database"
echo "========================================="
echo ""

# Step 1: Clear old database
echo "Step 1: Clearing old database..."
rm -rf chroma_db/
rm -rf contacts_db/
echo "✅ Old database cleared"
echo ""

# Step 2: Check if new CSV provided
if [ -z "$1" ]; then
    echo "Usage: ./replace_db.sh <new_contacts.csv>"
    echo ""
    echo "First, fix your CSV phone numbers:"
    echo "  python fix_phone_csv.py your_file.csv fixed_contacts.csv"
    echo ""
    echo "Then load it:"
    echo "  ./replace_db.sh fixed_contacts.csv"
    exit 1
fi

NEW_CSV="$1"

# Step 3: Activate venv
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Step 4: Load new data
echo "Step 2: Loading new CSV: $NEW_CSV"
python3 << EOF
from vectoric_search import AdvancedVectorDBQASystem

# Initialize system (creates fresh DB)
qa = AdvancedVectorDBQASystem(persist_directory="./chroma_db")

# Ingest the CSV
print(f"Loading: $NEW_CSV")
qa.ingest_file("$NEW_CSV")

# Show stats
stats = qa.get_collection_stats()
print(f"\n✅ Loaded {stats['document_count']} contacts")
print(f"✅ Database ready!")
EOF

echo ""
echo "========================================="
echo "Database replacement complete!"
echo "Restart servers with: ./start.sh"
echo "========================================="
