#!/bin/bash
# Clear the current ChromaDB database

echo "⚠️  This will DELETE all current contacts!"
read -p "Are you sure? (type 'yes' to confirm): " confirm

if [ "$confirm" != "yes" ]; then
    echo "Cancelled."
    exit 0
fi

# Remove the database
rm -rf chroma_db/
echo "✅ Database cleared!"

# Remove any PostgreSQL data if exists
if [ -d "contacts_db" ]; then
    rm -rf contacts_db/
    echo "✅ PostgreSQL data cleared!"
fi

echo ""
echo "Now you can load your new CSV file."
