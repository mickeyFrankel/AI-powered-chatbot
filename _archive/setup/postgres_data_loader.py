#!/usr/bin/env python3
import psycopg2
import pandas as pd
import json
import sys
from pathlib import Path

def load_file(file_path, table_name='data', schema='chatbot'):
    conn_params = {
        'host': 'localhost',
        'port': 5432,
        'dbname': 'chatbot_db',
        'user': 'chatbot_user',
        'password': 'chatbot_password'
    }
    
    path = Path(file_path)
    if not path.exists():
        print(f"✗ File not found: {file_path}")
        return False
    
    if path.suffix == '.csv':
        df = pd.read_csv(file_path)
    elif path.suffix == '.json':
        with open(file_path) as f:
            data = json.load(f)
        df = pd.DataFrame(data if isinstance(data, list) else [data])
    else:
        print(f"✗ Unsupported file type")
        return False
    
    # Clean column names - remove spaces and special characters
    df.columns = df.columns.str.replace(' ', '_').str.replace('-', '_').str.lower()
    
    print(f"Read {len(df)} rows from {file_path}")
    
    conn = psycopg2.connect(**conn_params)
    cursor = conn.cursor()
    
    # Drop table if exists and recreate
    cursor.execute(f"DROP TABLE IF EXISTS {schema}.{table_name};")
    
    type_map = {'int64': 'INTEGER', 'float64': 'FLOAT', 'object': 'TEXT', 'bool': 'BOOLEAN'}
    columns = [f'"{col}" {type_map.get(str(dtype), "TEXT")}' for col, dtype in df.dtypes.items()]
    
    create_stmt = f"CREATE TABLE {schema}.{table_name} (id SERIAL PRIMARY KEY, {', '.join(columns)});"
    cursor.execute(create_stmt)
    
    cols = ', '.join([f'"{col}"' for col in df.columns])
    placeholders = ', '.join(['%s'] * len(df.columns))
    insert_stmt = f'INSERT INTO {schema}.{table_name} ({cols}) VALUES ({placeholders})'
    
    for _, row in df.iterrows():
        cursor.execute(insert_stmt, tuple(row))
    
    conn.commit()
    print(f"✓ Loaded {len(df)} rows into {schema}.{table_name}")
    
    cursor.close()
    conn.close()
    return True

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('file')
    parser.add_argument('--table', default=None)
    args = parser.parse_args()
    
    table_name = args.table or Path(args.file).stem.lower().replace(' ', '_')
    load_file(args.file, table_name)
