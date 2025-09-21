import sqlite3
import os
from pathlib import Path

db_path = Path("data/journal_rec.db")

print("Database File Analysis:")
print("=" * 40)
print(f"File exists: {db_path.exists()}")
if db_path.exists():
    print(f"File size: {db_path.stat().st_size} bytes")
    
    conn = sqlite3.connect(str(db_path))
    
    # Get tables
    tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    print(f"Tables: {[table[0] for table in tables]}")
    
    # Get table info
    for table in tables:
        table_name = table[0]
        count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        print(f"  {table_name}: {count} records")
    
    conn.close()
    print("\n✅ Database is healthy and accessible!")
else:
    print("❌ Database file not found!")
