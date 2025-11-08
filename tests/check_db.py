import sqlite3
import sys
from pathlib import Path

# Connect to the database
db_path = Path("data/journal_rec.db")
if not db_path.exists():
    print("Database file not found!")
    sys.exit(1)

conn = sqlite3.connect(str(db_path))
cursor = conn.cursor()

# Check table schema
print("Recommendations table schema:")
cursor.execute('PRAGMA table_info(recommendations)')
schema = cursor.fetchall()
for row in schema:
    print(f"  {row}")

# Get table creation SQL
print("\nTable creation SQL:")
cursor.execute('SELECT sql FROM sqlite_master WHERE type="table" AND name="recommendations"')
creation_sql = cursor.fetchone()
if creation_sql:
    print(creation_sql[0])

# Check for any existing data that might cause conflicts
print("\nChecking for existing recommendations:")
cursor.execute('SELECT COUNT(*) FROM recommendations')
count = cursor.fetchone()[0]
print(f"  Total recommendations: {count}")

if count > 0:
    print("\nSample recommendations:")
    cursor.execute('SELECT * FROM recommendations LIMIT 5')
    samples = cursor.fetchall()
    for row in samples:
        print(f"  {row}")

conn.close()