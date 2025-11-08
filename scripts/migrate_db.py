import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.models.base import engine
from sqlalchemy import text

def migrate():
    """Add missing columns to existing tables"""
    with engine.begin() as conn:
        try:
            # Add source_type column to journals table
            conn.execute(text(
                "ALTER TABLE journals ADD COLUMN source_type VARCHAR(32) DEFAULT 'openalex'"
            ))
            print("✓ Added source_type column")
        except Exception as e:
            if "duplicate column" in str(e).lower() or "already exists" in str(e).lower():
                print("✓ source_type column already exists")
            else:
                print(f"✗ Error adding source_type: {e}")
        
        try:
            # Add external_id column to journals table
            conn.execute(text(
                "ALTER TABLE journals ADD COLUMN external_id VARCHAR(128)"
            ))
            print("✓ Added external_id column")
        except Exception as e:
            if "duplicate column" in str(e).lower() or "already exists" in str(e).lower():
                print("✓ external_id column already exists")
            else:
                print(f"✗ Error adding external_id: {e}")
        
        try:
            # Add eissn column to journals table
            conn.execute(text(
                "ALTER TABLE journals ADD COLUMN eissn VARCHAR(32)"
            ))
            print("✓ Added eissn column")
        except Exception as e:
            if "duplicate column" in str(e).lower() or "already exists" in str(e).lower():
                print("✓ eissn column already exists")
            else:
                print(f"✗ Error adding eissn: {e}")
        
        try:
            # Add impact_factor column to journals table
            conn.execute(text(
                "ALTER TABLE journals ADD COLUMN impact_factor FLOAT"
            ))
            print("✓ Added impact_factor column")
        except Exception as e:
            if "duplicate column" in str(e).lower() or "already exists" in str(e).lower():
                print("✓ impact_factor column already exists")
            else:
                print(f"✗ Error adding impact_factor: {e}")
        
        try:
            # Create index on source_type
            conn.execute(text(
                "CREATE INDEX IF NOT EXISTS idx_source_type ON journals(source_type)"
            ))
            print("✓ Created index on source_type")
        except Exception as e:
            print(f"✗ Error creating index: {e}")
        
        try:
            # Create index on external_id
            conn.execute(text(
                "CREATE INDEX IF NOT EXISTS idx_external_id ON journals(external_id)"
            ))
            print("✓ Created index on external_id")
        except Exception as e:
            print(f"✗ Error creating index: {e}")
        
        try:
            # Create index on eissn
            conn.execute(text(
                "CREATE INDEX IF NOT EXISTS idx_eissn ON journals(eissn)"
            ))
            print("✓ Created index on eissn")
        except Exception as e:
            print(f"✗ Error creating index: {e}")
    
    print("\nMigration completed!")

if __name__ == "__main__":
    migrate()
