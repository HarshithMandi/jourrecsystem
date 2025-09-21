import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.services.recommender import rank_journals
abstract = """studying the impacts of Artificial Intelligence in real world problems."""
print(rank_journals(abstract))
