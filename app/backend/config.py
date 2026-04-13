from pathlib import Path

# Resolve paths relative to project root (two levels up from app/backend/)
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
DATA_ROOT = PROJECT_ROOT / "data"
DB_PATH = DATA_ROOT / "salmon.db"

INPUT_IMAGES_DIR = DATA_ROOT / "input" / "images"
INPUT_VIDEOS_DIR = DATA_ROOT / "input" / "videos"

SPLITS = ("train", "val", "test")

PORT = 8000
