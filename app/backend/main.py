import sys
from contextlib import asynccontextmanager
from pathlib import Path

# Make sure imports resolve when run from app/backend/
sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import DATA_ROOT, INPUT_IMAGES_DIR, INPUT_VIDEOS_DIR, PORT, SPLITS
from db.client import create_db_and_tables
from routes.crops import router as crops_router
from routes.files import router as files_router
from routes.label import router as label_router
from routes.scan import router as scan_router
from routes.serve import router as serve_router


def _migrate(engine):
    """Add columns that may not exist in older DBs, and fix NOT NULL on annotations.file_id."""
    from sqlalchemy import text
    with engine.connect() as conn:
        # Add labeled_at to crops if missing
        try:
            conn.execute(text("ALTER TABLE crops ADD COLUMN labeled_at DATETIME"))
            conn.commit()
        except Exception:
            pass

        # Add crop_id to annotations if missing
        try:
            conn.execute(text("ALTER TABLE annotations ADD COLUMN crop_id INTEGER"))
            conn.commit()
        except Exception:
            pass

        # Fix file_id NOT NULL constraint by recreating the table if needed.
        # Check whether file_id is still NOT NULL.
        row = conn.execute(text("PRAGMA table_info(annotations)")).fetchall()
        file_id_col = next((r for r in row if r[1] == "file_id"), None)
        # r[3] is the 'notnull' flag (1 = NOT NULL)
        if file_id_col and file_id_col[3] == 1:
            conn.execute(text("""
                CREATE TABLE annotations_new (
                    id INTEGER PRIMARY KEY,
                    file_id INTEGER,
                    crop_id INTEGER,
                    points_json TEXT NOT NULL DEFAULT '[]',
                    updated_at DATETIME
                )
            """))
            conn.execute(text("""
                INSERT INTO annotations_new (id, file_id, crop_id, points_json, updated_at)
                SELECT id, file_id, crop_id, points_json, updated_at FROM annotations
            """))
            conn.execute(text("DROP TABLE annotations"))
            conn.execute(text("ALTER TABLE annotations_new RENAME TO annotations"))
            conn.commit()


@asynccontextmanager
async def lifespan(app: FastAPI):
    INPUT_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    INPUT_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        (DATA_ROOT / split / "raw" / "images").mkdir(parents=True, exist_ok=True)
        (DATA_ROOT / split / "images").mkdir(parents=True, exist_ok=True)
        (DATA_ROOT / split / "annotations").mkdir(parents=True, exist_ok=True)
        (DATA_ROOT / split / "unlabelled" / "images").mkdir(parents=True, exist_ok=True)
    create_db_and_tables()
    from db.client import engine as _engine
    _migrate(_engine)
    yield


app = FastAPI(title="Salmon Enumeration API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(scan_router, prefix="/api")
app.include_router(files_router, prefix="/api")
app.include_router(crops_router, prefix="/api")
app.include_router(label_router, prefix="/api")
app.include_router(serve_router, prefix="/api")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=True)
