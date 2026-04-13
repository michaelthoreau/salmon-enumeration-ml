import hashlib
import uuid
from pathlib import Path

from sqlmodel import Session, select

from config import INPUT_IMAGES_DIR, INPUT_VIDEOS_DIR
from db.models import DBFile

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def scan_input(session: Session, data_root: Path) -> dict:
    added = 0
    skipped = 0

    # Build fast lookup: known rel_paths already in DB
    known_paths = set(
        r.rel_path for r in session.exec(select(DBFile)).all()
    )

    dirs = [
        (INPUT_IMAGES_DIR, "image"),
        (INPUT_VIDEOS_DIR, "video"),
    ]

    for dir_path, file_type in dirs:
        if not dir_path.exists():
            continue
        exts = IMAGE_EXTS if file_type == "image" else VIDEO_EXTS
        for path in sorted(dir_path.iterdir()):
            if path.suffix.lower() not in exts:
                continue

            rel = str(path.relative_to(data_root))

            # Fast path: already in DB by path
            if rel in known_paths:
                skipped += 1
                continue

            # Hash only for new files (dedup check)
            sha = _sha256(path)
            existing = session.exec(
                select(DBFile).where(DBFile.sha256 == sha)
            ).first()

            if existing:
                skipped += 1
                continue

            record = DBFile(
                uuid=str(uuid.uuid4()),
                type=file_type,
                original_name=path.name,
                sha256=sha,
                status="input",
                rel_path=rel,
            )
            session.add(record)
            added += 1

    session.commit()
    return {"added": added, "skipped": skipped}
