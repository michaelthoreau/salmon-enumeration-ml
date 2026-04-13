from __future__ import annotations
import shutil
import uuid
from pathlib import Path

from sqlmodel import Session, select

from config import DATA_ROOT
from db.models import DBFile


def assign_files(session: Session, file_ids: list[int], split: str) -> dict:
    """Move image files from input/ into {split}/raw/images/ and update DB."""
    assigned = 0
    errors = []

    dest_dir = DATA_ROOT / split / "raw" / "images"
    dest_dir.mkdir(parents=True, exist_ok=True)

    for fid in file_ids:
        record = session.get(DBFile, fid)
        if not record:
            errors.append(f"file {fid} not found")
            continue
        if record.type != "image":
            errors.append(f"file {fid} is a video — use extract-frames instead")
            continue

        src = DATA_ROOT / record.rel_path
        if not src.exists():
            errors.append(f"file {fid} missing on disk: {src}")
            continue

        suffix = src.suffix.lower()
        new_name = f"{uuid.uuid4()}{suffix}"
        dest = dest_dir / new_name

        shutil.copy2(src, dest)

        record.status = "assigned"
        record.split = split
        record.rel_path = str(dest.relative_to(DATA_ROOT))
        session.add(record)
        assigned += 1

    session.commit()
    return {"assigned": assigned, "errors": errors}


def unassign_files(session: Session, file_ids: list[int]) -> dict:
    """Move files back to input/images/ and reset status to 'input'."""
    unassigned = 0
    errors = []

    dest_dir = DATA_ROOT / "input" / "images"
    dest_dir.mkdir(parents=True, exist_ok=True)

    for fid in file_ids:
        record = session.get(DBFile, fid)
        if not record:
            errors.append(f"file {fid} not found")
            continue

        src = DATA_ROOT / record.rel_path
        if src.exists():
            suffix = src.suffix.lower()
            dest = dest_dir / f"{uuid.uuid4()}{suffix}"
            shutil.move(str(src), dest)
            record.rel_path = str(dest.relative_to(DATA_ROOT))
        # Even if file is missing on disk, reset DB status
        record.status = "input"
        record.split = None
        session.add(record)
        unassigned += 1

    session.commit()
    return {"unassigned": unassigned, "errors": errors}
