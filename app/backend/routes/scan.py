from __future__ import annotations
import hashlib
import uuid as uuid_mod

from typing import List

from fastapi import APIRouter, Depends, File, UploadFile
from sqlmodel import Session, select

from config import DATA_ROOT, INPUT_IMAGES_DIR
from db.client import get_session
from db.models import DBFile
from lib.scanner import IMAGE_EXTS, VIDEO_EXTS, scan_input

router = APIRouter()


@router.post("/scan")
def scan(session: Session = Depends(get_session)):
    return scan_input(session, DATA_ROOT)


@router.get("/input-status")
def input_status(session: Session = Depends(get_session)):
    on_disk = sum(
        1 for p in INPUT_IMAGES_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ) if INPUT_IMAGES_DIR.exists() else 0
    ingested = session.exec(select(DBFile).where(DBFile.status == "input")).all()
    return {"on_disk": on_disk, "ingested": len(ingested)}


@router.post("/upload")
async def upload(
    files: List[UploadFile] = File(...),
    session: Session = Depends(get_session),
):
    INPUT_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    saved = []
    duplicates = []

    for upload in files:
        data = await upload.read()

        sha = hashlib.sha256(data).hexdigest()
        existing = session.exec(
            select(DBFile).where(DBFile.sha256 == sha)
        ).first()

        suffix = "." + (upload.filename or "file.jpg").rsplit(".", 1)[-1].lower()
        if suffix not in IMAGE_EXTS:
            suffix = ".jpg"

        dest_name = f"{uuid_mod.uuid4()}{suffix}"
        dest = INPUT_IMAGES_DIR / dest_name
        dest.write_bytes(data)

        if existing:
            duplicates.append({
                "original_name": upload.filename,
                "duplicate_of": existing.original_name,
                "saved_as": dest_name,
            })
        else:
            saved.append({"original_name": upload.filename, "saved_as": dest_name})

    return {"saved": saved, "duplicates": duplicates}
