from __future__ import annotations
import json
import shutil
from datetime import datetime
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlmodel import Session, select

from config import DATA_ROOT
from db.client import get_session
from db.models import DBAnnotation, DBCrop, DBFile

router = APIRouter()


# ---------------------------------------------------------------------------
# Crop label queue (used by the Label page)
# ---------------------------------------------------------------------------

@router.get("/label/crop-queue", response_model=List[DBCrop])
def crop_label_queue(split: str, session: Session = Depends(get_session)):
    return session.exec(
        select(DBCrop)
        .where(DBCrop.split == split)
        .where(DBCrop.labeled_at == None)  # noqa: E711
    ).all()


@router.get("/label/labeled-crops", response_model=List[DBCrop])
def labeled_crops(split: str, session: Session = Depends(get_session)):
    return session.exec(
        select(DBCrop)
        .where(DBCrop.split == split)
        .where(DBCrop.labeled_at != None)  # noqa: E711
    ).all()


@router.get("/label/crops/{crop_id}")
def get_crop_annotation(crop_id: int, session: Session = Depends(get_session)):
    ann = session.exec(
        select(DBAnnotation).where(DBAnnotation.crop_id == crop_id)
    ).first()
    points = json.loads(ann.points_json) if ann else []
    return {"crop_id": crop_id, "points": points}


class SaveBody(BaseModel):
    points: List[List[int]]


@router.put("/label/crops/{crop_id}")
def save_crop_annotation(
    crop_id: int, body: SaveBody, session: Session = Depends(get_session)
):
    ann = session.exec(
        select(DBAnnotation).where(DBAnnotation.crop_id == crop_id)
    ).first()
    if ann is None:
        ann = DBAnnotation(crop_id=crop_id, points_json=json.dumps(body.points))
    else:
        ann.points_json = json.dumps(body.points)
        ann.updated_at = datetime.utcnow()
    session.add(ann)
    session.commit()
    return {"ok": True, "count": len(body.points)}


@router.post("/label/crops/{crop_id}/commit")
def commit_crop_annotation(crop_id: int, session: Session = Depends(get_session)):
    crop = session.get(DBCrop, crop_id)
    if crop is None:
        raise HTTPException(status_code=404, detail="Crop not found")

    ann = session.exec(
        select(DBAnnotation).where(DBAnnotation.crop_id == crop_id)
    ).first()
    points = json.loads(ann.points_json) if ann else []

    src = DATA_ROOT / crop.rel_path
    if not src.exists():
        raise HTTPException(status_code=404, detail=f"Crop image not found: {src}")

    split = crop.split
    dest_dir = DATA_ROOT / split / "images"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name
    shutil.copy2(src, dest)

    try:
        from PIL import Image as PILImage
        with PILImage.open(src) as img:
            width, height = img.size
    except Exception:
        width, height = crop.w, crop.h

    # Write annotation JSON — named after the image file so image.py can find it
    ann_dir = DATA_ROOT / split / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)
    annotation = {
        "filename": src.name,
        "width": width,
        "height": height,
        "points": points,
    }
    (ann_dir / f"{src.stem}.json").write_text(json.dumps(annotation))

    crop.labeled_at = datetime.utcnow()
    session.add(crop)
    session.commit()

    return {"ok": True, "points": len(points), "path": str(dest.relative_to(DATA_ROOT))}


# ---------------------------------------------------------------------------
# Raw-file label queue (kept, not currently used by UI)
# ---------------------------------------------------------------------------

@router.get("/label/queue", response_model=List[DBFile])
def label_queue(split: str, session: Session = Depends(get_session)):
    return session.exec(
        select(DBFile)
        .where(DBFile.split == split)
        .where(DBFile.type == "image")
        .where(DBFile.status == "assigned")
    ).all()


@router.get("/label/{file_id}")
def get_annotation(file_id: int, session: Session = Depends(get_session)):
    ann = session.exec(
        select(DBAnnotation).where(DBAnnotation.file_id == file_id)
    ).first()
    points = json.loads(ann.points_json) if ann else []
    return {"file_id": file_id, "points": points}


@router.put("/label/{file_id}")
def save_annotation(
    file_id: int, body: SaveBody, session: Session = Depends(get_session)
):
    ann = session.exec(
        select(DBAnnotation).where(DBAnnotation.file_id == file_id)
    ).first()
    if ann is None:
        ann = DBAnnotation(file_id=file_id, points_json=json.dumps(body.points))
    else:
        ann.points_json = json.dumps(body.points)
        ann.updated_at = datetime.utcnow()
    session.add(ann)
    session.commit()
    return {"ok": True, "count": len(body.points)}


@router.post("/label/{file_id}/commit")
def commit_annotation(file_id: int, session: Session = Depends(get_session)):
    file = session.get(DBFile, file_id)
    if file is None:
        raise HTTPException(status_code=404, detail="File not found")
    if file.status != "assigned":
        raise HTTPException(status_code=400, detail=f"File status is '{file.status}', expected 'assigned'")

    ann = session.exec(
        select(DBAnnotation).where(DBAnnotation.file_id == file_id)
    ).first()
    points = json.loads(ann.points_json) if ann else []

    src = DATA_ROOT / file.rel_path
    if not src.exists():
        raise HTTPException(status_code=404, detail=f"Image not found: {src}")

    split = file.split
    dest_dir = DATA_ROOT / split / "images"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name
    shutil.copy2(src, dest)

    try:
        from PIL import Image as PILImage
        with PILImage.open(src) as img:
            width, height = img.size
    except Exception:
        width, height = 0, 0

    ann_dir = DATA_ROOT / split / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)
    annotation = {"filename": src.name, "width": width, "height": height, "points": points}
    (ann_dir / f"{file.uuid}.json").write_text(json.dumps(annotation))

    file.rel_path = str(dest.relative_to(DATA_ROOT))
    file.status = "labeled"
    session.add(file)
    try:
        src.unlink()
    except FileNotFoundError:
        pass
    session.commit()
    return {"ok": True, "points": len(points), "path": file.rel_path}
