from __future__ import annotations
from io import BytesIO
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from PIL import Image
from sqlmodel import Session

from config import DATA_ROOT
from db.client import get_session
from db.models import DBCrop, DBFile

router = APIRouter()

THUMB_SIZE = 200
THUMB_CACHE = DATA_ROOT / ".thumbcache"


def _thumb_path(source: Path, size: int) -> Path:
    rel = source.relative_to(DATA_ROOT)
    cache_path = THUMB_CACHE / str(size) / rel.with_suffix(".jpg")
    return cache_path


def _make_thumbnail(source: Path, size: int) -> bytes:
    cache = _thumb_path(source, size)
    if cache.exists() and cache.stat().st_mtime >= source.stat().st_mtime:
        return cache.read_bytes()

    img = Image.open(source).convert("RGB")
    img.thumbnail((size, size), Image.LANCZOS)
    cache.parent.mkdir(parents=True, exist_ok=True)
    img.save(cache, format="JPEG", quality=80)
    return cache.read_bytes()


def _full(path: Path) -> bytes:
    buf = BytesIO()
    Image.open(path).convert("RGB").save(buf, format="JPEG", quality=90)
    return buf.getvalue()


@router.get("/thumbnails/files/{file_id}")
def thumb_file(file_id: int, size: int = THUMB_SIZE, session: Session = Depends(get_session)):
    record = session.get(DBFile, file_id)
    if not record:
        raise HTTPException(status_code=404)
    path = DATA_ROOT / record.rel_path
    if not path.exists():
        raise HTTPException(status_code=404)
    return Response(
        content=_make_thumbnail(path, size),
        media_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=86400"},
    )


@router.get("/thumbnails/crops/{crop_id}")
def thumb_crop(crop_id: int, size: int = THUMB_SIZE, session: Session = Depends(get_session)):
    record = session.get(DBCrop, crop_id)
    if not record:
        raise HTTPException(status_code=404)
    path = DATA_ROOT / record.rel_path
    if not path.exists():
        raise HTTPException(status_code=404)
    return Response(
        content=_make_thumbnail(path, size),
        media_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=86400"},
    )


@router.get("/thumbnails/crops/uuid/{crop_uuid}")
def thumb_crop_by_uuid(crop_uuid: str, size: int = THUMB_SIZE, session: Session = Depends(get_session)):
    from sqlmodel import select
    record = session.exec(select(DBCrop).where(DBCrop.uuid == crop_uuid)).first()
    if not record:
        raise HTTPException(status_code=404)
    path = DATA_ROOT / record.rel_path
    if not path.exists():
        raise HTTPException(status_code=404)
    return Response(
        content=_make_thumbnail(path, size),
        media_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=86400"},
    )


@router.get("/images/files/{file_id}")
def serve_file(file_id: int, session: Session = Depends(get_session)):
    record = session.get(DBFile, file_id)
    if not record:
        raise HTTPException(status_code=404)
    path = DATA_ROOT / record.rel_path
    if not path.exists():
        raise HTTPException(status_code=404)
    return Response(content=_full(path), media_type="image/jpeg")


@router.get("/images/crops/{crop_id}")
def serve_crop(crop_id: int, session: Session = Depends(get_session)):
    record = session.get(DBCrop, crop_id)
    if not record:
        raise HTTPException(status_code=404)
    path = DATA_ROOT / record.rel_path
    if not path.exists():
        raise HTTPException(status_code=404)
    return Response(content=_full(path), media_type="image/jpeg")


@router.get("/images/crops/uuid/{crop_uuid}")
def serve_crop_by_uuid(crop_uuid: str, session: Session = Depends(get_session)):
    from sqlmodel import select
    record = session.exec(select(DBCrop).where(DBCrop.uuid == crop_uuid)).first()
    if not record:
        raise HTTPException(status_code=404)
    path = DATA_ROOT / record.rel_path
    if not path.exists():
        raise HTTPException(status_code=404)
    return Response(content=_full(path), media_type="image/jpeg")
