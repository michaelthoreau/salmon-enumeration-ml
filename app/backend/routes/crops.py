from __future__ import annotations
from typing import List, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlmodel import Session, select

from db.client import get_session
from db.models import DBCrop
from lib.cropper import generate_crops

router = APIRouter()


class CropsPage(BaseModel):
    items: List[DBCrop]
    total: int
    offset: int
    limit: int


@router.get("/crops", response_model=CropsPage)
def list_crops(
    split: Optional[str] = None,
    offset: int = 0,
    limit: int = 100,
    session: Session = Depends(get_session),
):
    query = select(DBCrop)
    if split:
        query = query.where(DBCrop.split == split)
    all_items = session.exec(query).all()
    total = len(all_items)
    items = all_items[offset: offset + limit]
    return CropsPage(items=items, total=total, offset=offset, limit=limit)


class CropsBody(BaseModel):
    fileIds: List[int]
    overlap: float = 0.2
    cropSize: int = 640


@router.post("/crops")
def create_crops(body: CropsBody, session: Session = Depends(get_session)):
    return generate_crops(session, body.fileIds, body.overlap, body.cropSize)
