from __future__ import annotations
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlmodel import Session, select

from db.client import get_session
from db.models import DBFile
from lib.fileops import assign_files, unassign_files
from lib.video import extract_frames

router = APIRouter()


@router.get("/files", response_model=List[DBFile])
def list_files(
    status: Optional[str] = None,
    split: Optional[str] = None,
    type: Optional[str] = None,
    session: Session = Depends(get_session),
):
    query = select(DBFile)
    if status:
        query = query.where(DBFile.status == status)
    if split:
        query = query.where(DBFile.split == split)
    if type:
        query = query.where(DBFile.type == type)
    return session.exec(query).all()


class AssignBody(BaseModel):
    fileIds: List[int]
    split: str


@router.post("/files/assign")
def assign(body: AssignBody, session: Session = Depends(get_session)):
    if body.split not in ("train", "val", "test"):
        raise HTTPException(status_code=400, detail="split must be train, val, or test")
    return assign_files(session, body.fileIds, body.split)


class UnassignBody(BaseModel):
    fileIds: List[int]


@router.post("/files/unassign")
def unassign(body: UnassignBody, session: Session = Depends(get_session)):
    return unassign_files(session, body.fileIds)


class ExtractBody(BaseModel):
    fileId: int
    split: str
    fps: float = 1.0


@router.post("/files/extract-frames")
def extract(body: ExtractBody, session: Session = Depends(get_session)):
    if body.split not in ("train", "val", "test"):
        raise HTTPException(status_code=400, detail="split must be train, val, or test")
    return extract_frames(session, body.fileId, body.split, body.fps)
