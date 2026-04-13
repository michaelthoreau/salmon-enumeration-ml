from datetime import datetime
from typing import Optional
from sqlmodel import SQLModel, Field


class DBFile(SQLModel, table=True):
    __tablename__ = "files"

    id: Optional[int] = Field(default=None, primary_key=True)
    uuid: str = Field(index=True, unique=True)
    # "image" | "video"
    type: str
    original_name: str
    sha256: str = Field(index=True, unique=True)
    # "input" | "assigned" | "extracted" | "labeled" | "cropped"
    status: str = Field(default="input")
    # "train" | "val" | "test" | None
    split: Optional[str] = Field(default=None)
    # path relative to DATA_ROOT
    rel_path: str
    created_at: datetime = Field(default_factory=datetime.utcnow)


class DBCrop(SQLModel, table=True):
    __tablename__ = "crops"

    id: Optional[int] = Field(default=None, primary_key=True)
    uuid: str = Field(index=True, unique=True)
    source_uuid: str = Field(index=True)
    split: str
    # path relative to DATA_ROOT
    rel_path: str
    x: int
    y: int
    w: int
    h: int
    created_at: datetime = Field(default_factory=datetime.utcnow)
    labeled_at: Optional[datetime] = Field(default=None)


class DBAnnotation(SQLModel, table=True):
    __tablename__ = "annotations"

    id: Optional[int] = Field(default=None, primary_key=True)
    file_id: Optional[int] = Field(default=None, index=True)
    crop_id: Optional[int] = Field(default=None, index=True)
    # JSON-encoded list of [x, y] pairs
    points_json: str = Field(default="[]")
    updated_at: datetime = Field(default_factory=datetime.utcnow)
