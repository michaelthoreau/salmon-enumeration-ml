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


class DBVideoFrame(SQLModel, table=True):
    """Tracks video frames that have been committed for annotation."""
    __tablename__ = "video_frames"

    id: Optional[int] = Field(default=None, primary_key=True)
    uuid: str = Field(index=True, unique=True)
    # FK to DBFile.id (must be type="video")
    source_file_id: int = Field(index=True)
    frame_number: int
    split: str
    # paths relative to DATA_ROOT — set after commit
    frame_rel_path:      Optional[str] = Field(default=None)   # {split}/images/{uuid}.jpg
    motion_rel_path:     Optional[str] = Field(default=None)   # {split}/images/{uuid}_prev10.jpg
    background_rel_path: Optional[str] = Field(default=None)   # {split}/images/{uuid}_background.jpg
    labeled_at: Optional[datetime] = Field(default=None)
    created_at: datetime = Field(default_factory=datetime.utcnow)
