"""Routes for video frame browsing and annotation."""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import uuid as uuid_mod
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel
from sqlmodel import Session, select

from config import DATA_ROOT
from db.client import get_session
from db.models import DBFile, DBVideoFrame
from lib.video_frames import (
    get_background_jpeg,
    get_bg_subtracted_frame_jpeg,
    get_frame_count,
    get_frame_jpeg,
    get_frame_png,
    get_video_fps,
)
import os
import subprocess
import tempfile

MOTION_OFFSET = 10   # frames earlier for the motion-cue image
MOTION_SUFFIX = "_prev10"

router = APIRouter(prefix="/video")


# ---------------------------------------------------------------------------
# Video metadata
# ---------------------------------------------------------------------------

@router.get("")
def list_videos(session: Session = Depends(get_session)):
    """Return all video DBFiles."""
    videos = session.exec(select(DBFile).where(DBFile.type == "video")).all()
    result = []
    for v in videos:
        path = DATA_ROOT / v.rel_path
        try:
            total = get_frame_count(path)
            fps   = get_video_fps(path)
        except Exception:
            total, fps = 0, 0.0
        result.append({**v.model_dump(), "total_frames": total, "fps": fps})
    return result


@router.get("/{file_id}/info")
def video_info(file_id: int, session: Session = Depends(get_session)):
    video = session.get(DBFile, file_id)
    if not video or video.type != "video":
        raise HTTPException(status_code=404, detail="Video not found")
    path = DATA_ROOT / video.rel_path
    return {
        "id": video.id,
        "original_name": video.original_name,
        "total_frames": get_frame_count(path),
        "fps": get_video_fps(path),
    }


# ---------------------------------------------------------------------------
# Frame serving
# ---------------------------------------------------------------------------

@router.get("/{file_id}/stream")
def stream_video(file_id: int, session: Session = Depends(get_session)):
    """Serve the raw video file with range-request support for browser seeking."""
    video = session.get(DBFile, file_id)
    if not video or video.type != "video":
        raise HTTPException(status_code=404, detail="Video not found")
    path = DATA_ROOT / video.rel_path
    if not path.exists():
        raise HTTPException(status_code=404, detail="Video file missing on disk")
    from fastapi.responses import FileResponse
    return FileResponse(str(path), media_type="video/mp4")


@router.get("/{file_id}/frame")
def serve_frame(
    file_id: int,
    n: int = Query(..., description="0-based frame number"),
    mode: str = Query("raw", description="'raw' or 'bg_sub'"),
    session: Session = Depends(get_session),
):
    """Stream a single video frame as JPEG. mode=bg_sub returns |frame − background|."""
    video = session.get(DBFile, file_id)
    if not video or video.type != "video":
        raise HTTPException(status_code=404, detail="Video not found")
    path = DATA_ROOT / video.rel_path
    try:
        if mode == "bg_sub":
            data     = get_bg_subtracted_frame_jpeg(path, n, quality=95)
            mimetype = "image/jpeg"
        else:
            data, mimetype = get_frame_png(path, n)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return Response(content=data, media_type=mimetype)


@router.get("/{file_id}/clip")
def serve_clip(
    file_id: int,
    end_frame: int = Query(..., description="Last frame of the clip (inclusive, 0-based)"),
    seconds: float = Query(1.0, description="Duration of clip in seconds"),
    session: Session = Depends(get_session),
):
    """Return a short video clip ending at end_frame, encoded as MP4."""
    video = session.get(DBFile, file_id)
    if not video or video.type != "video":
        raise HTTPException(status_code=404, detail="Video not found")
    path = DATA_ROOT / video.rel_path

    fps = get_video_fps(path) or 30.0
    end_time   = (end_frame + 1) / fps
    start_time = max(0.0, end_time - seconds)
    duration   = end_time - start_time

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
    os.close(tmp_fd)
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-ss", f"{start_time:.6f}",
                "-i", str(path),
                "-t",  f"{duration:.6f}",
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
                "-an",           # no audio
                "-movflags", "faststart",
                tmp_path,
            ],
            capture_output=True,
        )
        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=result.stderr.decode()[-300:])
        data = Path(tmp_path).read_bytes()
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    return Response(
        content=data,
        media_type="video/mp4",
        headers={"Content-Disposition": "inline"},
    )




class CommitFrameBody(BaseModel):
    split: str
    points: List[List[int]]


@router.post("/{file_id}/frame/{frame_number}/commit")
def commit_frame(
    file_id: int,
    frame_number: int,
    body: CommitFrameBody,
    session: Session = Depends(get_session),
):
    """
    Save the annotated frame (and its motion-cue frame) to {split}/images/,
    write the annotation JSON to {split}/annotations/, and record in DB.
    """
    if body.split not in ("train", "val", "test"):
        raise HTTPException(status_code=400, detail="split must be train/val/test")

    video = session.get(DBFile, file_id)
    if not video or video.type != "video":
        raise HTTPException(status_code=404, detail="Video not found")

    video_path = DATA_ROOT / video.rel_path
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video file missing on disk")

    frame_uuid = str(uuid_mod.uuid4())
    images_dir = DATA_ROOT / body.split / "images"
    ann_dir    = DATA_ROOT / body.split / "annotations"
    images_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    # --- Main annotated frame ---
    frame_name = f"{frame_uuid}.jpg"
    frame_path = images_dir / frame_name
    try:
        frame_bytes = get_frame_jpeg(video_path, frame_number)
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=422, detail=str(e))
    frame_path.write_bytes(frame_bytes)

    # --- Motion-cue frame (10 frames earlier, floored at 0) ---
    motion_frame_number = max(0, frame_number - MOTION_OFFSET)
    motion_name = f"{frame_uuid}{MOTION_SUFFIX}.jpg"
    motion_path = images_dir / motion_name
    try:
        motion_bytes = get_frame_jpeg(video_path, motion_frame_number)
        motion_path.write_bytes(motion_bytes)
        motion_rel = str(motion_path.relative_to(DATA_ROOT))
    except Exception:
        motion_rel = None  # don't fail the whole commit if motion frame errors

    # --- Background frame (5-second window average, 1-in-10 frames) ---
    bg_name = f"{frame_uuid}_background.jpg"
    bg_path = images_dir / bg_name
    try:
        bg_bytes = get_background_jpeg(video_path, frame_number)
        if bg_bytes:
            bg_path.write_bytes(bg_bytes)
            bg_rel = str(bg_path.relative_to(DATA_ROOT))
        else:
            bg_rel = None
    except Exception:
        bg_rel = None

    # --- Annotation JSON (same format as the rest of the pipeline) ---
    import cv2 as _cv2
    cap = _cv2.VideoCapture(str(video_path))
    cap.set(_cv2.CAP_PROP_POS_FRAMES, frame_number)
    _, fr = cap.read()
    cap.release()
    h, w = (fr.shape[:2] if fr is not None else (0, 0))

    annotation = {
        "filename": frame_name,
        "width":    w,
        "height":   h,
        "points":   body.points,
        "source_video":    video.original_name,
        "source_frame":    frame_number,
        "motion_file":     motion_name if motion_rel else None,
        "motion_offset":   MOTION_OFFSET,
        "background_file": bg_name if bg_rel else None,
    }
    (ann_dir / f"{frame_uuid}.json").write_text(json.dumps(annotation))

    # --- DB record ---
    db_frame = DBVideoFrame(
        uuid=frame_uuid,
        source_file_id=file_id,
        frame_number=frame_number,
        split=body.split,
        frame_rel_path=str(frame_path.relative_to(DATA_ROOT)),
        motion_rel_path=motion_rel,
        background_rel_path=bg_rel,
        labeled_at=datetime.utcnow(),
    )
    session.add(db_frame)
    session.commit()

    return {
        "ok": True,
        "uuid": frame_uuid,
        "frame": str(frame_path.relative_to(DATA_ROOT)),
        "motion": motion_rel,
        "background": bg_rel,
        "points": len(body.points),
    }


# ---------------------------------------------------------------------------
# List annotated frames for a video
# ---------------------------------------------------------------------------

@router.get("/{file_id}/frames")
def list_frames(
    file_id: int,
    split: Optional[str] = None,
    session: Session = Depends(get_session),
):
    q = select(DBVideoFrame).where(DBVideoFrame.source_file_id == file_id)
    if split:
        q = q.where(DBVideoFrame.split == split)
    return session.exec(q).all()
