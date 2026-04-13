import subprocess
import uuid
from pathlib import Path

from sqlmodel import Session

from config import DATA_ROOT
from db.models import DBFile


def extract_frames(
    session: Session, file_id: int, split: str, fps: float = 1.0
) -> dict:
    """Extract frames from a video file at the given FPS into {split}/raw/images/."""
    record = session.get(DBFile, file_id)
    if not record:
        return {"framesExtracted": 0, "errors": [f"file {file_id} not found"]}
    if record.type != "video":
        return {"framesExtracted": 0, "errors": [f"file {file_id} is not a video"]}

    src = DATA_ROOT / record.rel_path
    if not src.exists():
        return {"framesExtracted": 0, "errors": [f"source missing: {src}"]}

    dest_dir = DATA_ROOT / split / "raw" / "images"
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Use a unique prefix so parallel extractions don't collide
    prefix = str(uuid.uuid4())
    pattern = str(dest_dir / f"{prefix}_%06d.jpg")

    result = subprocess.run(
        [
            "ffmpeg",
            "-i", str(src),
            "-vf", f"fps={fps}",
            "-q:v", "2",
            pattern,
            "-y",
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        return {"framesExtracted": 0, "errors": [result.stderr[-400:]]}

    # Register each extracted frame as a DBFile
    frames = sorted(dest_dir.glob(f"{prefix}_*.jpg"))
    for frame_path in frames:
        rel = str(frame_path.relative_to(DATA_ROOT))
        frame_record = DBFile(
            uuid=str(uuid.uuid4()),
            type="image",
            original_name=frame_path.name,
            sha256="",          # frames share source hash; skip dedup
            status="assigned",
            split=split,
            rel_path=rel,
        )
        session.add(frame_record)

    record.status = "extracted"
    record.split = split
    session.add(record)
    session.commit()

    return {"framesExtracted": len(frames), "errors": []}
