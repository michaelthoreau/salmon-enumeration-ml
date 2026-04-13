from __future__ import annotations
import uuid
from pathlib import Path

from PIL import Image
from sqlmodel import Session

from config import DATA_ROOT
from db.models import DBCrop, DBFile

CROP_SIZE = 640


def generate_crops(
    session: Session, file_ids: list[int], overlap: float = 0.2, crop_size: int = CROP_SIZE
) -> dict:
    """Tile each image into CROP_SIZE×CROP_SIZE patches and save to {split}/unlabelled/images/."""
    generated = 0
    errors = []
    stride = int(crop_size * (1 - overlap))

    for fid in file_ids:
        record = session.get(DBFile, fid)
        if not record:
            errors.append(f"file {fid} not found")
            continue
        if record.type != "image":
            errors.append(f"file {fid} is not an image")
            continue
        if not record.split:
            errors.append(f"file {fid} has no split assigned")
            continue

        if record.status == "cropped":
            errors.append(f"file {fid} already cropped — skipped")
            continue

        src = DATA_ROOT / record.rel_path
        if not src.exists():
            errors.append(f"file {fid} missing on disk")
            continue

        dest_dir = DATA_ROOT / record.split / "unlabelled" / "images"
        dest_dir.mkdir(parents=True, exist_ok=True)

        try:
            img = Image.open(src).convert("RGB")
        except Exception as e:
            errors.append(f"file {fid}: {e}")
            continue

        W, H = img.size
        y = 0
        while y < H:
            x = 0
            while x < W:
                x2 = min(x + crop_size, W)
                y2 = min(y + crop_size, H)
                crop_img = img.crop((x, y, x2, y2))

                # Pad to crop_size if edge tile is smaller
                if crop_img.size != (crop_size, crop_size):
                    padded = Image.new("RGB", (crop_size, crop_size), (0, 0, 0))
                    padded.paste(crop_img, (0, 0))
                    crop_img = padded

                crop_uuid = str(uuid.uuid4())
                crop_name = f"{crop_uuid}.jpg"
                crop_path = dest_dir / crop_name
                crop_img.save(crop_path, quality=92)

                crop_record = DBCrop(
                    uuid=crop_uuid,
                    source_uuid=record.uuid,
                    split=record.split,
                    rel_path=str(crop_path.relative_to(DATA_ROOT)),
                    x=x,
                    y=y,
                    w=x2 - x,
                    h=y2 - y,
                )
                session.add(crop_record)
                generated += 1

                if x2 == W:
                    break
                x += stride
            if y2 == H:
                break
            y += stride

        record.status = "cropped"
        session.add(record)

    session.commit()
    return {"cropsGenerated": generated, "errors": errors}
