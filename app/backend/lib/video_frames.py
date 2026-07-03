"""Utilities for extracting individual frames from video files."""
from functools import lru_cache
from pathlib import Path


# ---------------------------------------------------------------------------
# Basic frame helpers
# ---------------------------------------------------------------------------

def get_frame_jpeg(video_path: Path, frame_number: int, quality: int = 90) -> bytes:
    """Return a JPEG-encoded byte string for the given frame number (0-based)."""
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        if not ret:
            raise ValueError(f"Could not read frame {frame_number} from {video_path.name}")
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise RuntimeError("JPEG encoding failed")
        return buf.tobytes()
    finally:
        cap.release()


def get_frame_png(video_path: Path, frame_number: int) -> tuple[bytes, str]:
    """Return a lossless PNG byte string for the given frame (0-based).

    PNG avoids the extra lossy JPEG round-trip — the displayed frame is
    pixel-identical to what the browser's H.264 decoder would produce.
    Returns (bytes, mime_type).
    """
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        if not ret:
            raise ValueError(f"Could not read frame {frame_number} from {video_path.name}")
        # PNG compression level 1 = fast encode, still lossless
        ok, buf = cv2.imencode(".png", frame, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        if not ok:
            raise RuntimeError("PNG encoding failed")
        return buf.tobytes(), "image/png"
    finally:
        cap.release()


def get_frame_count(video_path: Path) -> int:
    """Return the total number of frames in the video."""
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    try:
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()


def get_video_fps(video_path: Path) -> float:
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    try:
        return cap.get(cv2.CAP_PROP_FPS)
    finally:
        cap.release()


# ---------------------------------------------------------------------------
# Background model — MOG2 per-  5-second window (1 in every 10 frames)
# ---------------------------------------------------------------------------

_BG_WINDOW_SECONDS = 2
_BG_SAMPLE_EVERY   = 3    # stride between sampled frames
_BG_SCALE          = 4.0   # sensitivity of foreground detection (higher = more overlay area)
_BG_NOISE_FLOOR    = 20    # per-channel diff below this is ignored
_BG_BLUR_KERNEL    = 9     # Gaussian blur kernel — suppresses rock texture
_BG_MAX_SHIFT_PX   = 30    # max camera translation (px) phase-correlation will correct
_BG_OVERLAY_ALPHA  = 0.7   # max opacity of the foreground highlight overlay (0-1)


@lru_cache(maxsize=16)   # numpy arrays are large; keep fewer than JPEG cache
def _background_numpy(video_path_str: str, window_idx: int):
    """
    Build a MOG2 background model for the 5-second window and return the
    background as a raw uint8 numpy array (no JPEG encoding — avoids blocking
    artefacts when used for subtraction).

    Returns None on failure.
    """
    import cv2
    import numpy as np

    path = Path(video_path_str)
    cap  = cv2.VideoCapture(str(path))
    try:
        fps         = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total       = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        window_size = max(1, int(_BG_WINDOW_SECONDS * fps))
        start       = window_idx * window_size
        end         = min(start + window_size, total)
        indices     = list(range(start, end, _BG_SAMPLE_EVERY))
        if not indices:
            return None

        mog2 = cv2.createBackgroundSubtractorMOG2(
            history=len(indices),
            varThreshold=16,
            detectShadows=False,
        )
        for fn in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
            ok, f = cap.read()
            if ok:
                mog2.apply(f, learningRate=-1)
    finally:
        cap.release()

    bg = mog2.getBackgroundImage()
    return bg  # uint8 ndarray or None


def _window_idx_for_frame(video_path: Path, frame_number: int) -> int:
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    finally:
        cap.release()
    window_size = max(1, int(_BG_WINDOW_SECONDS * fps))
    return frame_number // window_size


def get_background_jpeg(video_path: Path, frame_number: int, quality: int = 90) -> bytes:
    """Return the background image as JPEG (for saving to disk)."""
    import cv2
    bg = _background_numpy(str(video_path), _window_idx_for_frame(video_path, frame_number))
    if bg is None:
        return b""
    _, buf = cv2.imencode(".jpg", bg, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()


def get_bg_subtracted_frame_jpeg(
    video_path: Path, frame_number: int, quality: int = 90
) -> bytes:
    """
    Return the original frame with detected foreground regions highlighted as
    a semi-transparent orange overlay, corrected for camera shake and rock texture.

    Pipeline:
      1. Phase correlation  — estimate & compensate camera translation.
      2. Gaussian blur      — suppress high-frequency rock/substrate texture.
      3. Absolute diff      — |blurred_raw − blurred_aligned_bg|
      4. Noise floor + mask — zero sub-threshold diffs; normalise to soft [0,1] mask.
      5. Overlay            — alpha-blend orange highlight onto the original frame.
    """
    import cv2
    import numpy as np

    # Read raw frame directly — no JPEG round-trip, no compression artefacts
    cap = cv2.VideoCapture(str(video_path))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ok, raw = cap.read()
    finally:
        cap.release()

    if not ok:
        return get_frame_jpeg(video_path, frame_number, quality)

    bg = _background_numpy(str(video_path), _window_idx_for_frame(video_path, frame_number))
    if bg is None:
        _, buf = cv2.imencode(".jpg", raw, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return buf.tobytes()

    # --- Step 1: motion compensation via phase correlation ---
    # Converts both frames to float32 grayscale and uses FFT to estimate the
    # rigid translation between background and current frame.
    raw_gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY).astype(np.float32)
    bg_gray  = cv2.cvtColor(bg,  cv2.COLOR_BGR2GRAY).astype(np.float32)
    (dx, dy), _response = cv2.phaseCorrelate(bg_gray, raw_gray)

    if abs(dx) <= _BG_MAX_SHIFT_PX and abs(dy) <= _BG_MAX_SHIFT_PX:
        # Shift the background to align with the current frame
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        bg_aligned = cv2.warpAffine(
            bg, M, (bg.shape[1], bg.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,   # replicate edges, no black borders
        )
    else:
        bg_aligned = bg   # shift too large — likely non-rigid motion, skip correction

    # --- Step 2: blur both to kill the rock checkerboard texture ---
    k = _BG_BLUR_KERNEL
    raw_blur = cv2.GaussianBlur(raw,        (k, k), 0)
    bg_blur  = cv2.GaussianBlur(bg_aligned, (k, k), 0)

    # --- Steps 3-4: diff → soft foreground mask ---
    diff = cv2.absdiff(raw_blur, bg_blur).astype(np.float32)
    diff[diff < _BG_NOISE_FLOOR] = 0.0
    diff = np.clip(diff * _BG_SCALE, 0, 255)

    # Collapse colour channels → single-channel mask in [0, 1]
    mask = diff.max(axis=2) / 255.0  # [H, W]

    # --- Step 5: overlay foreground highlight on original frame ---
    # Bright blue in BGR — stands out against the grey sonar background
    HIGHLIGHT_BGR = np.array([255, 100, 0], dtype=np.float32)
    alpha = np.clip(mask * _BG_OVERLAY_ALPHA, 0.0, _BG_OVERLAY_ALPHA)[:, :, np.newaxis]
    result = raw.astype(np.float32) * (1.0 - alpha) + HIGHLIGHT_BGR * alpha
    result = np.clip(result, 0, 255).astype(np.uint8)

    _, buf = cv2.imencode(".jpg", result, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()
