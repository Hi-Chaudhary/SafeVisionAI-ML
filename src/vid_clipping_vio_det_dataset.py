# convert_vd_to_clips.py
# Convert "violence-detection-dataset" into RWF-style clips -> frames
# Requires: pip install opencv-python pillow (Pillow only if you want later checks)

import os, sys, glob, math, random, shutil, re
from pathlib import Path
import cv2
import numpy as np

# ------------------ CONFIG ------------------
RAW_ROOT   = Path(r"D:\SEM2\AML\SafeVisionAIML\Data\Video\violence-detection-dataset")   # <-- update
CLIPS_OUT  = Path(r"D:\path\to\violence_detection_clips")     # <-- update

CLIP_LEN     = 16          # set 15 if you prefer
TARGET_FPS   = 6           # downsample videos to ~this fps before windowing
CLIP_OVERLAP = 0.50        # 0.0..0.9 (higher = more overlap, more clips)
IMG_SIZE     = 224         # 224 is common for modern backbones
TRAIN_RATIO  = 0.80        # per-class split
SEED         = 42

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".MP4", ".AVI", ".MOV", ".MKV")
LABEL_MAP  = {"violent": "Fight", "non-violent": "NonFight"}  # keep consistent with your other code

# ------------------ UTILS -------------------
def safe_print(*a, **kw):
    try:
        print(*a, **kw, flush=True)
    except Exception:
        print("".join(str(x) for x in a).encode("ascii","replace").decode("ascii"), flush=True)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def slugify(s: str) -> str:
    s = re.sub(r"[^\w\-]+", "_", s)
    return re.sub(r"_+", "_", s).strip("_")

def list_videos(root: Path):
    """
    Returns list of (video_path, class_name, cam_name).
    Expects: root/{violent,non-violent}/{cam1,cam2}/*.mp4
    """
    out = []
    for lbl in ("violent", "non-violent"):
        cls_dir = root / lbl
        if not cls_dir.exists():
            continue
        for cam_dir in cls_dir.iterdir():
            if not cam_dir.is_dir(): 
                continue
            for f in cam_dir.rglob("*"):
                if f.suffix in VIDEO_EXTS:
                    out.append((f, lbl, cam_dir.name))
    return out

def split_by_class(items, train_ratio=0.8, seed=42):
    """
    items: list of (path, class, cam)
    returns: list of (path, class, cam, split)
    """
    random.seed(seed)
    by_cls = {}
    for p, cls, cam in items:
        by_cls.setdefault(cls, []).append((p, cls, cam))
    result = []
    for cls, vids in by_cls.items():
        vids = sorted(vids, key=lambda x: x[0].name)
        random.shuffle(vids)
        cut = int(len(vids)*train_ratio)
        for i, (p, c, cam) in enumerate(vids):
            split = "train" if i < cut else "val"
            result.append((p, c, cam, split))
    return result

def read_frame(cap):
    ok, fr = cap.read()
    return fr if ok else None

def resize_frame(frame, size):
    return cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)

def make_clips_for_video(video_path: Path, cls_name: str, cam_name: str, split: str) -> int:
    """
    Save clips for one video under:
      CLIPS_OUT / split / {Fight|NonFight} / {video_id} / clip_### / img_###.jpg
    Returns number of saved clips.
    """
    label_dir = LABEL_MAP.get(cls_name.lower(), None)
    if label_dir is None:
        safe_print(f"[SKIP] Unknown class: {cls_name} -> {video_path}")
        return 0

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        safe_print(f"[SKIP] Cannot open: {video_path}")
        return 0

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if not src_fps or src_fps <= 1e-3:
        src_fps = 30.0  # fallback

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total_frames <= 0:
        safe_print(f"[SKIP] No frames: {video_path}")
        cap.release()
        return 0

    # Downsample to target fps by index stepping
    sample_step = max(int(round(src_fps / TARGET_FPS)), 1)
    sampled_idx = list(range(0, total_frames, sample_step))
    if len(sampled_idx) == 0:
        safe_print(f"[SKIP] No sampled frames: {video_path}")
        cap.release()
        return 0

    # Sliding window over sampled frames
    stride = max(int(round(CLIP_LEN * (1.0 - CLIP_OVERLAP))), 1)
    starts = list(range(0, max(len(sampled_idx) - CLIP_LEN + 1, 1), stride))
    if len(sampled_idx) < CLIP_LEN:
        starts = [0]

    # Unique video id: dataset prefix + cam + basename (no extension)
    base = video_path.stem
    vid_id = slugify(f"vd_{cam_name}_{base}")
    out_base = CLIPS_OUT / split / label_dir / vid_id
    ensure_dir(out_base)

    def read_at(idx):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        fr = read_frame(cap)
        if fr is None:
            return None
        return resize_frame(fr, IMG_SIZE)

    saved = 0
    for k, st in enumerate(starts):
        needed = sampled_idx[st: st + CLIP_LEN]
        if len(needed) < CLIP_LEN:
            if not needed:
                continue
            needed = needed + [needed[-1]] * (CLIP_LEN - len(needed))

        frames = []
        last_ok = None
        for idx in needed:
            fr = read_at(idx)
            if fr is None:
                fr = last_ok
                if fr is None:
                    frames = []
                    break
            frames.append(fr)
            last_ok = fr

        if len(frames) != CLIP_LEN:
            continue

        clip_dir = out_base / f"clip_{k:03d}"
        ensure_dir(clip_dir)

        ok_all = True
        for i, fr in enumerate(frames):
            fp = clip_dir / f"img_{i:03d}.jpg"
            if not cv2.imwrite(str(fp), fr):
                ok_all = False
                break
        if not ok_all:
            # cleanup partial
            shutil.rmtree(clip_dir, ignore_errors=True)
            continue

        saved += 1

    cap.release()
    return saved

# ------------------ MAIN --------------------
def main():
    ensure_dir(CLIPS_OUT)
    vids = list_videos(RAW_ROOT)
    if not vids:
        safe_print(f"[ERROR] No videos found under {RAW_ROOT}")
        return

    items = split_by_class(vids, train_ratio=TRAIN_RATIO, seed=SEED)
    safe_print(f"Found videos: {len(vids)}  -> train/val split by class ({TRAIN_RATIO*100:.0f}/{(1-TRAIN_RATIO)*100:.0f})")

    totals = {"videos": 0, "clips": 0}
    by_split = {"train": 0, "val": 0}

    for i, (vp, cls, cam, split) in enumerate(items, 1):
        safe_print(f"[{i}/{len(items)}] {split.upper():3s} | {cls:11s} | {cam:5s} | {vp.name}")
        try:
            n = make_clips_for_video(vp, cls, cam, split)
            totals["videos"] += 1
            totals["clips"]  += n
            by_split[split]  += n
        except Exception as e:
            safe_print(f"  [ERROR] {vp}: {e}")

    safe_print(f"\nDone. Videos processed: {totals['videos']}, Clips saved: {totals['clips']}")
    safe_print(f"Clips by split -> train={by_split.get('train',0)}, val={by_split.get('val',0)}")
    safe_print(f"Output root: {CLIPS_OUT.resolve()}")

if __name__ == "__main__":
    random.seed(SEED)
    np.random.seed(SEED)
    main()
