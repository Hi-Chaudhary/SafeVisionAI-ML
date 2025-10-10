# scvd_preprocess_to_clips.py
import os, re, csv, glob, random, shutil
from pathlib import Path
import cv2
import numpy as np

# ---------------- CONFIG (edit paths) ----------------
SCVD_ROOT   = Path(r"D:\SEM2\AML\SafeVisionAIML\Data\Video\SCVD\SCVD_converted")  # your screenshot path
OUT_ROOT    = Path(r"D:\SEM2\AML\SafeVisionAIML\Data\Video\scvd_clips")

# Clip spec (match your RWF/PMC pipeline)
CLIP_LEN       = 16
TARGET_FPS     = 6         # effective FPS after index-stepping
CLIP_OVERLAP   = 0.50      # 0..0.9 (higher = more overlap)
IMG_SIZE       = 112       # 112 (R3D/C3D) or 224 (I3D/X3D)
VAL_RATIO_FROM_TRAIN = 0.15  # carve val from Train at video level
SEED           = 42
SAVE_AS_NPY    = False     # False -> save JPEG frames; True -> save clip.npy

VIDEO_EXTS = (".mp4",".avi",".mov",".mkv",".MP4",".AVI",".MOV",".MKV")

# Label mapping for SCVD_converted
#   Normal -> NonFight
#   Violence + Weaponized -> Fight
CLASS_MAP = {
    "normal": "NonFight",
    "violence": "Fight",
    "weaponized": "Fight",
}

# ---------------- Utils ----------------
def safe_print(*a, **kw):
    try:
        print(*a, **kw, flush=True)
    except Exception:
        print(" ".join(str(x) for x in a).encode("ascii","replace").decode("ascii"), flush=True)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def slug(s: str) -> str:
    s = re.sub(r"[^\w\-]+", "_", s)
    return re.sub(r"_+", "_", s).strip("_")

def list_split_items(split_dir: Path):
    """
    Return list of (video_path, mapped_label['Fight'|'NonFight']).
    Expects split_dir has subfolders: Normal, Violence, Weaponized (case-insensitive).
    """
    items = []
    if not split_dir.exists():
        return items
    for cls_dir in split_dir.iterdir():
        if not cls_dir.is_dir():
            continue
        key = CLASS_MAP.get(cls_dir.name.lower(), None)
        if key is None:
            # unknown class folder -> skip
            continue
        for f in cls_dir.rglob("*"):
            if f.suffix in VIDEO_EXTS:
                items.append((f, key))
    return items

def carve_train_val(train_items, val_ratio=0.15, seed=42):
    """
    train_items: list[(Path, 'Fight'|'NonFight')]
    Create val by sampling whole videos per class.
    """
    random.seed(seed)
    by_cls = {"Fight": [], "NonFight": []}
    for p, lab in train_items:
        by_cls[lab].append(p)
    out_train, out_val = [], []
    for lab, vids in by_cls.items():
        vids = sorted(vids, key=lambda x: x.name)
        random.shuffle(vids)
        cut = int(round(len(vids) * (1.0 - val_ratio)))
        tr, va = vids[:cut], vids[cut:]
        out_train.extend([(p, lab) for p in tr])
        out_val.extend([(p, lab) for p in va])
    return out_train, out_val

def resize_frame(fr, size):
    return cv2.resize(fr, (size, size), interpolation=cv2.INTER_AREA)

def save_clip_jpegs(out_dir: Path, frames):
    ensure_dir(out_dir)
    ok_all = True
    for i, fr in enumerate(frames):
        fp = out_dir / f"img_{i:03d}.jpg"
        if not cv2.imwrite(str(fp), fr):
            ok_all = False
            break
    if not ok_all:
        shutil.rmtree(out_dir, ignore_errors=True)
    return ok_all

def save_clip_npy(out_dir: Path, frames):
    ensure_dir(out_dir)
    arr = np.stack(frames, axis=0)  # (T,H,W,C)
    try:
        np.save(out_dir / "clip.npy", arr)
        return True
    except Exception:
        shutil.rmtree(out_dir, ignore_errors=True)
        return False

def make_clips_for_video(video_path: Path, label: str, split: str):
    """
    Save under: OUT_ROOT/split/{Fight|NonFight}/{video_id}/clip_xxx/(img_yyy.jpg or clip.npy)
    Return list of clip rows for manifest.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        safe_print(f"[SKIP] cannot open: {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1e-3:
        fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total_frames <= 0:
        cap.release()
        return []

    # downsample by index stepping -> approx TARGET_FPS
    step = max(int(round(fps / TARGET_FPS)), 1)
    sampled_idx = list(range(0, total_frames, step))
    if not sampled_idx:
        cap.release()
        return []

    stride = max(int(round(CLIP_LEN * (1.0 - CLIP_OVERLAP))), 1)
    starts = list(range(0, max(len(sampled_idx) - CLIP_LEN + 1, 1), stride))
    if len(sampled_idx) < CLIP_LEN:
        starts = [0]

    vid_id = slug(video_path.stem)
    base_out = OUT_ROOT / split / label / vid_id
    ensure_dir(base_out)

    def read_at(idx):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, fr = cap.read()
        if not ok:
            return None
        return resize_frame(fr, IMG_SIZE)

    rows = []
    for k, st in enumerate(starts):
        need = sampled_idx[st: st + CLIP_LEN]
        if len(need) < CLIP_LEN:
            if not need: 
                continue
            need = need + [need[-1]] * (CLIP_LEN - len(need))

        frames = []
        last = None
        for idx in need:
            fr = read_at(idx)
            if fr is None:
                fr = last
                if fr is None:
                    frames = []
                    break
            frames.append(fr)
            last = fr
        if len(frames) != CLIP_LEN:
            continue

        clip_dir = base_out / f"clip_{k:03d}"
        ok = save_clip_npy(clip_dir, frames) if SAVE_AS_NPY else save_clip_jpegs(clip_dir, frames)
        if ok:
            rows.append({
                "clip_dir": str(clip_dir).replace("\\","/"),
                "label": 0 if label=="Fight" else 1,  # Fight=0, NonFight=1 (consistent)
                "split": split,
                "dataset": "SCVD",
                "video_id": vid_id,
                "start_idx": need[0],
                "end_idx": need[-1],
                "fps": round(float(fps), 3),
                "img_size": IMG_SIZE,
            })

    cap.release()
    return rows

# ---------------- Main ----------------
def main():
    random.seed(SEED)
    np.random.seed(SEED)

    # 1) Collect Train/Test (map to Fight/NonFight)
    train_items = list_split_items(SCVD_ROOT / "Train")
    test_items  = list_split_items(SCVD_ROOT / "Test")   # keep as 'test' split
    safe_print(f"Found Train videos: {len(train_items)} | Test videos: {len(test_items)}")

    # 2) Make val from train (video-level, stratified)
    train_items, val_items = carve_train_val(train_items, VAL_RATIO_FROM_TRAIN, SEED)
    safe_print(f"After split -> train: {len(train_items)}, val: {len(val_items)}, test: {len(test_items)}")

    # 3) Build clips + manifest
    ensure_dir(OUT_ROOT)
    manifest_rows = []

    def process(items, split):
        total_videos, total_clips = 0, 0
        for i, (vp, lab) in enumerate(items, 1):
            safe_print(f"[{split.upper():4s}] {i}/{len(items)} | {lab:9s} | {vp.name}")
            rows = make_clips_for_video(vp, lab, split)
            total_videos += 1
            total_clips  += len(rows)
            manifest_rows.extend(rows)
        safe_print(f" -> {split}: videos={total_videos}, clips={total_clips}")

    process(train_items, "train")
    process(val_items,   "val")
    process(test_items,  "test")

    # 4) Write manifest CSV
    out_csv = OUT_ROOT / "manifest_scvd.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "clip_dir","label","split","dataset","video_id","start_idx","end_idx","fps","img_size"
        ])
        w.writeheader(); w.writerows(manifest_rows)
    safe_print(f"[DONE] wrote manifest -> {out_csv} | rows={len(manifest_rows)}")

if __name__ == "__main__":
    main()
