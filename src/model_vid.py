# model_vid_full.py
# End-to-end: make robust 3D clips from RWF-2000 and train an R3D-18 classifier.

import os, sys, glob, shutil, unicodedata, math, random
from typing import List, Tuple

# ---------- Safe print for Windows consoles ----------
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

def safe_print(msg: str) -> None:
    try:
        print(msg, flush=True)
    except Exception:
        try:
            print(msg.encode("ascii", "replace").decode("ascii"), flush=True)
        except Exception:
            print("[[unprintable message]]", flush=True)

# ---------- Config ----------
DATASET_PATH   = r"D:\SEM2\AML\SafeVisionAIML\Data\Video\rwf2000"   # <- parent that has train/val
CLIPS_OUT      = r"D:\SEM2\AML\SafeVisionAIML\Data\video\rwf2000_clips"            # <- where fixed-length clips will be saved

# Clip-making params (OK defaults for 3D CNNs)
CLIP_LEN       = 16          # frames per clip
TARGET_FPS     = 6           # effective fps after sampling
CLIP_OVERLAP   = 0.5         # 0.0..0.9
IMG_SIZE       = 112         # 112 (C3D/R3D) or 224 (I3D)
SAVE_AS_NPY    = False       # save JPEG frames (recommended); True -> save clip.npy per clip

# Training params
BATCH_SIZE     = 8           # reduce if OOM; CPU might need 2-4
EPOCHS         = 15
LR             = 1e-4
WEIGHT_DECAY   = 1e-2
SEED           = 42

# Toggle steps
REGENERATE_CLIPS = True      # set False if clips already prepared
CLEAN_BROKEN     = True      # remove any clip_* dirs with wrong frame counts

# ---------- Imports that need installed deps ----------
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF
from torchvision.models.video import r3d_18, R3D_18_Weights

# ---------- Utils ----------
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def unicode_ascii_safe(name: str) -> str:
    # Normalize to ASCII-safe name for directory
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    return "".join(c if c.isalnum() else "_" for c in s)

def detect_split_label(dataset_root: str, video_path: str) -> Tuple[str, str]:
    rel = os.path.relpath(video_path, dataset_root).replace("\\", "/").lower()
    parts = rel.split("/")

    # --- split ---
    split = "unknown"
    for s in ("train", "val", "valid", "validation", "test"):
        if s in parts:
            split = "val" if s in ("val", "valid", "validation") else s
            break

    # --- label ---
    # We keep internal labels as "Fight" / "NonFight" to match the rest of your code.
    label = "unknown"
    for p in parts:
        if p in ("violence", "violent", "fight", "fights"):
            label = "Fight"; break
        if p in ("nonviolence", "non-violence", "non_violence", "nonviolent", "normal", "nonfight", "non-fight", "non_fight"):
            label = "NonFight"; break

    return split, label


def read_frame(cap):
    ok, fr = cap.read()
    return fr if ok else None

def resize_frame(frame, size: int):
    return cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)

def save_clip_jpegs(out_dir: str, frames: List[np.ndarray]) -> bool:
    ensure_dir(out_dir)
    ok_all = True
    for i, fr in enumerate(frames):
        fp = os.path.join(out_dir, f"img_{i:03d}.jpg")
        ok = cv2.imwrite(fp, fr)
        if not ok:
            ok_all = False
            break
    if not ok_all:
        shutil.rmtree(out_dir, ignore_errors=True)
    return ok_all

def save_clip_npy(out_dir: str, frames: List[np.ndarray]) -> bool:
    ensure_dir(out_dir)
    arr = np.stack(frames, axis=0)  # (T, H, W, C)
    fp = os.path.join(out_dir, "clip.npy")
    try:
        np.save(fp, arr)
        return True
    except Exception:
        shutil.rmtree(out_dir, ignore_errors=True)
        return False

# ---------- Clip maker ----------
def make_clips_from_video(video_path: str) -> int:
    split, label = detect_split_label(DATASET_PATH, video_path)
    if split == "unknown" or label == "unknown":
        safe_print(f"[WARN] Unknown split/label -> skip: {video_path}")
        return 0

    cap = cv2.VideoCapture(video_path)
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

    sample_step = max(int(round(src_fps / TARGET_FPS)), 1)
    sampled_indices = list(range(0, total_frames, sample_step))
    if len(sampled_indices) == 0:
        safe_print(f"[SKIP] No sampled frames: {video_path}")
        cap.release()
        return 0

    stride = max(int(round(CLIP_LEN * (1.0 - CLIP_OVERLAP))), 1)
    starts = list(range(0, max(len(sampled_indices) - CLIP_LEN + 1, 1), stride))
    if len(sampled_indices) < CLIP_LEN:
        starts = [0]

    vname = unicode_ascii_safe(os.path.splitext(os.path.basename(video_path))[0])
    base_out = os.path.join(CLIPS_OUT, split, label, vname)
    ensure_dir(base_out)

    def read_by_index(idx: int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        fr = read_frame(cap)
        if fr is None:
            return None
        return resize_frame(fr, IMG_SIZE)

    saved = 0
    for k, st in enumerate(starts):
        needed = sampled_indices[st: st + CLIP_LEN]
        if len(needed) < CLIP_LEN:
            if not needed:
                continue
            last = needed[-1]
            needed = needed + [last] * (CLIP_LEN - len(needed))

        frames = []
        last_ok = None
        for idx in needed:
            fr = read_by_index(idx)
            if fr is None:
                fr = last_ok
                if fr is None:
                    frames = []
                    break
            frames.append(fr)
            last_ok = fr

        if len(frames) != CLIP_LEN:
            continue

        clip_dir = os.path.join(base_out, f"clip_{k:03d}")
        ok = save_clip_npy(clip_dir, frames) if SAVE_AS_NPY else save_clip_jpegs(clip_dir, frames)
        if ok:
            saved += 1

    cap.release()
    return saved

def process_dataset_to_clips():
    exts = (".avi", ".mp4", ".mov", ".mkv")
    total_videos, total_clips = 0, 0
    for root, _, files in os.walk(DATASET_PATH):
        vids = [f for f in files if f.lower().endswith(exts)]
        if not vids: 
            continue
        safe_print(f"Folder: {os.path.basename(root)} | videos: {len(vids)}")
        for i, f in enumerate(vids, 1):
            vp = os.path.join(root, f)
            safe_print(f"  [{i}/{len(vids)}] {vp}")
            try:
                n = make_clips_from_video(vp)
                total_videos += 1
                total_clips  += n
            except Exception as e:
                safe_print(f"  [ERROR] {vp}: {e}")
    safe_print(f"Clip build done. Videos={total_videos}, Clips={total_clips}")

def clean_broken_clips():
    removed = 0
    for split in ("train", "val", "test"):
        sdir = os.path.join(CLIPS_OUT, split)
        if not os.path.isdir(sdir):
            continue
        for label in ("Fight", "NonFight"):
            ldir = os.path.join(sdir, label)
            if not os.path.isdir(ldir):
                continue
            for vname in os.listdir(ldir):
                vpath = os.path.join(ldir, vname)
                if not os.path.isdir(vpath):
                    continue
                for cdir in glob.glob(os.path.join(vpath, "clip_*")):
                    if SAVE_AS_NPY:
                        ok = os.path.isfile(os.path.join(cdir, "clip.npy"))
                    else:
                        imgs = glob.glob(os.path.join(cdir, "img_*.jpg"))
                        ok = (len(imgs) == CLIP_LEN)
                    if not ok:
                        shutil.rmtree(cdir, ignore_errors=True)
                        removed += 1
    safe_print(f"Cleaned {removed} broken clip folders.")

def count_dataset_items():
    counts = {}
    for split in ("train", "val", "test"):
        total = 0
        for label in ("Fight", "NonFight"):
            pattern = os.path.join(CLIPS_OUT, split, label, "*", "clip_*")
            total += len(glob.glob(pattern))
        counts[split] = total
    safe_print(f"Clips count -> train={counts.get('train',0)}, val={counts.get('val',0)}, test={counts.get('test',0)}")
    return counts

# ---------- Dataset & Training ----------
class ClipDataset(Dataset):
    def __init__(self, split: str, augment: bool=False):
        self.split = split
        self.augment = augment
        self.items: List[Tuple[str, int]] = []
        for label_name, label_int in [("Fight", 0), ("NonFight", 1)]:
            base = os.path.join(CLIPS_OUT, split, label_name)
            if not os.path.isdir(base):
                continue
            for vname in os.listdir(base):
                vpath = os.path.join(base, vname)
                if not os.path.isdir(vpath):
                    continue
                for clip_dir in sorted(glob.glob(os.path.join(vpath, "clip_*"))):
                    if SAVE_AS_NPY:
                        if os.path.isfile(os.path.join(clip_dir, "clip.npy")):
                            self.items.append((clip_dir, label_int))
                    else:
                        imgs = glob.glob(os.path.join(clip_dir, "img_*.jpg"))
                        if len(imgs) == CLIP_LEN:
                            self.items.append((clip_dir, label_int))

    def __len__(self):
        return len(self.items)

    def _augment_img(self, img: Image.Image) -> Image.Image:
        if self.augment and random.random() < 0.5:
            img = TF.hflip(img)
        return img

    def _load_clip_jpegs(self, clip_dir: str) -> torch.Tensor:
        files = sorted(glob.glob(os.path.join(clip_dir, "img_*.jpg")))
        frames = []
        for fp in files:
            img = Image.open(fp).convert("RGB")
            img = self._augment_img(img)
            img = TF.resize(img, [IMG_SIZE, IMG_SIZE])
            t = TF.pil_to_tensor(img).float() / 255.0
            t = TF.normalize(t, mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
            frames.append(t)
        video = torch.stack(frames, dim=1)  # (C, T, H, W)
        return video

    def _load_clip_npy(self, clip_dir: str) -> torch.Tensor:
        arr = np.load(os.path.join(clip_dir, "clip.npy"))  # (T,H,W,C)
        frames = []
        for i in range(arr.shape[0]):
            img = Image.fromarray(arr[i].astype(np.uint8))
            img = self._augment_img(img)
            img = TF.resize(img, [IMG_SIZE, IMG_SIZE])
            t = TF.pil_to_tensor(img).float() / 255.0
            t = TF.normalize(t, mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
            frames.append(t)
        video = torch.stack(frames, dim=1)  # (C, T, H, W)
        return video

    def __getitem__(self, idx: int):
        clip_dir, label = self.items[idx]
        if SAVE_AS_NPY:
            video = self._load_clip_npy(clip_dir)
        else:
            video = self._load_clip_jpegs(clip_dir)
        return video, torch.tensor(label, dtype=torch.long)

def make_loaders(batch_size=BATCH_SIZE):
    # workers=0 for Windows debugging; increase to 2 later if stable
    pin = torch.cuda.is_available()
    train_ds = ClipDataset("train", augment=True)
    val_ds   = ClipDataset("val", augment=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=pin)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=pin)
    safe_print(f"Dataset -> train clips: {len(train_ds)}, val clips: {len(val_ds)}")
    return train_loader, val_loader

def make_model(num_classes=2, finetune=True):
    weights = R3D_18_Weights.KINETICS400_V1
    model = r3d_18(weights=weights)
    in_feats = model.fc.in_features
    model.fc = nn.Linear(in_feats, num_classes)
    if finetune:
        for p in model.parameters():
            p.requires_grad = True
    return model

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total, correct = 0, 0
    # Confusion matrix: rows=gt (0,1), cols=pred (0,1)
    cm = np.zeros((2,2), dtype=int)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        preds = logits.argmax(dim=1)
        total += y.size(0)
        correct += (preds == y).sum().item()
        for gt, pr in zip(y.cpu().numpy(), preds.cpu().numpy()):
            cm[gt, pr] += 1
    acc = correct / max(total, 1)
    safe_print(f"Val Acc: {acc:.4f}")
    safe_print(f"Confusion Matrix:\n{cm}")
    return acc, cm

def train():
    set_seed()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    safe_print(f"Device: {device}")

    train_loader, val_loader = make_loaders(BATCH_SIZE)

    model = make_model(num_classes=2, finetune=True).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    best_path = os.path.join(CLIPS_OUT, "best_r3d18.pth")
    ensure_dir(CLIPS_OUT)

    for epoch in range(1, EPOCHS+1):
        model.train()
        running = 0.0
        seen = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running += loss.item() * y.size(0)
            seen += y.size(0)
        train_loss = running / max(seen, 1)
        safe_print(f"Epoch {epoch}/{EPOCHS} - Train Loss: {train_loss:.4f}")

        acc, _ = evaluate(model, val_loader, device)
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), best_path)
            safe_print(f"  ✓ New best: {best_acc:.4f} -> saved {best_path}")

    safe_print(f"Training done. Best val acc = {best_acc:.4f}")

# ---------- Main ----------
if __name__ == "__main__":
    safe_print(f"Prep: CLIP_LEN={CLIP_LEN}, TARGET_FPS={TARGET_FPS}, OVERLAP={CLIP_OVERLAP}, SIZE={IMG_SIZE}, SAVE_AS_NPY={SAVE_AS_NPY}")

    if REGENERATE_CLIPS:
        safe_print("Building clips from videos...")
        process_dataset_to_clips()

    if CLEAN_BROKEN:
        safe_print("Cleaning broken/empty clip folders...")
        clean_broken_clips()

    count_dataset_items()

    safe_print("Starting training...")
    train()
