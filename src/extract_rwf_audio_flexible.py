import argparse, subprocess, json, csv, sys
from pathlib import Path
# add near the imports
import shutil


#if not FFMPEG or not FFPROBE:
#    raise SystemExit("ffmpeg/ffprobe not found. Install them or pass --ffmpeg/--ffprobe with full paths.")

# replace all hardcoded 'ffmpeg'/'ffprobe' calls with FFMPEG/FFPROBE, e.g.:
# p = run([FFPROBE, "-v","error","-show_streams", ...])
# p = run([FFMPEG, "-hide_banner","-loglevel","error","-y","-i", str(video), ...])


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

SPLIT_TOKENS = {
    "train": {"train", "training"},
    "val":   {"val", "valid", "validation", "dev"},
    "test":  {"test", "testing", "eval", "evaluation"},
}

# tokens we look for in any parent folder names (case-insensitive)
VIOLENT_TOKENS = {"fight","violent","violence","aggress","assault","riot"}
NONVIOLENT_TOKENS = {"nonfight","non-fight","non_fight","nonviolent","non-violent","non_violent","nonviolence","normal","benign"}

def run(cmd):
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

def has_audio(video: Path) -> bool:
    p = run(["ffprobe","-v","error","-show_streams","-select_streams","a","-of","json",str(video)])
    if p.returncode != 0:
        return False
    try:
        info = json.loads(p.stdout or "{}")
        return len(info.get("streams", [])) > 0
    except json.JSONDecodeError:
        return False

def extract_audio(video: Path, wav: Path, sr: int) -> bool:
    wav.parent.mkdir(parents=True, exist_ok=True)
    p = run([
        "ffmpeg","-hide_banner","-loglevel","error","-y",
        "-i",str(video),
        "-vn","-ac","1","-ar",str(sr),"-acodec","pcm_s16le",
        str(wav)
    ])
    return p.returncode == 0

def get_duration_sec(wav: Path) -> float:
    p = run(["ffprobe","-v","error","-show_entries","format=duration","-of","default=noprint_wrappers=1:nokey=1",str(wav)])
    try: return float((p.stdout or "0").strip())
    except: return -1.0

def guess_split(path: Path) -> str:
    parts = [p.lower() for p in path.parts]
    for split, toks in SPLIT_TOKENS.items():
        if any(t in part for part in parts for t in toks):
            return split
    return "train"  # default if not found

def guess_label(path: Path) -> str | None:
    parts = " ".join([p.lower() for p in path.parts])
    if any(tok in parts for tok in NONVIOLENT_TOKENS): return "non_violent"
    if any(tok in parts for tok in VIOLENT_TOKENS):    return "violent"
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Path to RWF-2000 root (the folder that contains the videos)")
    ap.add_argument("--out",  required=True, help="Output folder for extracted WAVs")
    ap.add_argument("--sr",   type=int, default=16000, help="Target sample rate (default: 16000)")
    # in argparse section:
    ap.add_argument("--ffmpeg", default=None, help="Path to ffmpeg.exe (optional)")
    ap.add_argument("--ffprobe", default=None, help="Path to ffprobe.exe (optional)")

# after args parsed:
    #FFMPEG = args.ffmpeg or shutil.which("ffmpeg")
    #FFPROBE = args.ffprobe or shutil.which("ffprobe")

    args = ap.parse_args()

    root = Path(args.root)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    rows = []
    n_seen = n_noaudio = n_unknown = n_ok = 0

    for v in root.rglob("*"):
        if v.suffix.lower() in VIDEO_EXTS and v.is_file():
            n_seen += 1
            label = guess_label(v.parent)
            if label is None:
                n_unknown += 1
                continue
            if not has_audio(v):
                n_noaudio += 1
                continue
            split = guess_split(v.parent)
            rel = v.relative_to(root)
            # write into out/<split>/<label>/<same_subtree>/<file>.wav
            wav_path = out_root / split / label / rel.with_suffix(".wav")
            if extract_audio(v, wav_path, args.sr):
                dur = get_duration_sec(wav_path)
                rows.append({
                    "split": split,
                    "label": label,
                    "video_path": str(v),
                    "audio_path": str(wav_path),
                    "duration_sec": f"{dur:.3f}",
                })
                n_ok += 1

    man = out_root / "audio_manifest.csv"
    with man.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["split","label","video_path","audio_path","duration_sec"])
        w.writeheader(); w.writerows(rows)

    print(f"\nVideos seen: {n_seen}")
    print(f"Extracted OK: {n_ok}")
    print(f"Skipped (no audio stream): {n_noaudio}")
    print(f"Skipped (unknown label folder): {n_unknown}")
    # per-split/class counts
    from collections import Counter
    c = Counter((r["split"], r["label"]) for r in rows)
    for split in ("train","val","test"):
        for lbl in ("violent","non_violent"):
            print(f"{split:5s} {lbl:13s}: {c.get((split,lbl),0)}")
    print(f"\nManifest -> {man}")

if __name__ == "__main__":
    main()
