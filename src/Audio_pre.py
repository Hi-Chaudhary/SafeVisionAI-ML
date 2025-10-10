import argparse, random, csv, json, warnings
from pathlib import Path
from collections import defaultdict, Counter

import torch
import torchaudio

# Silence torchaudio future warnings (purely cosmetic)
#warnings.filterwarnings("ignore", message=r"In 2\.9, this function's implementation will be changed")

# ---------- tiny utils ----------
def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def infer_label(fname: str) -> str:
    # Kaggle VSD: "noviolence_*.wav" are negatives; everything else = violent
    f = fname.lower()
    return "non_violent" if "noviolence" in f or "no_violence" in f else "violent"

def load_mono_resample(path: Path, target_sr: int) -> tuple[torch.Tensor, int]:
    wav, sr = torchaudio.load(str(path))         # [C, T]
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)      # to mono
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
        sr = target_sr
    return wav.squeeze(0), sr                     # [T]

def save_wav_pcm16(path: Path, wav_1d: torch.Tensor, sr: int):
    ensure_dir(path.parent)
    torchaudio.save(str(path), wav_1d.unsqueeze(0), sr, encoding="PCM_S", bits_per_sample=16)

def segment_waveform(wav: torch.Tensor, sr: int, win_s: float, hop_s: float,
                     pad_last=True, min_rms=0.0):
    win = int(win_s * sr); hop = int(hop_s * sr)
    T = wav.numel(); i = 0
    out = []
    def rms(x): return float(torch.sqrt((x**2).mean() + 1e-12))
    while i < T:
        j = i + win
        chunk = wav[i:j]
        if chunk.numel() < win:
            if not pad_last: break
            chunk = torch.nn.functional.pad(chunk, (0, win - chunk.numel()))
        if min_rms <= 0 or rms(chunk) >= min_rms:
            out.append((i, j, chunk))
        i += hop
    return out

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Simple audio prep: standardize -> split -> segment -> manifests")
    ap.add_argument("--raw_dir",   required=True, help="Folder containing the original WAVs (flat or nested)")
    ap.add_argument("--out_root",  required=True, help="Output root folder (will create std16k/, segments/, manifests/)")
    ap.add_argument("--sr",        type=int, default=16000, help="Target sample rate (default 16000)")
    ap.add_argument("--win",       type=float, default=4.0,  help="Segment length seconds (default 4.0)")
    ap.add_argument("--hop",       type=float, default=2.0,  help="Segment hop seconds (default 2.0)")
    ap.add_argument("--train",     type=float, default=0.8,  help="Train ratio (default 0.8)")
    ap.add_argument("--val",       type=float, default=0.1,  help="Val ratio (default 0.1)")
    ap.add_argument("--seed",      type=int,   default=42,   help="Random seed")
    ap.add_argument("--min_rms",   type=float, default=0.0,  help="Skip near-silence segments below this RMS (default 0.0)")
    args = ap.parse_args()

    random.seed(args.seed)

    RAW  = Path(args.raw_dir)
    ROOT = Path(args.out_root)
    STD  = ROOT / "std16k"
    SEG  = ROOT / "segments"
    MAN  = ROOT / "manifests"
    for p in (STD, SEG, MAN): ensure_dir(p)

    if not RAW.exists():
        raise SystemExit(f"--raw_dir not found: {RAW.resolve()}")

    # 1) Discover WAVs (recursive)
    files = [p for p in RAW.rglob("*") if p.suffix.lower() == ".wav" and p.is_file()]
    if not files:
        raise SystemExit(f"No .wav files found under {RAW.resolve()}")

    # 2) Standardize to mono/16k under std16k/<label>/
    std_paths_by_label = defaultdict(list)
    durations = []
    for f in files:
        label = infer_label(f.name)
        wav, sr = load_mono_resample(f, args.sr)
        out = STD / label / f.name
        save_wav_pcm16(out, wav, sr)
        std_paths_by_label[label].append(out)
        durations.append(float(wav.numel() / sr))

    # 3) Stratified split (per file)
    def split_list(lst):
        lst = lst[:]  # copy
        random.shuffle(lst)
        n = len(lst)
        n_tr = int(round(args.train * n))
        n_va = int(round(args.val   * n))
        tr = lst[:n_tr]
        va = lst[n_tr:n_tr + n_va]
        te = lst[n_tr + n_va:]
        return tr, va, te

    splits = {"train": [], "val": [], "test": []}
    for label, lst in std_paths_by_label.items():
        tr, va, te = split_list(lst)
        splits["train"] += [(p, label) for p in tr]
        splits["val"]   += [(p, label) for p in va]
        splits["test"]  += [(p, label) for p in te]

    # 4) Segment each standardized file into segments/<split>/<label>/*.wav
    seg_rows = []
    for split, items in splits.items():
        for p, label in items:
            wav, sr = load_mono_resample(p, args.sr)  # idempotent
            base = p.stem
            for (i, j, chunk) in segment_waveform(wav, sr, args.win, args.hop,
                                                  pad_last=True, min_rms=args.min_rms):
                out = SEG / split / label / f"{base}_{i:010d}_{j:010d}.wav"
                save_wav_pcm16(out, chunk, sr)
                seg_rows.append({
                    "split": split,
                    "label": label,
                    "audio_path": str(out).replace("\\", "/"),
                    "start_sec": round(i / sr, 2),
                    "duration_sec": round(args.win, 2),
                })

    # 5) Write manifests (segments + balanced)
    ensure_dir(MAN)
    seg_csv = MAN / "segments_manifest.csv"
    with seg_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["split", "label", "audio_path", "start_sec", "duration_sec"])
        w.writeheader(); w.writerows(seg_rows)

    # build a 50/50 balanced version for each split
    by_split = defaultdict(list)
    for r in seg_rows: by_split[r["split"]].append(r)
    balanced = []
    for split, rows in by_split.items():
        v  = [r for r in rows if r["label"] == "violent"]
        nv = [r for r in rows if r["label"] == "non_violent"]
        random.shuffle(v); random.shuffle(nv)
        n = min(len(v), len(nv))
        balanced += v[:n] + nv[:n]
    random.shuffle(balanced)

    bal_csv = MAN / "balanced_segments_manifest.csv"
    with bal_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["split", "label", "audio_path", "start_sec", "duration_sec"])
        w.writeheader(); w.writerows(balanced)

    # 6) Print a tiny summary
    counts = Counter((r["split"], r["label"]) for r in seg_rows)
    bal_counts = Counter((r["split"], r["label"]) for r in balanced)
    print(json.dumps({
        "num_raw_files": len(files),
        "avg_file_duration_sec": round(sum(durations)/max(1,len(durations)), 2),
        "segments_per_split_label": {f"{s}:{l}": counts.get((s,l), 0)
                                     for s in ("train","val","test") for l in ("violent","non_violent")},
        "balanced_segments_per_split_label": {f"{s}:{l}": bal_counts.get((s,l), 0)
                                     for s in ("train","val","test") for l in ("violent","non_violent")},
        "sr": args.sr, "win_sec": args.win, "hop_sec": args.hop
    }, indent=2))
    print("\nWrote:", seg_csv)
    print("Wrote:", bal_csv)
    print("Done.")
if __name__ == "__main__":
    main()
