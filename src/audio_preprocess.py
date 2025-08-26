# audio_preprocess.py
import os, sys, math, warnings, uuid
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import librosa
import soundfile as sf
import noisereduce as nr

warnings.filterwarnings("ignore", category=UserWarning)

# ========= USER CONFIG =========
ROOT = r"D:\SEM2\AML\SafeVisionAIML"                  # <-- CHANGE if needed
RAW_ROOT = Path(ROOT) / "data" / "audio" / "raw"
PROC_ROOT = Path(ROOT) / "data" / "audio" / "processed"
MANI_ROOT = Path(ROOT) / "data" / "audio" / "manifests"

SAMPLE_RATE = 16000
TARGET_SECS = 5.0
TARGET_SAMPLES = int(SAMPLE_RATE * TARGET_SECS)

DO_TRIM_SILENCE = True
TRIM_TOP_DB = 30

DO_NOISE_REDUCTION = True           # turn off if it hurts
NOISE_REDUCE_PROP = 0.8

# Datasets present (toggle if a folder is missing)
USE_VIOLENCE_KAGGLE = True
USE_HUMAN_SCREAMS   = True          # optional
USE_URBAN_SOUND_8K  = True         # set True if you downloaded it

# UrbanSound8K classes to treat as NON-VIOLENT (exclude gun_shot)
US8K_NONVIOLENT_CLASSES = {
    "air_conditioner","car_horn","children_playing","dog_bark","drilling",
    "engine_idling","jackhammer","siren","street_music"
}
RANDOM_STATE = 42
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.80, 0.10, 0.10
# =================================


def ensure_dirs():
    for split in ["train", "val", "test"]:
        for label in ["violent", "non_violent"]:
            (PROC_ROOT / split / label).mkdir(parents=True, exist_ok=True)
    MANI_ROOT.mkdir(parents=True, exist_ok=True)


def list_audio_files(folder, exts=(".wav",".mp3",".flac",".ogg",".m4a",".aac",".wma",".au",".aiff",".aif")):
    if not Path(folder).exists():
        return []
    return [str(p) for p in Path(folder).rglob("*") if p.suffix.lower() in exts]


def gather_records():
    """Return a DataFrame with columns:
       ['orig_path','label','dataset','orig_file','class_hint']
    """
    records = []

    # A) violence_kaggle
    if USE_VIOLENCE_KAGGLE:
        base = RAW_ROOT / "violence_kaggle"
        for label in ["violent","non_violent"]:
            for p in list_audio_files(base/label):
                records.append({"orig_path": p, "label": label, "dataset":"violence_kaggle",
                                "orig_file": Path(p).name, "class_hint": label})

    # B) human_screams_kaggle -> violent
    if USE_HUMAN_SCREAMS:
        base = RAW_ROOT / "human_screams_kaggle" / "scream"
        for p in list_audio_files(base):
            records.append({"orig_path": p, "label": "violent", "dataset":"human_screams",
                            "orig_file": Path(p).name, "class_hint":"scream"})

    # C) UrbanSound8K -> non_violent classes only
    if USE_URBAN_SOUND_8K:
        meta = RAW_ROOT / "UrbanSound8K" / "metadata" / "UrbanSound8K.csv"
        audio_root = RAW_ROOT / "UrbanSound8K" / "audio"
        if meta.exists():
            df = pd.read_csv(meta)
            # columns usually include: slice_file_name, fold, class, classID, ...
            df = df[df["class"].isin(US8K_NONVIOLENT_CLASSES)].copy()
            for _, row in df.iterrows():
                fold = f"fold{int(row['fold'])}"
                fpath = audio_root / fold / row["slice_file_name"]
                if fpath.exists():
                    records.append({"orig_path": str(fpath),
                                    "label": "non_violent",
                                    "dataset":"UrbanSound8K",
                                    "orig_file": row["slice_file_name"],
                                    "class_hint": row["class"]})

    recs = pd.DataFrame(records)
    if recs.empty:
        print("No audio found. Check RAW folders and toggles at top of script.")
        sys.exit(1)
    return recs


def stratified_split(df):
    # 80/10/10 stratified by label
    train_df, tmp_df = train_test_split(
        df, test_size=(1-TRAIN_RATIO), stratify=df["label"], random_state=RANDOM_STATE
    )
    rel = TEST_RATIO / (VAL_RATIO + TEST_RATIO)
    val_df, test_df = train_test_split(
        tmp_df, test_size=rel, stratify=tmp_df["label"], random_state=RANDOM_STATE
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def load_audio(path, target_sr=SAMPLE_RATE):
    # librosa handles resample & mono
    y, sr = librosa.load(path, sr=target_sr, mono=True)
    return y.astype(np.float32), target_sr


def trim_silence(y):
    if not DO_TRIM_SILENCE:
        return y
    yt, _ = librosa.effects.trim(y, top_db=TRIM_TOP_DB)
    return yt if len(yt) > 0 else y


def noise_reduce(y, sr):
    if not DO_NOISE_REDUCTION:
        return y
    # Estimate noise from the first 0.5s (or all if shorter)
    nsamp = min(len(y), int(0.5 * sr))
    noise_clip = y[:nsamp] if nsamp > 0 else y
    try:
        return nr.reduce_noise(y=y, sr=sr, y_noise=noise_clip, prop_decrease=NOISE_REDUCE_PROP, verbose=False)
    except Exception:
        return y  # fail-safe


def peak_normalize(y, peak_db=-1.0):
    # Simple peak normalization to target dBFS (approx by scaling max to ~0.89 for -1 dBFS)
    peak = np.max(np.abs(y)) + 1e-9
    y = y / peak
    return y * (10 ** (peak_db / 20.0))  # scale to target peak


def center_crop(y, target_len):
    if len(y) <= target_len:
        return y
    start = (len(y) - target_len) // 2
    return y[start:start+target_len]


def pad_to(y, target_len):
    if len(y) >= target_len:
        return y
    pad = target_len - len(y)
    return np.pad(y, (0, pad), mode="constant")


def chunkify_train(y, target_len):
    """Non-overlapping 5s windows for TRAIN. If < target, pad to 1 window."""
    if len(y) < target_len:
        return [pad_to(y, target_len)]
    chunks = []
    i = 0
    while i + target_len <= len(y):
        chunks.append(y[i:i+target_len])
        i += target_len
    # if tail is small (< 1s) ignore; if moderate, pad tail to 5s as extra window
    tail = y[i:]
    if len(tail) >= int(0.6 * target_len):  # keep sizeable tail
        chunks.append(pad_to(tail, target_len))
    return chunks if chunks else [pad_to(y, target_len)]


def process_and_save(split, rows):
    mani_rows = []
    for _, r in tqdm(rows.iterrows(), total=len(rows), desc=f"Processing {split}"):
        try:
            y, sr = load_audio(r["orig_path"], SAMPLE_RATE)
            y = trim_silence(y)
            y = noise_reduce(y, sr)
            y = peak_normalize(y, peak_db=-1.0)

            if split == "train":
                windows = chunkify_train(y, TARGET_SAMPLES)
            else:
                y5 = center_crop(y, TARGET_SAMPLES)
                y5 = pad_to(y5, TARGET_SAMPLES)
                windows = [y5]

            seg_idx = 0
            for w in windows:
                # Path & name: <dataset>__<orig>__segK__.wav
                base_name = f"{r['dataset']}__{Path(r['orig_file']).stem}__seg{seg_idx}.wav"
                out_dir = PROC_ROOT / split / r["label"]
                out_path = out_dir / base_name

                sf.write(str(out_path), w, SAMPLE_RATE, subtype="PCM_16")

                mani_rows.append({
                    "path": str(out_path),
                    "label": r["label"],
                    "duration_sec": round(len(w) / SAMPLE_RATE, 3),
                    "orig_dataset": r["dataset"],
                    "orig_file": r["orig_file"],
                    "segment_idx": seg_idx
                })
                seg_idx += 1

        except Exception as e:
            print(f"[WARN] Failed: {r['orig_path']} -> {e}")
            continue

    mani = pd.DataFrame(mani_rows)
    return mani


def main():
    ensure_dirs()
    all_df = gather_records()
    print(f"Found {len(all_df)} raw files | label counts:\n{all_df['label'].value_counts()}")

    train_df, val_df, test_df = stratified_split(all_df)
    print(f"Split sizes: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    mani_train = process_and_save("train", train_df)
    mani_val   = process_and_save("val", val_df)
    mani_test  = process_and_save("test", test_df)

    mani_train.to_csv(MANI_ROOT / "train.csv", index=False)
    mani_val.to_csv(MANI_ROOT / "val.csv", index=False)
    mani_test.to_csv(MANI_ROOT / "test.csv", index=False)

    print("\nDone ✅")
    print(f"Saved manifests:\n  {MANI_ROOT / 'train.csv'}\n  {MANI_ROOT / 'val.csv'}\n  {MANI_ROOT / 'test.csv'}")


if __name__ == "__main__":
    main()
