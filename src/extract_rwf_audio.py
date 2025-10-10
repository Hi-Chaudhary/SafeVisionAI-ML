import subprocess, json, csv
from pathlib import Path
from collections import Counter

RWF = Path(r"D:\SEM2\AML\SafeVisionAIML\Data\Video\rwf2000")
OUT = Path(r"D:\SEM2\AML\SafeVisionAIML\Data\Audio\rwf2000extracted")
MAP = {"Fight": "violent", "NonFight": "non_violent"}
SPLITS = ["train", "val", "test"]

def run(cmd):
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

def has_audio(video: Path) -> bool:
    # Use ffprobe to detect an audio stream
    p = run(["ffprobe","-v","error","-show_streams","-select_streams","a","-of","json",str(video)])
    if p.returncode != 0:
        return False
    try:
        info = json.loads(p.stdout or "{}")
        return len(info.get("streams", [])) > 0
    except json.JSONDecodeError:
        return False

def extract_audio(video: Path, wav: Path) -> bool:
    wav.parent.mkdir(parents=True, exist_ok=True)
    p = run([
        "ffmpeg","-hide_banner","-loglevel","error","-y",
        "-i",str(video),
        "-vn","-ac","1","-ar","16000","-acodec","pcm_s16le",
        str(wav)
    ])
    return p.returncode == 0

def duration_sec(wav: Path) -> float:
    p = run(["ffprobe","-v","error",
             "-show_entries","format=duration",
             "-of","default=noprint_wrappers=1:nokey=1",
             str(wav)])
    try:
        return float(p.stdout.strip())
    except:
        return -1.0

def main():
    rows = []
    for split in SPLITS:
        for src_cls, tgt_cls in MAP.items():
            vdir = RWF / split / src_cls
            if not vdir.exists(): 
                continue
            outdir = OUT / split / tgt_cls
            for v in sorted(vdir.glob("*")):
                if not v.is_file():
                    continue
                if not has_audio(v):
                    continue  # skip videos with no audio track
                wav = outdir / (v.stem + ".wav")
                if extract_audio(v, wav):
                    dur = duration_sec(wav)
                    rows.append({
                        "split": split,
                        "label": tgt_cls,
                        "video_path": str(v),
                        "audio_path": str(wav),
                        "duration_sec": f"{dur:.3f}"
                    })

    OUT.mkdir(parents=True, exist_ok=True)
    manifest = OUT / "audio_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["split","label","video_path","audio_path","duration_sec"])
        w.writeheader(); w.writerows(rows)

    # Summary
    cnt = Counter((r["split"], r["label"]) for r in rows)
    print(f"Wrote {len(rows)} rows -> {manifest}")
    for split in SPLITS:
        for lbl in ("violent","non_violent"):
            print(f"{split:5s} {lbl:13s}: {cnt.get((split,lbl),0)}")

if __name__ == "__main__":
    main()
