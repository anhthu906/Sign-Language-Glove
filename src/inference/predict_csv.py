"""Offline CSV prediction pipeline.

Đọc trực tiếp 1 file CSV (schema giống data/raw/Good Data/: flex1..flex5,
imu_x, imu_y, imu_z[, SIGN]), trượt window cố định, batched model.predict,
in ra label đã remap.

Nhanh hơn predict_pipeline.py vì:
  - Không qua serial/stdin
  - Không qua FLX: parser
  - Gom toàn bộ window thành 1 batch → 1 call model.predict cho cả file
    (thay vì ~1000 call riêng lẻ)

Reuse:
  - stage_4_preprocess.cyclic_encode + apply_scaler  (KHÔNG /4095, vì CSV đã normalize)
  - stage_5_predictor.Predictor + LABEL_REMAP

Usage:
    python3 -m src.inference.predict_csv "data/raw/Test/xinchao2_1778993189.csv"
    python3 -m src.inference.predict_csv path/to/file.csv --summary
    python3 -m src.inference.predict_csv path/to/file.csv --conf-threshold 0.7
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import Counter

import numpy as np

from .stage_4_preprocess import apply_scaler, cyclic_encode, load_scaler
from .stage_5_predictor import Predictor, remap_label

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))

DEFAULT_MODEL = os.path.join(ROOT, "results", "models", "model_best.h5")
DEFAULT_SCALER = os.path.join(ROOT, "data", "processed", "scaler.npz")
DEFAULT_LABELS = os.path.join(ROOT, "data", "processed", "label_classes.npy")

FEATURE_COLS = ["flex1", "flex2", "flex3", "flex4", "flex5", "imu_x", "imu_y", "imu_z"]
WINDOW_SIZE = 20


def read_csv_features(csv_path: str) -> tuple[np.ndarray, str | None]:
    """Read a CSV and return (frames (N, 8) float32, sign_label_if_present)."""
    rows: list[list[float]] = []
    sign: str | None = None
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        missing = [c for c in FEATURE_COLS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"CSV {csv_path} missing columns: {missing}")
        for row in reader:
            rows.append([float(row[c]) for c in FEATURE_COLS])
            if sign is None and "SIGN" in row:
                sign = row["SIGN"]
    if not rows:
        raise ValueError(f"CSV {csv_path} is empty")
    return np.asarray(rows, dtype=np.float32), sign


def slide_windows(frames: np.ndarray, window_size: int, stride: int) -> np.ndarray:
    """frames: (N, 8) -> (B, window_size, 8)."""
    n = frames.shape[0]
    if n < window_size:
        raise ValueError(f"need at least {window_size} frames, got {n}")
    starts = range(0, n - window_size + 1, stride)
    return np.stack([frames[s : s + window_size] for s in starts], axis=0)


def preprocess_batch(windows: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """(B, 20, 8) -> (B, 20, 9) ready for model.predict."""
    b, t, _ = windows.shape
    flat = windows.reshape(b * t, 8)
    encoded = cyclic_encode(flat)                  # (B*T, 9)
    scaled = apply_scaler(encoded, mean, scale)    # (B*T, 9)
    return scaled.reshape(b, t, 9)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv_path")
    p.add_argument("--stride", type=int, default=10, help="frames between windows (default 10, matches training)")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--scaler", default=DEFAULT_SCALER)
    p.add_argument("--labels", default=DEFAULT_LABELS)
    p.add_argument("--conf-threshold", type=float, default=0.0)
    p.add_argument("--summary", action="store_true", help="chỉ in distribution của label cuối, không in từng window")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    frames, sign = read_csv_features(args.csv_path)
    print(f"[init] csv: {args.csv_path}  rows={len(frames)}  sign_in_file={sign!r}", file=sys.stderr)

    windows = slide_windows(frames, WINDOW_SIZE, args.stride)
    print(f"[init] windows: {windows.shape}  (stride={args.stride})", file=sys.stderr)

    mean, scale = load_scaler(args.scaler)
    predictor = Predictor(args.model, args.labels)
    print(f"[init] classes ({len(predictor.classes)}): {list(predictor.classes)}", file=sys.stderr)

    x = preprocess_batch(windows, mean, scale)
    probs = predictor.model.predict(x, batch_size=args.batch_size, verbose=0)  # (B, n_classes)

    top_idx = np.argmax(probs, axis=1)
    top_conf = probs[np.arange(len(probs)), top_idx]

    labels = [remap_label(str(predictor.classes[i])) for i in top_idx]

    if args.summary:
        counter = Counter(
            lbl for lbl, c in zip(labels, top_conf) if c >= args.conf_threshold
        )
        total = sum(counter.values())
        print(f"\n# {args.csv_path}")
        print(f"# {total} windows above conf-threshold {args.conf_threshold}")
        if sign:
            print(f"# sign_in_file = {sign!r}")
        for lbl, n in counter.most_common():
            pct = 100 * n / total if total else 0
            print(f"  {n:6d}  ({pct:5.1f}%)  {lbl}")
    else:
        for i, (lbl, c) in enumerate(zip(labels, top_conf)):
            if c < args.conf_threshold:
                continue
            start_frame = i * args.stride
            print(f"window[{i:4d}] frame={start_frame:5d}  {lbl:<12s}  conf={c:.2f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
