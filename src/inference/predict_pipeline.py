"""Realtime sign-language prediction from a glove serial stream.

Composes the five stages (parser, stream, buffer, preprocess, predictor) into a
single loop. Run from the project root.

Examples
--------
Live from a connected glove (auto-detects /dev/ttyACM0 or /dev/ttyUSB0):
    python3 -m src.inference.predict_pipeline

Specific serial port:
    python3 -m src.inference.predict_pipeline --port /dev/ttyACM0

Replay from a saved CSV (no hardware needed). NOTE: training CSVs store flex
values already divided by 4095, so the replay must multiply back to simulate the
raw integer stream the firmware emits:
    python3 scripts/replay_csv.py "data/raw/Good Data/tôi3_xxx.csv" \
        | python3 -m src.inference.predict_pipeline --stdin
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

from . import preprocess, stream
from .buffer import FrameBuffer
from .parser import parse_flx_line
from .predictor import Predictor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))

DEFAULT_MODEL = os.path.join(ROOT, "results", "models", "model_best.h5")
DEFAULT_SCALER = os.path.join(ROOT, "data", "processed", "scaler.npz")
DEFAULT_LABELS = os.path.join(ROOT, "data", "processed", "label_classes.npy")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group()
    src.add_argument("--port", help="serial port (e.g. /dev/ttyACM0). Auto-detect if omitted.")
    src.add_argument("--stdin", action="store_true", help="read FLX lines from stdin instead of serial")
    p.add_argument("--baud", type=int, default=stream.DEFAULT_BAUD)
    p.add_argument("--stride", type=int, default=10, help="frames between predictions (default 10, matches training stride)")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--scaler", default=DEFAULT_SCALER)
    p.add_argument("--labels", default=DEFAULT_LABELS)
    p.add_argument("--conf-threshold", type=float, default=0.0, help="only print predictions with confidence >= this")
    return p.parse_args()


def format_prediction(pred, ts: datetime) -> str:
    top3_str = "  ".join(f"{lbl}/{c:.2f}" for lbl, c in pred.top3)
    return f"[{ts.strftime('%H:%M:%S')}] {pred.label:<12s} conf={pred.confidence:.2f}   top3: {top3_str}"


def main() -> int:
    args = parse_args()

    print(f"[init] model:  {args.model}", file=sys.stderr)
    print(f"[init] scaler: {args.scaler}", file=sys.stderr)
    print(f"[init] labels: {args.labels}", file=sys.stderr)

    mean, scale = preprocess.load_scaler(args.scaler)
    predictor = Predictor(args.model, args.labels)
    print(f"[init] classes ({len(predictor.classes)}): {list(predictor.classes)}", file=sys.stderr)

    buf = FrameBuffer(stride=args.stride)
    line_source = stream.auto_stream(args)

    print(f"[ready] stride={args.stride}  conf_threshold={args.conf_threshold}", file=sys.stderr)

    last_warmup_print = -1
    for line in line_source:
        frame = parse_flx_line(line)
        if frame is None:
            continue

        ready = buf.push(frame)
        if not ready:
            cur, total = buf.warmup_state()
            if cur < total and cur != last_warmup_print and cur % 5 == 0:
                print(f"[warmup] {cur}/{total}", file=sys.stderr)
                last_warmup_print = cur
            continue

        x = preprocess.preprocess_window(buf.as_array(), mean, scale)
        pred = predictor.predict(x)
        if pred.confidence >= args.conf_threshold:
            print(format_prediction(pred, datetime.now()), flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
