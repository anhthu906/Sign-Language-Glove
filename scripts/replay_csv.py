"""Replay a training CSV as FLX: lines on stdout, to test predict_pipeline.py
without the physical glove.

The training CSVs in data/raw/Good Data/ store flex values already divided by
4095 (12-bit ADC max). The firmware in config/main.cpp emits raw integers, so
this script multiplies flex columns back by 4095 to simulate the wire format.

Usage:
    python3 scripts/replay_csv.py "data/raw/Good Data/tôi3_1778996226.csv" \\
        | python3 -m src.inference.predict_pipeline --stdin
"""
import argparse
import csv
import sys


FLEX_SCALE_BACK = 4095


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv_path")
    args = ap.parse_args()

    try:
        with open(args.csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                f1 = int(round(float(row["flex1"]) * FLEX_SCALE_BACK))
                f2 = int(round(float(row["flex2"]) * FLEX_SCALE_BACK))
                f3 = int(round(float(row["flex3"]) * FLEX_SCALE_BACK))
                f4 = int(round(float(row["flex4"]) * FLEX_SCALE_BACK))
                f5 = int(round(float(row["flex5"]) * FLEX_SCALE_BACK))
                ix = float(row["imu_x"])
                iy = float(row["imu_y"])
                iz = float(row["imu_z"])
                print(f"FLX:{f1},{f2},{f3},{f4},{f5},{ix:.2f},{iy:.2f},{iz:.2f}", flush=True)
    except BrokenPipeError:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
