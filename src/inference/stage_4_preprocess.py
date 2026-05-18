"""Stage 4: normalize flex, cyclic-encode imu_x, and apply the trained StandardScaler.

Mirrors notebooks/Pipeline.ipynb (data was saved after the friend's
config/realtime_nobno.py applied min-max scaling `(v - 0) / (4095 - 0)` to
each raw flex int from the glove).

Pipeline A (glove → predict_pipeline.py) chain:
  raw window (20, 8) [flex1..5 (int from firmware), imu_x, imu_y, imu_z]
    --normalize-flex-> (20, 8)   flex_i /= 4095        (handled by `preprocess_window`)
    --cyclic-encode--> (20, 9)   [flex1..5, imu_y, imu_z, sin_imu_x, cos_imu_x]
    --standardize---> (20, 9)    zero-mean / unit-scale
    --batch-axis----> (1, 20, 9) float32

Pipeline B (CSV → predict_csv.py) calls `cyclic_encode` + `apply_scaler`
directly, bypassing `normalize_raw` because training CSVs already store
flex in normalized units.
"""
from __future__ import annotations

import numpy as np

FLEX_DIVISOR = 4095.0  # 12-bit ADC max, matches realtime_nobno.py min-max scaling
ENCODED_FEATURE_COUNT = 9
EXPECTED_FEATURE_NAMES = (
    "flex1", "flex2", "flex3", "flex4", "flex5",
    "imu_y", "imu_z", "sin_imu_x", "cos_imu_x",
)


def normalize_raw(window: np.ndarray) -> np.ndarray:
    """Divide the 5 flex columns by FLEX_DIVISOR; leave IMU columns untouched.

    Mirrors `(v - FLEX_MIN) / (FLEX_MAX - FLEX_MIN)` with FLEX_MIN=0, FLEX_MAX=4095
    from config/realtime_nobno.py (line 36-37, 386-389).
    """
    if window.ndim != 2 or window.shape[1] != 8:
        raise ValueError(f"normalize_raw expects (N, 8), got {window.shape}")
    out = window.astype(np.float32, copy=True)
    out[:, 0:5] /= FLEX_DIVISOR
    return out


def load_scaler(scaler_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load mean and scale (shape (9,) each) from scaler.npz."""
    data = np.load(scaler_path)
    mean = data["mean"].astype(np.float32)
    scale = data["scale"].astype(np.float32)
    if mean.shape != (ENCODED_FEATURE_COUNT,) or scale.shape != (ENCODED_FEATURE_COUNT,):
        raise ValueError(
            f"scaler shape mismatch: mean={mean.shape}, scale={scale.shape}, "
            f"expected ({ENCODED_FEATURE_COUNT},)"
        )
    return mean, scale


def cyclic_encode(window: np.ndarray) -> np.ndarray:
    """(N, 8) [flex×5, imu_x, imu_y, imu_z] -> (N, 9) [flex×5, imu_y, imu_z, sin_imu_x, cos_imu_x]."""
    if window.ndim != 2 or window.shape[1] != 8:
        raise ValueError(f"cyclic_encode expects (N, 8), got {window.shape}")

    flex = window[:, 0:5]
    imu_x = window[:, 5]
    imu_y = window[:, 6:7]
    imu_z = window[:, 7:8]

    rad = np.deg2rad(imu_x)
    sin_x = np.sin(rad).reshape(-1, 1)
    cos_x = np.cos(rad).reshape(-1, 1)

    return np.concatenate([flex, imu_y, imu_z, sin_x, cos_x], axis=1).astype(np.float32)


def apply_scaler(window_9: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return ((window_9 - mean) / scale).astype(np.float32)


def preprocess_window(raw_window: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Full Pipeline A pipeline: (20, 8) raw-from-firmware -> (1, 20, 9) ready for model.predict.

    Includes /4095 normalize. For already-normalized input (CSV), call
    `cyclic_encode` + `apply_scaler` directly instead.
    """
    normalized = normalize_raw(raw_window)
    encoded = cyclic_encode(normalized)
    scaled = apply_scaler(encoded, mean, scale)
    return scaled[np.newaxis, ...]


if __name__ == "__main__":
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    scaler_path = os.path.normpath(os.path.join(here, "..", "..", "data", "processed", "scaler.npz"))

    mean, scale = load_scaler(scaler_path)
    print(f"scaler mean: {mean}")
    print(f"scaler scale: {scale}")

    # Sanity: feed a raw firmware-style row (FLX:120,98,150,77,66,12.45,-3.21,88.00)
    firmware_like = np.tile([120, 98, 150, 77, 66, 12.45, -3.21, 88.00], (20, 1)).astype(np.float32)
    norm = normalize_raw(firmware_like)
    print(f"\nraw row:        {firmware_like[0]}")
    print(f"after /4095:    {norm[0]}  (flex should be ~0.01-0.04, IMU unchanged)")

    x = preprocess_window(firmware_like, mean, scale)
    print(f"\nfinal shape:    {x.shape}, dtype: {x.dtype}")
    print(f"x row 0:        {x[0, 0]}  (each value should be in [-3, 3] roughly)")
