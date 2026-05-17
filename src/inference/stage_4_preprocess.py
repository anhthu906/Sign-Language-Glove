"""Stage 4: cyclic-encode imu_x and apply the trained StandardScaler.

Mirrors notebooks/Pipeline.ipynb exactly:
  window (20, 8)  [flex1..5 (normalized), imu_x, imu_y, imu_z]
    --cyclic-encode--> (20, 9)  [flex1..5, imu_y, imu_z, sin_imu_x, cos_imu_x]
    --standardize---> (20, 9)   zero-mean / unit-scale
    --batch-axis----> (1, 20, 9) float32

Stage này giả định input flex đã ở khoảng normalized training (~0.001–0.07).
CSV trong data/raw/Good Data/ đã lưu sẵn dạng này.

TODO (khi cắm glove vật lý lại): kiểm tra xem firmware config/main.cpp gửi raw
int hay đã normalize. Nếu raw int → thêm `window[:, 0:5] /= 4095.0` ở đầu
`preprocess_window`. Nếu đã normalize trong firmware → giữ nguyên.
"""
from __future__ import annotations

import numpy as np

ENCODED_FEATURE_COUNT = 9
EXPECTED_FEATURE_NAMES = (
    "flex1", "flex2", "flex3", "flex4", "flex5",
    "imu_y", "imu_z", "sin_imu_x", "cos_imu_x",
)


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
    """Full pipeline: (20, 8) raw-from-firmware -> (1, 20, 9) ready for model.predict."""
    encoded = cyclic_encode(raw_window)
    scaled = apply_scaler(encoded, mean, scale)
    return scaled[np.newaxis, ...]


if __name__ == "__main__":
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    scaler_path = os.path.normpath(os.path.join(here, "..", "..", "data", "processed", "scaler.npz"))

    mean, scale = load_scaler(scaler_path)
    print(f"scaler mean: {mean}")
    print(f"scaler scale: {scale}")

    # Dummy window: all zeros except imu_x = 0 -> sin=0, cos=1
    raw = np.zeros((20, 8), dtype=np.float32)
    encoded = cyclic_encode(raw)
    print(f"\nencoded shape: {encoded.shape}")
    print(f"encoded row 0: {encoded[0]}  (sin_imu_x=0, cos_imu_x=1 expected)")

    x = preprocess_window(raw, mean, scale)
    print(f"\nfinal shape: {x.shape}, dtype: {x.dtype}")
    print(f"per-feature mean after scaling: {x[0].mean(axis=0)}")

    # Sanity: feed a row that matches the first training-CSV row.
    firmware_like = np.tile([31, 11, 5, 3, 15, 342.87, -13.31, 39.25], (20, 1)).astype(np.float32)
    x2 = preprocess_window(firmware_like, mean, scale)
    print(f"x2 row 0: {x2[0, 0]}  (each value should be small, in [-3, 3] roughly)")
