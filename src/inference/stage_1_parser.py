"""Stage 1: parse a single serial line into 8 raw sensor floats.

Firmware (config/main.cpp) emits, mỗi ~60ms:
    FLX:<thumb>,<index>,<middle>,<ring>,<pinky>,<imu_x>,<imu_y>,<imu_z>

Trong đó:
- 5 flex: int 0..4095 (giá trị thô từ ADC, sẽ được stage_4 chia /4095)
- 3 IMU: float (Euler degrees), có thể âm (vd `-3.21`)

Firmware cũng in boot/calibration messages ("=== GLOVE BOOT ===",
"Offsets -> T:...") — module skip silently.
"""
from __future__ import annotations

FLX_PREFIX = "FLX:"
EXPECTED_TOKEN_COUNT = 8


def parse_flx_line(line: str) -> list[float] | None:
    """Return [flex1..flex5, imu_x, imu_y, imu_z] or None if the line is not a valid FLX frame."""
    if not line:
        return None
    line = line.strip()
    if not line.startswith(FLX_PREFIX):
        return None

    payload = line[len(FLX_PREFIX):]
    tokens = payload.split(",")
    if len(tokens) != EXPECTED_TOKEN_COUNT:
        return None

    try:
        return [float(t) for t in tokens]
    except ValueError:
        return None


if __name__ == "__main__":
    samples = [
        "FLX:120,98,150,77,66,12.45,-3.21,88.00",     # exact format from glove image
        "FLX:12,34,56,78,90,180.50,90.20,45.10",
        "FLX:0,0,0,0,0,0.00,0.00,0.00\n",
        "FLX:200,200,200,200,200,-179.99,-89.50,0.00", # negative IMU
        "=== GLOVE BOOT ===",
        "Offsets -> T:1234 I:5678",
        "FLX:1,2,3",
        "FLX:1,2,3,4,5,abc,90,45",
        "",
    ]
    for s in samples:
        print(f"{s!r:55s} -> {parse_flx_line(s)}")
