"""Stage 3: rolling window buffer + stride trigger.

After the buffer fills to `window_size` frames the first time, `push()`
returns True every `stride` newly pushed frames so the caller knows when
to run a prediction.
"""
from __future__ import annotations

from collections import deque

import numpy as np

WINDOW_SIZE = 20
DEFAULT_STRIDE = 10
RAW_FEATURE_COUNT = 8  # flex1..5, imu_x, imu_y, imu_z


class FrameBuffer:
    def __init__(self, window_size: int = WINDOW_SIZE, stride: int = DEFAULT_STRIDE):
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")
        self.window_size = window_size
        self.stride = stride
        self._buf: deque[list[float]] = deque(maxlen=window_size)
        self._since_last_trigger = 0
        self._ever_full = False

    def push(self, frame: list[float]) -> bool:
        if len(frame) != RAW_FEATURE_COUNT:
            raise ValueError(
                f"expected {RAW_FEATURE_COUNT} features per frame, got {len(frame)}"
            )
        self._buf.append(frame)

        if len(self._buf) < self.window_size:
            return False

        if not self._ever_full:
            self._ever_full = True
            self._since_last_trigger = 0
            return True

        self._since_last_trigger += 1
        if self._since_last_trigger >= self.stride:
            self._since_last_trigger = 0
            return True
        return False

    def as_array(self) -> np.ndarray:
        return np.asarray(self._buf, dtype=np.float32)

    def warmup_state(self) -> tuple[int, int]:
        return len(self._buf), self.window_size


if __name__ == "__main__":
    buf = FrameBuffer(window_size=20, stride=10)
    triggers = []
    for i in range(45):
        fired = buf.push([float(i)] * RAW_FEATURE_COUNT)
        if fired:
            triggers.append(i + 1)  # 1-indexed frame count when fired
    print(f"trigger frames: {triggers}")
    print(f"final shape:    {buf.as_array().shape}")
    print(f"warmup state:   {buf.warmup_state()}")
    print(f"last frame[0]:  {buf.as_array()[-1, 0]} (should be 44.0)")
