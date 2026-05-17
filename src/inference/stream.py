"""Stage 2: stream lines from a glove serial port or stdin.

Each yielded item is a single decoded text line (no trailing newline).
"""
from __future__ import annotations

import os
import sys
from typing import Iterator

DEFAULT_PORT_CANDIDATES = ("/dev/ttyACM0", "/dev/ttyUSB0")
DEFAULT_BAUD = 115200


def from_stdin() -> Iterator[str]:
    for line in sys.stdin:
        line = line.rstrip("\r\n")
        if line:
            yield line


def from_serial(port: str, baud: int = DEFAULT_BAUD) -> Iterator[str]:
    try:
        import serial
    except ImportError as e:
        raise RuntimeError(
            "pyserial not installed. Run `pip install pyserial` or use --stdin to replay from a file."
        ) from e

    try:
        ser = serial.Serial(port, baud, timeout=1)
    except serial.SerialException as e:
        raise RuntimeError(
            f"Cannot open serial port {port}: {e}. Try another port (e.g. /dev/ttyUSB0) "
            f"or use --stdin."
        ) from e

    try:
        while True:
            raw = ser.readline()
            if not raw:
                continue
            try:
                line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
            except Exception:
                continue
            if line:
                yield line
    finally:
        ser.close()


def auto_stream(args) -> Iterator[str]:
    """Dispatch based on argparse Namespace: --stdin, --port, or auto-detect."""
    if getattr(args, "stdin", False):
        return from_stdin()

    port = getattr(args, "port", None)
    if port:
        return from_serial(port, getattr(args, "baud", DEFAULT_BAUD))

    for cand in DEFAULT_PORT_CANDIDATES:
        if os.path.exists(cand):
            print(f"[stream] auto-detected serial port: {cand}", file=sys.stderr)
            return from_serial(cand, getattr(args, "baud", DEFAULT_BAUD))

    raise RuntimeError(
        f"No serial port specified and none of {DEFAULT_PORT_CANDIDATES} exist. "
        f"Use --port /dev/ttyXXX or --stdin to replay from a file."
    )


if __name__ == "__main__":
    print("Type lines, EOF (Ctrl-D) to stop:", file=sys.stderr)
    for ln in from_stdin():
        print(f"got: {ln!r}")
