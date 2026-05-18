"""
Glove Sign Language — Real-Time Live Plotter + CSV Logger
NOW WITH:
✓ Min-Max scaling
✓ Saved normalized values
✓ Live plot uses normalized values
"""

import re
import time
import threading
import os
import atexit
import csv
from collections import deque

import numpy as np
import serial
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
from scipy.signal import find_peaks, savgol_filter

# ========================= CONFIG =========================
PORT     = "/dev/cu.usbmodem14401"
BAUD     = 115200
Fs       = 30
PLOT_SEC = 6

PEAK_PROMINENCE_RATIO = 0.25
PEAK_MIN_DISTANCE_SEC = 0.8
SAVGOL_WINDOW         = 11
SAVGOL_POLY           = 2

# ===== MIN-MAX SCALING =====
FLEX_MIN = 0
FLEX_MAX = 4095

# ==========================================================

print("=" * 40)
print("  GLOVE DATA LOGGER")
print("=" * 40)

SIGN_LABEL = input("  Enter sign name: ").strip()

if not SIGN_LABEL:
    SIGN_LABEL = "unknown"

safe_name = re.sub(r'[\\/:*?"<>|]', '_', SIGN_LABEL).replace(' ', '_')

timestamp = int(time.time())

CSV_FILE = f"{safe_name}_{timestamp}.csv"
PNG_FILE = f"{safe_name}_{timestamp}.png"

print(f"\n  Sign    : {SIGN_LABEL}")
print(f"  CSV     : {CSV_FILE}")
print(f"  Graph   : {PNG_FILE}")
print(f"  Port    : {PORT}")
print("  Close window or Ctrl+C to stop and save.")
print("=" * 40 + "\n")

# ========================= BUFFERS ========================
BUF_LEN = Fs * PLOT_SEC * 2

flex_bufs = [deque(maxlen=BUF_LEN) for _ in range(5)]

full_data = [[] for _ in range(5)]
full_imu  = [[], [], []]

IMU_LABELS = ["Yaw (X)", "Pitch (Y)", "Roll (Z)"]
IMU_COLORS = ["cyan", "magenta", "yellow"]

FINGER_LABELS = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
FINGER_COLORS = ["red", "orange", "limegreen", "dodgerblue", "mediumpurple"]

# ========================= SERIAL PATTERN =========================
PATTERN = re.compile(
    r"FLX:"
    r"([\d]+),([\d]+),([\d]+),([\d]+),([\d]+)"
    r",([-\d.]+),([-\d.]+),([-\d.]+)"
)

# ========================= CSV ============================
print(f"Saving to: {os.path.join(os.getcwd(), CSV_FILE)}")

csv_file = open(CSV_FILE, "w", newline="", encoding="utf-8")
csv_writer = csv.writer(csv_file)

csv_writer.writerow([
    "flex1",
    "flex2",
    "flex3",
    "flex4",
    "flex5",
    "imu_x",
    "imu_y",
    "imu_z",
    "SIGN"
])

# ========================= GLOBALS ========================
start_time   = time.time()
valid_rows   = 0
skipped_rows = 0

data_lock = threading.Lock()

# ========================= GRAPH SAVE =====================
def save_summary_graph():

    with data_lock:
        snapshots    = [list(d) for d in full_data]
        imu_snapshot = [list(d) for d in full_imu]

    total_samples = len(snapshots[0])

    if total_samples < 10:
        print("Not enough data to save a graph.")
        return

    t = np.arange(total_samples) / Fs

    min_dist_samples = max(1, int(PEAK_MIN_DISTANCE_SEC * Fs))

    fig = plt.figure(figsize=(14, 14))
    fig.patch.set_facecolor("#0e0e0e")

    gs = gridspec.GridSpec(6, 1, hspace=0.6)

    fig.suptitle(
        f"Sign: '{SIGN_LABEL}' — {total_samples/Fs:.1f}s | {total_samples} samples",
        fontsize=14,
        fontweight="bold",
        color="white",
        y=0.98
    )

    # ================= FLEX PLOTS =================
    for i, (label, color, raw) in enumerate(
        zip(FINGER_LABELS, FINGER_COLORS, snapshots)
    ):

        ax = fig.add_subplot(gs[i])

        raw = np.array(raw, dtype=float)

        win = min(
            SAVGOL_WINDOW,
            len(raw) if len(raw) % 2 == 1 else len(raw) - 1
        )

        win = max(win, 3)

        if win % 2 == 0:
            win += 1

        try:
            smooth = savgol_filter(
                raw,
                window_length=win,
                polyorder=SAVGOL_POLY
            )
        except Exception:
            smooth = raw.copy()

        sig_range = raw.max() - raw.min()
        sig_min   = raw.min()
        sig_max   = raw.max()

        prominence_thresh = max(
            0.02,
            PEAK_PROMINENCE_RATIO * sig_range
        )

        peaks, _ = find_peaks(
            smooth,
            prominence=prominence_thresh,
            distance=min_dist_samples
        )

        peak_amplitudes = (
            smooth[peaks] - sig_min
            if len(peaks)
            else np.array([])
        )

        ax.plot(
            t,
            raw,
            color=color,
            lw=0.9,
            alpha=0.6,
            label="Raw"
        )

        ax.plot(
            t,
            smooth,
            color="white",
            lw=1.2,
            alpha=0.85,
            label="Smoothed"
        )

        if len(peaks):
            ax.scatter(
                t[peaks],
                smooth[peaks],
                color="yellow",
                zorder=5,
                s=40,
                label=f"Peaks ({len(peaks)})"
            )

        mean_amp = (
            peak_amplitudes.mean()
            if len(peak_amplitudes)
            else 0.0
        )

        std_amp = (
            peak_amplitudes.std()
            if len(peak_amplitudes)
            else 0.0
        )

        ax.set_title(
            f"Range: {sig_range:.3f} | Peaks: {len(peaks)} | "
            f"Mean amp: {mean_amp:.3f} ± {std_amp:.3f}",
            fontsize=8.5,
            color="#cccccc",
            pad=3
        )

        ax.set_facecolor("#1a1a1a")

        ax.set_ylabel(
            label,
            color=color,
            fontsize=9,
            fontweight="bold"
        )

        ax.tick_params(colors="#aaaaaa", labelsize=7)

        for spine in ax.spines.values():
            spine.set_edgecolor("#333333")

        ax.set_xlim(t[0], t[-1])

        pad = sig_range * 0.15 + 0.02

        ax.set_ylim(sig_min - pad, sig_max + pad)

        ax.grid(alpha=0.15, color="#444444")

        ax.set_xticklabels([])

        ax.legend(
            loc="upper right",
            fontsize=7.5,
            framealpha=0.3,
            labelcolor="white"
        )

    # ================= IMU PLOT =================
    ax_imu = fig.add_subplot(gs[5])

    ax_imu.set_facecolor("#1a1a1a")

    imu_all_present = all(
        len(ch) == total_samples
        for ch in imu_snapshot
    )

    if imu_all_present and any(np.ptp(ch) > 0.01 for ch in imu_snapshot):

        imu_t = np.arange(len(imu_snapshot[0])) / Fs

        for ch_data, ch_label, ch_color in zip(
            imu_snapshot,
            IMU_LABELS,
            IMU_COLORS
        ):

            arr = np.array(ch_data, dtype=float)

            ax_imu.plot(
                imu_t,
                arr,
                lw=1.1,
                color=ch_color,
                alpha=0.85,
                label=ch_label
            )

    else:

        ax_imu.text(
            0.5,
            0.5,
            "IMU not available",
            transform=ax_imu.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            color="#666666",
            style="italic"
        )

    ax_imu.set_ylabel("IMU")
    ax_imu.set_xlabel("Time (s)")

    ax_imu.grid(alpha=0.15)

    ax_imu.legend()

    plt.savefig(
        PNG_FILE,
        dpi=150,
        bbox_inches="tight",
        facecolor=fig.get_facecolor()
    )

    plt.close(fig)

    print(f"Graph saved : {PNG_FILE}")

# ========================= CLEANUP ========================
def close_file():

    print(f"\nCSV saved : {CSV_FILE}")

    csv_file.close()

    save_summary_graph()

atexit.register(close_file)

# ========================= SERIAL READER ==================
def reader():

    global valid_rows, skipped_rows

    try:

        ser = serial.Serial(PORT, BAUD, timeout=1)

        time.sleep(2)

        ser.reset_input_buffer()

        print(f"Serial connected on {PORT}\n")

        while True:

            try:

                raw = ser.readline()

                if not raw:
                    continue

                line = raw.decode("utf-8").strip()

                if not line:
                    continue

                m = PATTERN.search(line)

                if not m:
                    skipped_rows += 1
                    continue

                groups = m.groups()

                # ===== RAW FLEX =====
                flex_raw = [int(v) for v in groups[:5]]

                # ===== IMU =====
                imu = [float(v) for v in groups[5:]]

                # ===== MIN-MAX SCALE =====
                flex = [
                    (v - FLEX_MIN) / (FLEX_MAX - FLEX_MIN)
                    for v in flex_raw
                ]

                # ===== STORE =====
                with data_lock:

                    for i, v in enumerate(flex):

                        flex_bufs[i].append(v)

                        full_data[i].append(v)

                    for i, v in enumerate(imu):

                        full_imu[i].append(v)

                # ===== SAVE CSV =====
                csv_writer.writerow(
                    flex + imu + [SIGN_LABEL]
                )

                csv_file.flush()

                valid_rows += 1

            except Exception as e:

                print(f"[READ ERROR] {e}")

    except Exception as e:

        print(f"[SERIAL ERROR] {e}")

        print("Is the port correct? Is Serial Monitor closed?")

threading.Thread(
    target=reader,
    daemon=True
).start()

# ========================= LIVE PLOT ======================
plt.style.use("dark_background")

# ===== 2 SUBPLOTS =====
fig, (ax_flex, ax_imu) = plt.subplots(
    2,
    1,
    figsize=(13, 8),
    sharex=True
)

fig.suptitle(
    f"Glove Live — {SIGN_LABEL}",
    fontsize=14,
    fontweight="bold"
)

# ================= FLEX PLOT =================
ax_flex.set_title("Normalized Flex Sensors (live)")

ax_flex.set_ylabel("Normalized Value")

ax_flex.set_xlim(-PLOT_SEC, 0)

ax_flex.set_ylim(0, 1)

ax_flex.grid(alpha=0.3)

flex_lines = [
    ax_flex.plot([], [], lw=1.5, label=lbl, color=clr)[0]
    for lbl, clr in zip(FINGER_LABELS, FINGER_COLORS)
]

ax_flex.legend(loc="upper left", ncol=5, fontsize=9)

# ================= IMU PLOT =================
ax_imu.set_title("BNO055 IMU (live)")

ax_imu.set_ylabel("Angle (deg)")

ax_imu.set_xlabel("Time (s)")

ax_imu.set_xlim(-PLOT_SEC, 0)

ax_imu.grid(alpha=0.3)

imu_lines = [
    ax_imu.plot([], [], lw=1.5, label=lbl, color=clr)[0]
    for lbl, clr in zip(IMU_LABELS, IMU_COLORS)
]

ax_imu.legend(loc="upper left", ncol=3, fontsize=9)

# ================= STATUS TEXT =================
stats_txt = ax_flex.text(
    0.99,
    0.95,
    "",
    transform=ax_flex.transAxes,
    ha="right",
    va="top",
    fontsize=10,
    color="white"
)

t_axis = np.linspace(
    -PLOT_SEC,
    0,
    int(PLOT_SEC * Fs)
)

# ========================= SMOOTH LIVE ====================
def smooth_live(x, k=5):

    if len(x) < k:
        return x

    return np.convolve(
        x,
        np.ones(k) / k,
        mode="same"
    )

# ========================= UPDATE =========================
def update(frame):

    n = int(PLOT_SEC * Fs)

    if len(flex_bufs[0]) < 10:
        return flex_lines + imu_lines + [stats_txt]

    # ===== FLEX =====
    for line, buf in zip(flex_lines, flex_bufs):

        arr = smooth_live(np.array(buf)[-n:])

        t = t_axis[-len(arr):]

        line.set_data(t, arr)

    # ===== IMU =====
    imu_min = np.inf
    imu_max = -np.inf

    for line, imu_buf in zip(imu_lines, full_imu):

        arr = smooth_live(np.array(imu_buf[-n:]))

        t = t_axis[-len(arr):]

        line.set_data(t, arr)

        if len(arr):
            imu_min = min(imu_min, arr.min())
            imu_max = max(imu_max, arr.max())

    # Auto-scale IMU
    if imu_min != np.inf:

        pad = (imu_max - imu_min) * 0.15 + 5

        ax_imu.set_ylim(
            imu_min - pad,
            imu_max + pad
        )

    elapsed = time.time() - start_time

    stats_txt.set_text(
        f"Sign: {SIGN_LABEL}"
        f" | Saved: {valid_rows}"
        f" | Skipped: {skipped_rows}"
        f" | {elapsed:.0f}s"
    )

    return flex_lines + imu_lines + [stats_txt]

# ========================= RUN ============================
ani = FuncAnimation(
    fig,
    update,
    interval=40,
    blit=True,
    cache_frame_data=False
)

plt.tight_layout()

plt.show()