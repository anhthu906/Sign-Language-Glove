# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Sign-language recognition from a wearable glove. An ESP32-based glove streams sensor readings (5 flex sensors + IMU position/orientation, plus a MediaPipe-derived mouth-ratio "face" feature) which are accumulated into fixed-length windows and classified by a Keras LSTM. Labels are Vietnamese signs (e.g. "Tôi", "Xin Chào", "Hẹn Hò", "Không Thích").

Install deps with `pip install -r requirements.txt`. The repo was forked from a prior MacOS-based codebase ("Stella"), and the legacy material lives in [stella/](stella/).

## Repo layout

```
.
├── data/
│   ├── raw/         # All raw sensor CSVs from glove collection (one sample = 1 file)
│   │   └── test/    # Held-out test samples
│   ├── processed/   # (empty) for normalized / windowed features
│   └── interim/     # (empty) for intermediate artifacts
├── notebooks/       # Active EDA / training notebooks
├── src/
│   ├── data_pipeline/   # Ingestion, labeling, preprocessing
│   ├── models/          # (empty) model architecture modules
│   ├── training/        # (empty) training scripts / strategies
│   ├── inference/       # Realtime / batch prediction
│   ├── testing/         # (empty) unit + integration tests
│   └── utils/           # (empty) shared helpers
├── results/         # Trained model artifacts, label encoders, plots
│   ├── models/      # .h5 checkpoints (model_best.h5, model_fold_*.h5, Training.h5)
│   └── plots/       # Sensor-data visualizations (PNG)
├── config/          # (empty) YAML/JSON config files
├── scripts/         # (empty) shell scripts for download/train/serve
├── stella/          # Legacy / non-portable MacOS code (do not run as-is)
└── requirements.txt
```

## Pipeline A — the active pipeline (input shape `(20, 9)`)

This is the only pipeline that lives outside [stella/](stella/). All current files in [src/](src/) and [notebooks/](notebooks/) belong to it.

- **Data**: one CSV per sample under [data/raw/](data/raw/). Filename pattern `<SIGN>_<timestamp>.csv`. Columns: `flex1,flex2,flex3,flex4,flex5,x,y,z,face,SIGN`. Each file is exactly 20 rows (frames) belonging to a single sample, with the same SIGN on every row.
- **Trainer**: [notebooks/Training.ipynb](notebooks/Training.ipynb). Reshapes concatenated frames as `(-1, 20, 9)`, label = first frame's SIGN. 5-fold `KFold` CV. Outputs go to [results/models/](results/models/) (`model_fold_{1..5}.h5`, `model_best.h5`, `Training.h5`) and label encoder to [results/label_classes.npy](results/label_classes.npy).
- **Realtime inference**: [src/inference/realtime_predict.py](src/inference/realtime_predict.py). Reads the glove over a serial port (default `/dev/tty.usbmodem101` @ 115200 — **change for Linux**, typically `/dev/ttyUSB0` or `/dev/ttyACM0`) and uses `mediapipe.solutions.holistic` to compute `face = mouth_height / mouth_width`. Collects 25 frames, trims/pads to `model.input_shape[1]` (20). Loads from [results/models/model_best.h5](results/models/model_best.h5) + [results/label_classes.npy](results/label_classes.npy). **Run from project root** — paths are resolved relative to CWD.
- **Labeling helper**: [src/data_pipeline/label.py](src/data_pipeline/label.py) appends a `SIGN` column to every CSV in `data/raw/`, derived from the filename prefix before `_`. Useful after raw collection. Edit `folder = "data/raw"` if collecting elsewhere.

### Contract for new data (pipeline A)

The trainer's reshape relies on a strict contract: a sample CSV must be exactly 20 rows, columns in the exact order `flex1,flex2,flex3,flex4,flex5,x,y,z,face,SIGN`, and the SIGN value must be identical on every row of the file (the trainer reads only the first frame's label per 20-row block). Breaking any of these silently corrupts training.

## stella/ — legacy / do not run as-is

Everything in [stella/](stella/) was forked from a prior MacOS-based codebase and contains hardcoded `/Users/stella/...` paths or targets an older, incompatible pipeline B (input shape `(50, 440)`, two-hand glove format with right-hand placeholders, hardcoded `SIGN='A'`).

- `Test.ipynb`, `learn.ipynb`, `run_model.ipynb` — pipeline B notebooks with hardcoded MacOS paths
- `Test.h5`, `Test1.h5`, `Test.onnx`, `testmodel.csv` — pipeline B model + test artifacts
- `save_data.py` (Flask), `savedata.py` (FastAPI) — ingest servers writing pipeline-B 440-feature rows; reference for building a new pipeline-A ingest server, not directly usable
- `save_model.py`, `run_model1.py` — h5→onnx export / onnx sanity-check, with hardcoded MacOS paths to pipeline-B artifacts

When porting anything out of `stella/`, expect to rewrite paths and (for the ingest servers) the CSV schema to match the pipeline-A 20-row × 9-column contract.

## Known cleanup items

- [notebooks/Training.ipynb](notebooks/Training.ipynb) still has `data_dir = '/Users/stella/Downloads/Data Training/data'` hardcoded inside — change to `data/raw` (relative to project root) before running.
- [data/raw/](data/raw/) contains Unicode-normalization doublets (NFC + NFD variants of the same Vietnamese filename, identical content) carried over from the MacOS filesystem. They show up as two files with seemingly-identical names on Linux. Safe to dedupe — they have the same bytes.
- No requirements pinning: [requirements.txt](requirements.txt) lists packages without versions. Pin once a working environment is captured.
- [src/models/](src/models/), [src/training/](src/training/) are empty placeholders — the LSTM architecture and training loop currently live inline in [notebooks/Training.ipynb](notebooks/Training.ipynb). Promoting them into reusable modules is the next refactor.

## Common commands

```bash
# Install dependencies
pip install -r requirements.txt

# Backfill SIGN column into every CSV in data/raw (from filename prefix)
python3 src/data_pipeline/label.py

# Realtime prediction loop (requires camera + serial glove + a trained model)
python3 src/inference/realtime_predict.py

# Open the active training notebook
jupyter notebook notebooks/Training.ipynb
```
