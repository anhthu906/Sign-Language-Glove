# Sign-Language Glove — inference modules

Project có **2 pipeline inference song song**, chia sẻ cùng các stage 4 + 5
(preprocess + predictor), dùng cùng artifacts (`model_best.h5`, `scaler.npz`,
`label_classes.npy`) sinh ra từ [notebooks/Pipeline.ipynb](notebooks/Pipeline.ipynb).

| Pipeline | Module | Khi nào dùng |
|---|---|---|
| **Realtime (glove)** | [src/inference/predict_pipeline.py](src/inference/predict_pipeline.py) | Khi có glove vật lý cắm USB. Đọc serial / stdin, FLX: parser, rolling buffer, predict liên tục |
| **Offline (CSV)** | [src/inference/predict_csv.py](src/inference/predict_csv.py) | Khi chỉ có file CSV (test/replay). Đọc thẳng CSV, batched model.predict, in label cho cả file trong 1 call — nhanh hơn nhiều |

## Setup môi trường

Cần Python **3.10–3.12** (TensorFlow chưa có wheel cho 3.13+).

```bash
cd /home/anchin/Projects/Sign-Language-Glove---AI

python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Thoát venv: `deactivate`.

> Hai pipeline thực sự chỉ cần `tensorflow`, `numpy` (+ `pyserial` cho glove).
> [requirements.txt](requirements.txt) cài thêm pandas/jupyter… phục vụ
> training notebook.

---

## Pipeline A — Realtime (glove)

```
serial / stdin  →  parser  →  buffer  →  preprocess  →  predictor  →  stdout
                  (FLX:..)   (rolling   (cyclic-encode   (Keras +
                              20×8)      + scaler)        LABEL_REMAP)
```

| Stage | File |
|---|---|
| 1. Parse 1 dòng `FLX:` → 8 float | [stage_1_parser.py](src/inference/stage_1_parser.py) |
| 2. Iterator yield dòng từ pyserial/stdin | [stage_2_stream.py](src/inference/stage_2_stream.py) |
| 3. Rolling window 20 frame + stride trigger | [stage_3_buffer.py](src/inference/stage_3_buffer.py) |
| 4. Cyclic-encode `imu_x`, áp StandardScaler | [stage_4_preprocess.py](src/inference/stage_4_preprocess.py) |
| 5. Load model + predict + remap | [stage_5_predictor.py](src/inference/stage_5_predictor.py) |
| Compose | [predict_pipeline.py](src/inference/predict_pipeline.py) |

### Chạy

```bash
# Glove cắm USB (auto-detect /dev/ttyACM0 → /dev/ttyUSB0)
python3 -m src.inference.predict_pipeline

# Chỉ định port
python3 -m src.inference.predict_pipeline --port /dev/ttyACM0
```

### Tham số CLI

```
--port PORT             Serial port (mặc định: auto-detect)
--stdin                 Đọc FLX: từ stdin thay vì serial
--baud BAUD             Mặc định 115200
--stride STRIDE         Số frame giữa các lần predict (mặc định 10)
--model / --scaler / --labels   Override artifact paths
--conf-threshold FLOAT  Lọc prediction theo confidence
```

### Output

```
[19:24:13] tôi          conf=0.94   top3: tôi/0.94  bao nhiêu/0.03  C/0.02
[19:24:14] tôi          conf=0.91   top3: tôi/0.91  C/0.05  pink/0.02
```

---

## Pipeline B — Offline CSV (nhanh, không cần glove)

Đọc 1 file CSV (schema `flex1..flex5, imu_x, imu_y, imu_z [, SIGN]`), trượt
window cố định, batched model.predict, in label.

Khác Pipeline A:
- **Không qua parser/serial/stdin** — đọc CSV trực tiếp với `csv.DictReader`
- **1 call `model.predict(batch)`** cho cả file (thay vì ~1000 call riêng lẻ)
- **Reuse stage 4 + 5** (chia sẻ logic preprocess + predictor + LABEL_REMAP)

### Chạy

```bash
# Per-window output
python3 -m src.inference.predict_csv "data/raw/Test/xinchao2_1778993189.csv"

# Summary (distribution của label cho cả file)
python3 -m src.inference.predict_csv "data/raw/Test/xinchao2_1778993189.csv" --summary

# Chỉ in window có conf cao
python3 -m src.inference.predict_csv path/to/file.csv --conf-threshold 0.7
```

### Output

**Per-window mode:**
```
window[   0] frame=    0  xin chào      conf=0.92
window[   1] frame=   10  xin chào      conf=0.95
window[   2] frame=   20  pink          conf=0.71
...
```

**Summary mode:**
```
# data/raw/Test/xinchao2_1778993189.csv
# 1004 windows above conf-threshold 0.0
# sign_in_file = 'xinchao2'
     850  ( 84.7%)  xin chào
     120  ( 12.0%)  pink
      34  (  3.4%)  test
```

### Tham số CLI

```
csv_path                 Đường dẫn file CSV (positional)
--stride STRIDE          Bước trượt giữa các window (mặc định 10)
--batch-size N           Batch size cho model.predict (mặc định 128)
--model / --scaler / --labels   Override artifact paths
--conf-threshold FLOAT   Lọc theo confidence
--summary                In bảng phân phối label thay vì từng window
```

---

## Debug từng stage

Mỗi file stage có block `if __name__ == "__main__":` chạy thử riêng:

```bash
python3 src/inference/stage_1_parser.py        # parse vài dòng mẫu
python3 src/inference/stage_3_buffer.py        # trigger frames 20, 30, 40
python3 src/inference/stage_4_preprocess.py    # scaler + cyclic-encode
python3 src/inference/stage_5_predictor.py     # load model + predict zeros
```

## Label remap

`label_classes.npy` lưu **key thô** từ tên file (vd `baonhieu2`, `tôi3`).
[stage_5_predictor.py](src/inference/stage_5_predictor.py) mirror `LABEL_REMAP`
từ [notebooks/Pipeline.ipynb](notebooks/Pipeline.ipynb) (cell 3) để API
trả về nhãn tiếng Việt sạch:

| Raw key (npy) | Display |
|---|---|
| `baonhieu2` | `bao nhiêu` |
| `C` | `C` |
| `khong_2` / `khong?2` | `không` |
| `O2` | `O` |
| `pink4` | `pink` |
| `tôi3` | `tôi` |
| `xinchao0` | `xin chào` |
| `test2` | `test` |

Key không có mapping sẽ pass through. Khi thêm class mới trong Pipeline.ipynb
thì update LABEL_REMAP ở cả 2 nơi.

## Artifacts cần có

| File | Sinh từ |
|---|---|
| `results/models/model_best.h5` | [notebooks/Pipeline.ipynb](notebooks/Pipeline.ipynb) |
| `data/processed/scaler.npz` | [notebooks/Pipeline.ipynb](notebooks/Pipeline.ipynb) |
| `data/processed/label_classes.npy` | [notebooks/Pipeline.ipynb](notebooks/Pipeline.ipynb) |

## TODO — khi cắm lại glove vật lý

Hiện tại [stage_4_preprocess.py](src/inference/stage_4_preprocess.py) giả định
input flex đã ở đơn vị normalized (~0.001–0.07), khớp với CSV training.

Khi kết nối glove ([config/main.cpp](config/main.cpp)) và muốn dùng Pipeline A
trở lại, kiểm tra:
- Nếu firmware gửi **raw int** (hiện tại `abs((int)fThumb - offsetThumb)`) → thêm `window[:, 0:5] /= 4095.0` ở đầu `preprocess_window`
- Nếu firmware đã normalize sẵn → giữ nguyên

## Khác biệt so với realtime_predict.py cũ

| | `realtime_predict.py` | `predict_pipeline.py` + `predict_csv.py` |
|---|---|---|
| Features | flex×5 + imu_xyz + **face (MediaPipe)** = 9 | flex×5 + imu_y/z + sin/cos(imu_x) = 9 |
| Camera | Bắt buộc | Không dùng |
| StandardScaler | ❌ | ✅ |
| Cyclic encode `imu_x` | ❌ | ✅ |
| LABEL_REMAP | ❌ | ✅ |
| Cấu trúc | 1 file monolith | 5 stage + 2 compose, dễ debug + reuse |
