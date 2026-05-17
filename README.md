# predict_pipeline — realtime inference cho glove

Module nhận stream `FLX:<thumb>,<index>,<middle>,<ring>,<pinky>,<imu_x>,<imu_y>,<imu_z>`
từ glove (qua serial hoặc stdin), preprocess đúng như [notebooks/Pipeline.ipynb](notebooks/Pipeline.ipynb),
rồi predict label.

## Kiến trúc

```
serial / stdin  →  parser  →  buffer  →  preprocess  →  predictor  →  stdout
                  (FLX:..)   (rolling   (/4095, cyclic   (Keras +
                              20×8)      encode, scale)   LABEL_REMAP)
```

| File | Trách nhiệm |
|---|---|
| [src/inference/stage_1_parser.py](src/inference/stage_1_parser.py) | Parse 1 dòng `FLX:` → 8 float |
| [src/inference/stage_2_stream.py](src/inference/stage_2_stream.py) | Iterator yield dòng từ pyserial hoặc stdin |
| [src/inference/stage_3_buffer.py](src/inference/stage_3_buffer.py) | Rolling window 20 frame + stride trigger |
| [src/inference/stage_4_preprocess.py](src/inference/stage_4_preprocess.py) | flex /= 4095, cyclic-encode `imu_x`, áp StandardScaler |
| [src/inference/stage_5_predictor.py](src/inference/stage_5_predictor.py) | Load `model_best.h5` + `label_classes.npy`, predict, apply LABEL_REMAP |
| [src/inference/predict_pipeline.py](src/inference/predict_pipeline.py) | argparse + nối 5 stage trên |

## Setup môi trường

Module cần Python **3.10–3.12** (TensorFlow chưa có wheel cho 3.13+).

```bash
cd /home/anchin/Projects/Sign-Language-Glove---AI

# Tạo venv
python3.12 -m venv .venv
source .venv/bin/activate

# Cài deps
pip install --upgrade pip
pip install -r requirements.txt
```

Để thoát venv khi xong: `deactivate`.

> Module realtime này chỉ thực sự cần `tensorflow`, `numpy`, `pyserial`.
> [requirements.txt](requirements.txt) cài thêm pandas/mediapipe/jupyter…
> phục vụ training notebook và pipeline cũ — vẫn cài cả file để env đồng nhất
> với Pipeline.ipynb.

## Chạy

**Mọi lệnh đều chạy từ project root** (`/home/anchin/Projects/Sign-Language-Glove---AI`).

### 1. Replay test (không cần phần cứng)

Lấy 1 CSV training/test rồi pipe vào module — kiểm tra preprocessing + model có đúng:

```bash
python3 scripts/replay_csv.py "data/raw/Good Data/baonhieu2_1778907665.csv" \
    | python3 -m src.inference.predict_pipeline --stdin
```

Kỳ vọng: in ra label `bao nhiêu` (đã remap từ `baonhieu2`) với confidence cao (>0.8).

### 2. Live với glove

Cắm glove vào USB, chạy:

```bash
python3 -m src.inference.predict_pipeline --port /dev/ttyACM0
```

Nếu không truyền `--port`, module sẽ tự dò `/dev/ttyACM0` rồi `/dev/ttyUSB0`.

### 3. Debug từng stage

Mỗi file có block `if __name__ == "__main__":` chạy thử riêng:

```bash
python3 src/inference/stage_1_parser.py        # in kết quả parse vài dòng mẫu
python3 src/inference/stage_3_buffer.py        # verify trigger frames 20, 30, 40
python3 src/inference/stage_4_preprocess.py    # kiểm scaler load + /4095 normalize
python3 src/inference/stage_5_predictor.py     # load model + predict tensor zeros
```

## Tham số CLI

```
--port PORT              Serial port (mặc định: auto-detect /dev/ttyACM0 → /dev/ttyUSB0)
--stdin                  Đọc từ stdin thay vì serial (cho replay/test)
--baud BAUD              Mặc định 115200, khớp firmware
--stride STRIDE          Số frame giữa các lần predict (mặc định 10, khớp training)
--model PATH             Mặc định: results/models/model_best.h5
--scaler PATH            Mặc định: data/processed/scaler.npz
--labels PATH            Mặc định: data/processed/label_classes.npy
--conf-threshold FLOAT   Chỉ in prediction có conf ≥ ngưỡng (mặc định 0.0)
```

## Output format

```
[19:24:13] tôi          conf=0.94   top3: tôi/0.94  bao nhiêu/0.03  C/0.02
[19:24:14] tôi          conf=0.91   top3: tôi/0.91  C/0.05  pink/0.02
```

Stderr in các message khởi tạo và warmup (`[init] ...`, `[warmup] 15/20`).

## Label remap

`label_classes.npy` lưu **key thô** từ tên file (vd `baonhieu2`, `tôi3`).
[notebooks/Pipeline.ipynb](notebooks/Pipeline.ipynb) cell 3 + cell 8 dùng
`LABEL_REMAP` để map key này sang nhãn tiếng Việt sạch. Module
[stage_5_predictor.py](src/inference/stage_5_predictor.py) mirror lại
`LABEL_REMAP` ở cấp API:

| Raw key (npy) | Display label |
|---|---|
| `baonhieu2` | `bao nhiêu` |
| `C` | `C` |
| `khong_2` / `khong?2` | `không` |
| `O2` | `O` |
| `pink4` | `pink` |
| `tôi3` | `tôi` |
| `xinchao0` | `xin chào` |
| `test2` | `test` |

Key nào không có mapping sẽ trả về nguyên dạng. Nếu thêm class mới trong
Pipeline.ipynb thì update `LABEL_REMAP` trong cả notebook và
[stage_5_predictor.py](src/inference/stage_5_predictor.py).

## Artifacts cần có trước khi chạy

| File | Sinh từ |
|---|---|
| `results/models/model_best.h5` | [notebooks/Pipeline.ipynb](notebooks/Pipeline.ipynb) |
| `data/processed/scaler.npz` | [notebooks/Pipeline.ipynb](notebooks/Pipeline.ipynb) |
| `data/processed/label_classes.npy` | [notebooks/Pipeline.ipynb](notebooks/Pipeline.ipynb) |

Nếu thiếu → chạy lại Pipeline.ipynb trước.

## Lưu ý preprocessing (quan trọng)

Training CSV trong `data/raw/Good Data/` lưu flex **đã chia cho 4095** (max 12-bit ADC),
còn firmware ([config/main.cpp](config/main.cpp)) gửi **số nguyên thô**.
Module tự chia `/4095` trong [stage_4_preprocess.py](src/inference/stage_4_preprocess.py) → khớp đúng phân phối training.

Nếu sau này thay firmware để gửi giá trị đã normalize sẵn → sửa
`FLEX_DIVISOR = 1.0` trong [stage_4_preprocess.py](src/inference/stage_4_preprocess.py).

## Khác biệt so với realtime_predict.py cũ

| | `realtime_predict.py` | `predict_pipeline.py` |
|---|---|---|
| Features | flex×5 + imu_xyz + **face (MediaPipe)** = 9 | flex×5 + imu_y/z + sin/cos(imu_x) = 9 |
| Camera | Bắt buộc | Không dùng |
| StandardScaler | ❌ không load | ✅ load từ `scaler.npz` |
| Cyclic encode `imu_x` | ❌ | ✅ |
| Normalize flex | ❌ | ✅ `/4095` |
| LABEL_REMAP | ❌ | ✅ raw → tiếng Việt |
| Input format | Whitespace-split | Prefix `FLX:` + comma-split |
| Cách chạy | 1 file monolith | 5 stage tách rời, dễ debug |
