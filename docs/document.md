# Tài liệu — `notebooks/Pipeline.ipynb`

File này ghi lại các bước đã làm trong `notebooks/Pipeline.ipynb` — từ đọc CSV thô tới huấn luyện mô hình BiLSTM với Stratified 5-Fold CV.

---

## 1. Tổng quan

Pipeline đi theo trình tự:

```
data/raw/Good Data/*.csv
        │
        ▼  (window 20 frame, stride 10)
   X shape (n_windows, 20, 8)
        │
        ▼  (sin/cos cho imu_x)
   X shape (n_windows, 20, 9)
        │
        ▼  (encode label, shuffle)
   X, y (đã shuffle)
        │
        ▼  (stratified split 70 / 15 / 15)
   train  val  test
        │
        ▼  (StandardScaler, fit on train rows)
   train_s val_s test_s
        │
        ▼  lưu vào data/processed/
        │
        ▼  Stratified 5-Fold CV — BiLSTM
   results/models/model_fold_{1..5}.h5
   results/models/model_best.h5
```

---

## 2. Dữ liệu nguồn

- 7 file CSV trong `data/raw/Good Data/`, mỗi file là 1 phiên ghi liên tục (1140–4080 dòng).
- Mỗi dòng có 9 cột: `flex1, flex2, flex3, flex4, flex5, imu_x, imu_y, imu_z, SIGN`.
- **Không có cột `face`** (khác với pipeline cũ trong `src/inference/realtime_predict.py`).
- Mỗi file có duy nhất 1 nhãn `SIGN`, lặp lại trên mọi dòng.
- Nhãn hiện có: `baonhieu2`, `test2` (file `C.csv`), `khong?2`, `O2`, `pink4`, `tôi3`, `xinchao0`.

---

## 3. Chi tiết từng bước

### Bước 1 — Setup & Config

- `PROJECT_ROOT` được auto-resolve: chạy từ `notebooks/` hay từ root đều OK.
- Các hằng số quan trọng:

| Tên | Giá trị | Ý nghĩa |
|---|---|---|
| `WINDOW_SIZE` | 20 | Số frame mỗi sample. |
| `STRIDE` | 10 | Bước trượt window (50% overlap). |
| `TEST_FRAC` | 0.15 | Tỉ lệ test set. |
| `VAL_FRAC` | 0.15 | Tỉ lệ val trong phần còn lại. |
| `N_SPLITS` | 5 | Số fold cho StratifiedKFold. |
| `EPOCHS` | 200 | Trần epoch — sẽ early-stop trước khi đạt. |
| `BATCH_SIZE` | 64 | |
| `SEED` | 42 | |

### Bước 2 — Discover & inspect

In bảng tóm tắt mỗi CSV: tên file, số dòng, nhãn trong file, có NaN không. Sanity check trước khi xử lý.

### Bước 3 — Cắt window

- Hàm `window_recording(df, window, stride)` cắt 1 recording dài thành các đoạn `(20, 8)`, bước trượt 10 (overlap 50%).
- Nếu trong 1 window có nhiều nhãn khác nhau → bỏ window đó (defensive; không xảy ra với schema hiện tại).
- NaN trong feature được fill bằng mean cột.

### Bước 4 — Cyclic encoding của `imu_x`

- `imu_x` là góc heading (đơn vị độ). Nếu để nguyên thì 359° và 1° "rất xa nhau" trong không gian số, nhưng thực tế chỉ cách nhau 2°. → Mạng học sẽ thấy bất liên tục giả tạo.
- Cách xử lý: thay `imu_x` bằng cặp `(sin(imu_x), cos(imu_x))` — biểu diễn vị trí trên vòng tròn đơn vị, không có discontinuity.
- **Số feature thực tế là 9** (không phải 8 như spec ban đầu) vì sin và cos là 2 biến độc lập:

```
flex1, flex2, flex3, flex4, flex5, imu_y, imu_z, sin_imu_x, cos_imu_x
```

- Model build với `input_shape = X.shape[1:]` nên tự khớp với 9 feature.
- Nếu muốn về đúng 8 feature: sửa cell 11 trong notebook để chỉ append 1 trong 2 component (mất một nửa thông tin góc — không khuyến nghị).

### Bước 5 — Encode label & shuffle

- `LabelEncoder` ánh xạ nhãn string → integer id.
- Shuffle bằng `np.random.permutation` với seed cố định để reproducible.

### Bước 6 — Train / val / test split

- Stratified split 70 / 15 / 15 theo **window** (sample-level).
- Test set giữ riêng cho bước evaluation cuối; phần `train + val` sẽ được tái-split trong K-Fold ở bước 10.
- **Cảnh báo leakage**: vì mỗi nhãn chỉ có 1 file gốc, các window kề nhau (do overlap STRIDE=10) có thể bị rò rỉ giữa train / val / test. Khi có nhiều bản ghi hơn cho mỗi nhãn, nên chuyển sang group-aware split (group key = filename).

### Bước 7 — Chuẩn hoá feature

- `StandardScaler` fit **chỉ trên rows của train** (reshape 3D → 2D, fit theo từng feature, reshape lại 3D).
- Apply cùng transform lên val, test, và sau này là inference input.
- Vì sao cần: `flex*` ~ [0, 0.025], còn `imu_y`, `imu_z` ~ hàng trăm. Không scale thì các flex bị "chìm".
- Mean / scale được lưu vào `data/processed/scaler.npz` để inference dùng lại đúng.

### Bước 8 — Class distribution

Bar chart số window mỗi nhãn để kiểm tra cân bằng. Hiện tại `pink4` ít hơn các nhãn khác đáng kể (file gốc chỉ 1140 dòng).

### Bước 9 — Lưu dataset đã xử lý

Output trong `data/processed/`:

| File | Nội dung |
|---|---|
| `X_train.npy`, `y_train.npy` | Tập train đã scale + label encoded. |
| `X_val.npy`, `y_val.npy` | Tập val. |
| `X_test.npy`, `y_test.npy` | Tập test giữ riêng. |
| `label_classes.npy` | Mảng nhãn theo thứ tự encoded. |
| `scaler.npz` | `mean`, `scale`, `feature_names`. |

Ngoài ra, `label_classes.npy` được copy ra `results/` để `src/inference/realtime_predict.py` có thể load thẳng theo đường dẫn nó đang dùng.

### Bước 10 — Train BiLSTM + Stratified 5-Fold CV

#### 10.1 Kiến trúc

```
Input (20, 9)
   │
   ▼  BatchNormalization
   ▼  Bidirectional(LSTM 64, return_sequences=True)
   ▼  Dropout(0.3)
   ▼  Bidirectional(LSTM 32, return_sequences=False)
   ▼  Dropout(0.3)
   ▼  Dense(32, ReLU)
   ▼  Dropout(0.2)
   ▼  Dense(n_classes, Softmax)
```

- Optimizer: **Adam(lr=1e-3)**
- Loss: **sparse_categorical_crossentropy** (vì `y` là int, không cần one-hot)
- Metric: `accuracy`

#### 10.2 Callbacks

| Callback | Setup |
|---|---|
| `ReduceLROnPlateau` | monitor `val_loss`, patience=10, factor=0.5, min_lr=1e-5 |
| `EarlyStopping` | monitor `val_loss`, patience=20, `restore_best_weights=True` |
| `ModelCheckpoint` | monitor `val_loss`, `save_best_only=True`, lưu mỗi fold |

#### 10.3 K-Fold setup

- Gộp `X_train + X_val` thành tập `X_full`, `y_full` (test giữ riêng cho evaluation).
- `StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)`.
- **Split theo chỉ số sample** (mỗi window là 1 sample shape `(20, 9)`), **không split theo row** — đảm bảo 1 window không bị xé đôi.
- Trong mỗi fold, defensive reshape về `(n_samples, 20, n_feat)` đúng spec trước khi fit.
- Với K=5 → tỉ lệ train/val mỗi fold tự động là 80/20.
- Reset seed mỗi fold (`SEED + fold`) để mô hình init khác nhau, tránh tình trạng 5 fold ra y hệt.

#### 10.4 Lưu mô hình

- `results/models/model_fold_{1..5}.h5` — best weights của từng fold (theo `val_loss`).
- `results/models/model_best.h5` — copy của fold có `val_loss` thấp nhất, là bản model production.

#### 10.5 Đánh giá

- Load `model_best.h5`, evaluate trên `X_test`.
- In `classification_report` (precision / recall / f1 từng nhãn) và confusion matrix.
- Plot training curves (loss & accuracy) chồng 5 fold để check overfitting / stability.

---

## 4. Khác biệt với pipeline cũ

| | Pipeline cũ (`Training.ipynb` / `realtime_predict.py`) | Pipeline mới (`Pipeline.ipynb`) |
|---|---|---|
| Input shape | `(20, 9)` | `(20, 9)` — nhưng feature khác |
| Feature thứ 9 | `face` (MediaPipe mouth ratio) | `cos_imu_x` |
| Feature thứ 8 | `imu_z` | `sin_imu_x` |
| Feature 6, 7 | `imu_x`, `imu_y` | `imu_y`, `imu_z` |
| 1 sample = | 1 file CSV (20 dòng) | 1 window 20 dòng trong recording dài |
| Source dir | `data/raw/*.csv` | `data/raw/Good Data/*.csv` |
| CV | `KFold` thường | `StratifiedKFold` |
| Cyclic encoding | Không | Có (sin/cos `imu_x`) |
| Feature scaling | Không có scaler persistent | `StandardScaler` lưu vào `scaler.npz` |

---

## 5. Việc cần làm tiếp (TODO)

1. **Cập nhật `src/inference/realtime_predict.py`** để khớp pipeline mới:
   - Bỏ `face` feature (hoặc thêm `face` vào pipeline mới nếu muốn giữ MediaPipe).
   - Apply `sin/cos` cho `imu_x` đầu vào từ glove.
   - Load `data/processed/scaler.npz` và apply cùng transform.
2. **Thu thêm dữ liệu** — hiện mỗi nhãn chỉ có 1 file gốc → leakage trong stratified split. Cần nhiều phiên ghi hơn cho mỗi nhãn để dùng group-aware split.
3. **Chuẩn hoá label** — `khong?2` có mojibake (Unicode hỏng), `C.csv` có `SIGN=test2` không khớp filename. Cần fix nhãn ngọn nguồn hoặc cleanup khi load.
4. **Pin requirements** — `requirements.txt` chưa pin version; pin lại sau khi setup environment ổn định.
5. **Move model + training code ra module Python** — hiện đang inline trong notebook; nên promote ra `src/models/` và `src/training/` để reuse được.
