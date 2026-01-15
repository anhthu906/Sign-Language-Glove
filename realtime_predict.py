import cv2
import mediapipe as mp
import numpy as np
import time
import serial
import csv
import os
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

# ================= CONFIG =================
SERIAL_PORT = "/dev/tty.usbmodem101"
BAUD        = 115200

SAVE_FOLDER = "data_test"
FRAMES_PER_SAMPLE = 25        # số frame thu 1 mẫu test
FRAME_DELAY = 0.08            # delay giữa các frame

MODEL_PATH = "model_best.h5"
LABEL_ENCODER_PATH = "label_classes.npy"


LABEL = input("\nNhập ký hiệu cần test model (Tên để lưu file): ")

os.makedirs(SAVE_FOLDER, exist_ok=True)

# Serial to get glove sensor
ser = serial.Serial(SERIAL_PORT, BAUD, timeout=1)
time.sleep(2)

mp_holistic = mp.solutions.holistic


# ------- Extract mouth feature --------
def get_face_feature(results,w,h):
    if not results.face_landmarks: return 0
    lm = results.face_landmarks.landmark
    def P(i): return np.array([lm[i].x*w, lm[i].y*h])
    mouth_w=np.linalg.norm(P(61)-P(291))
    mouth_h=np.linalg.norm(P(13)-P(14))
    return round(mouth_h/(mouth_w+1e-6),3)


# ---------------- Collecting Section ----------------
print(f"\n🟩 THU 1 SAMPLE TEST — Target {FRAMES_PER_SAMPLE} frames\n"
      "❗ Giữ tay đúng ký hiệu → camera tự thu đủ\n"
      "❗ Nhấn Q để dừng\n")

cap = cv2.VideoCapture(0)
frames = []

with mp_holistic.Holistic(
    model_complexity=1, refine_face_landmarks=True,
    min_detection_confidence=0.5, min_tracking_confidence=0.5
) as holistic:

    for t in range(3,0,-1):
        ret, frame = cap.read()
        frame = cv2.flip(frame,1)
        cv2.putText(frame,f"Chuẩn bị thu... {t}s",(10,50),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),3)
        cv2.imshow("Collecting",frame); cv2.waitKey(1)
        time.sleep(1)

    print("\n▶ BẮT ĐẦU THU")

    while len(frames)<FRAMES_PER_SAMPLE:
        ret,frame = cap.read()
        if not ret: break
        frame=cv2.flip(frame,1)
        h,w,_ = frame.shape
        
        rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        result=holistic.process(rgb)
        face = get_face_feature(result,w,h)

        # read glove
        data = ser.readline().decode().strip().split()
        if len(data)<8: continue
        
        f1,f2,f3,f4,f5 = map(int,data[:5])
        x,y,z = map(float,data[5:8])
        frames.append([f1,f2,f3,f4,f5,x,y,z,face])

        cv2.putText(frame,f"Frame {len(frames)}/{FRAMES_PER_SAMPLE}",
                    (10,40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        cv2.imshow("Collecting",frame); cv2.waitKey(1)
        time.sleep(FRAME_DELAY)

cap.release()
ser.close()
cv2.destroyAllWindows()


# Save sample
filename=f"{SAVE_FOLDER}/{LABEL}_{int(time.time())}.csv"
with open(filename,'w',newline='') as f:
    writer=csv.writer(f)
    writer.writerow(["flex1","flex2","flex3","flex4","flex5","x","y","z","face"])
    writer.writerows(frames)

print(f"\n📁 File sample saved → {filename}")
print(f"📊 Total frames collected: {len(frames)}")


# ==================== PREDICT ====================
print("\n🔍 Loading model & predicting...")

model = load_model(MODEL_PATH)
classes = np.load(LABEL_ENCODER_PATH, allow_pickle=True)

le = LabelEncoder()
le.classes_ = classes

sample = np.array(frames)

# ---- AUTO FIX SHAPE (Cách 2) ----
TIME_STEPS = model.input_shape[1]   # lấy từ model (ex: 20)

if sample.shape[0] > TIME_STEPS:
    sample = sample[:TIME_STEPS]  # trim bớt
else:
    pad_len = TIME_STEPS - sample.shape[0]
    pad = np.tile(sample[-1], (pad_len,1))
    sample = np.concatenate([sample, pad], axis=0)

sample = sample.reshape(1,TIME_STEPS, -1)

# Predict
probs = model.predict(sample)[0]
pred_idx = np.argmax(probs)
pred_label = le.inverse_transform([pred_idx])[0]
confidence = probs[pred_idx]*100

print(f"\n Model dự đoán ký hiệu bạn vừa làm là: **{pred_label}**")
print(f" Độ tin cậy: {confidence:.2f}%\n")

print("Thu thêm sample khác để kiểm tra độ ổn định.")
