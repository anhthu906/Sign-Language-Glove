from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
import csv
import os
import uvicorn

app = FastAPI()

# File CSV để lưu dữ liệu
CSV_FILE = 'sensors_data_A.csv'

# List tạm để thu thập 20 frames (chỉ cho tay trái, phần tay phải sẽ đặt mặc định là 0 hoặc giá trị placeholder)
frames = []

# Hàm để tạo header CSV dựa trên format của file bạn cung cấp (chỉ left, right placeholder)
def get_csv_headers():
    headers = []
    # Flex Left (5 flex sensors, mỗi cái 20 frames)
    for flex in range(1, 6):
        for frame in range(1, 21):
            headers.append(f'Flex-Left-{flex}-Frame-{frame}')
    # Position Left (X,Y,Z cho 20 frames)
    for frame in range(1, 21):
        headers.append(f'Position-X-Left-Frame-{frame}')
        headers.append(f'Position-Y-Left-Frame-{frame}')
        headers.append(f'Position-Z-Left-Frame-{frame}')
    # Orientation Left (X,Y,Z cho 20 frames)
    for frame in range(1, 21):
        headers.append(f'Orientation-X-Left-Frame-{frame}')
        headers.append(f'Orientation-Y-Left-Frame-{frame}')
        headers.append(f'Orientation-Z-Left-Frame-{frame}')
    # Flex Right (placeholder, vì chỉ có tay trái)
    for flex in range(1, 6):
        for frame in range(1, 21):
            headers.append(f'Flex-Right-{flex}-Frame-{frame}')
    # Position Right (placeholder)
    for frame in range(1, 21):
        headers.append(f'Position-X-Right-Frame-{frame}')
        headers.append(f'Position-Y-Right-Frame-{frame}')
        headers.append(f'Position-Z-Right-Frame-{frame}')
    # Orientation Right (placeholder)
    for frame in range(1, 21):
        headers.append(f'Orientation-X-Right-Frame-{frame}')
        headers.append(f'Orientation-Y-Right-Frame-{frame}')
        headers.append(f'Orientation-Z-Right-Frame-{frame}')
    # SIGN
    headers.append('SIGN')
    return headers

# Kiểm tra và tạo file CSV nếu chưa tồn tại, thêm header
print(f"Checking CSV file: {CSV_FILE}")
if not os.path.exists(CSV_FILE):
    print("Creating new CSV file with headers...")
    with open(CSV_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(get_csv_headers())
print("CSV setup complete.")

class SensorData(BaseModel):
    flex1: int = 0
    flex2: int = 0
    flex3: int = 0
    flex4: int = 0
    flex5: int = 0
    posX: float = 0.0
    posY: float = 0.0
    posZ: float = 0.0
    oriX: float = 0.0
    oriY: float = 0.0
    oriZ: float = 0.0

@app.post("/data")
async def receive_data(data: SensorData):
    global frames
    # Lưu frame hiện tại
    frame_data = {
        'flex1': data.flex1,
        'flex2': data.flex2,
        'flex3': data.flex3,
        'flex4': data.flex4,
        'flex5': data.flex5,
        'posX': data.posX,
        'posY': data.posY,
        'posZ': data.posZ,
        'oriX': data.oriX,
        'oriY': data.oriY,
        'oriZ': data.oriZ
    }
    frames.append(frame_data)
    print(f"Received frame {len(frames)}/20")
    
    # Nếu đủ 20 frames, lưu vào CSV
    if len(frames) == 20:
        row = []
        # Flex Left
        for flex in ['flex1', 'flex2', 'flex3', 'flex4', 'flex5']:
            for frame in frames:
                row.append(frame[flex])
        # Position Left
        for frame in frames:
            row.append(frame['posX'])
            row.append(frame['posY'])
            row.append(frame['posZ'])
        # Orientation Left
        for frame in frames:
            row.append(frame['oriX'])
            row.append(frame['oriY'])
            row.append(frame['oriZ'])
        # Flex Right (placeholder: 0 cho tất cả)
        row.extend([0] * 100)  # 5 flex * 20 frames
        # Position Right (placeholder: 0.0)
        row.extend([0.0] * 60)  # 3 coords * 20 frames
        # Orientation Right (placeholder: 0.0)
        row.extend([0.0] * 60)  # 3 coords * 20 frames
        # SIGN
        row.append('A')  # Lưu SIGN "A" như yêu cầu
        
        # Append vào CSV
        with open(CSV_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
        
        # Reset frames
        frames = []
        print("Saved 20 frames to CSV with SIGN 'A'")
    
    return {"status": "success"}

if __name__ == "__main__":
    print("Starting FastAPI server on http://0.0.0.0:8000 ...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")