from flask import Flask, request, jsonify
import csv
import os

app = Flask(__name__)

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
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(get_csv_headers())

@app.route('/data', methods=['POST'])
def receive_data():
    global frames
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No JSON data received'}), 400
    
    # Lưu frame hiện tại (flex1-5, posX/Y/Z, oriX/Y/Z)
    frame_data = {
        'flex1': data.get('flex1', 0),
        'flex2': data.get('flex2', 0),
        'flex3': data.get('flex3', 0),
        'flex4': data.get('flex4', 0),
        'flex5': data.get('flex5', 0),
        'posX': data.get('posX', 0.0),
        'posY': data.get('posY', 0.0),
        'posZ': data.get('posZ', 0.0),
        'oriX': data.get('oriX', 0.0),
        'oriY': data.get('oriY', 0.0),
        'oriZ': data.get('oriZ', 0.0)
    }
    frames.append(frame_data)
    
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
    
    return jsonify({'status': 'success'}), 200

if __name__ == '__main__':
    # Chạy server trên port 5000, host '0.0.0.0' để nhận từ ESP32
    app.run(host='0.0.0.0', port=5000, debug=True)