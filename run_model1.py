import onnx
import onnxruntime as ort
import numpy as np

# Đường dẫn file ONNX
onnx_path = "/Users/stella/Downloads/IS/MVP2/Data Training/Test.onnx"

print("🔍 Kiểm tra model ONNX...")

# 1️⃣ Kiểm tra file ONNX có hợp lệ không
model = onnx.load(onnx_path)
onnx.checker.check_model(model)
print("✅ Model hợp lệ về cấu trúc!")

# 2️⃣ Tạo session để chạy thử model
session = ort.InferenceSession(onnx_path)
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape
output_name = session.get_outputs()[0].name

print("📄 Input name:", input_name)
print("📐 Input shape:", input_shape)
print("🎯 Output name:", output_name)

# 3️⃣ Tạo input giả (dummy input) có đúng kích thước
dummy_input = np.random.randn(*[s if isinstance(s, int) else 1 for s in input_shape]).astype(np.float32)

# 4️⃣ Chạy thử model
outputs = session.run(None, {input_name: dummy_input})

print("✅ Model chạy inference thành công!")
print("🧾 Kết quả dự đoán mẫu:")
print(outputs[0])
