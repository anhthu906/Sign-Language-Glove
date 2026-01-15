import tensorflow as tf
import tf2onnx
import os

# Đường dẫn đến model .h5
model_path = "/Users/stella/Downloads/IS/MVP2/Data Training/Test1.h5"

# Load lại model
print(" Đang load model...")
model = tf.keras.models.load_model(model_path)
print("✅ Model đã được load thành công!")

# 🔹 Kiểm tra input shape
print(f"Input shape: {model.inputs[0].shape}")
print(f"Output shape: {model.outputs[0].shape}")

# 🔹 Đường dẫn file .onnx đầu ra
onnx_path = "/Users/stella/Downloads/IS/MVP2/Data Training/Test1.onnx"

# 🔹 Chuyển đổi sang ONNX
print("🔄 Đang chuyển đổi sang ONNX...")
try:
    spec = (tf.TensorSpec(model.inputs[0].shape, tf.float32, name="input"),)
    model_proto, _ = tf2onnx.convert.from_keras(
        model,
        input_signature=spec,
        opset=13,
        output_path=onnx_path,
    )
    print(f"Model đã được xuất thành công: {onnx_path}")
except Exception as e:
    print("Lỗi khi chuyển sang ONNX:")
    print(e)

# 🔹 Kiểm tra file có thực sự tồn tại không
if os.path.exists(onnx_path):
    print(f"📁 File ONNX đã được tạo: {onnx_path}")
else:
    print("⚠️ Không tìm thấy file ONNX trong thư mục.")
