import sys
import os
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

# =========================
# Config
# =========================
SAVE_DIR = "saved_models"
os.makedirs(SAVE_DIR, exist_ok=True)
MODEL_PATH = os.path.join(SAVE_DIR, "emotion_cnn_final_20260225_161636.keras")

IMG_HEIGHT = 48
IMG_WIDTH = 48

CLASS_LABELS = ['Anger', 'Disgust', 'Fear', 'Happy',
                'Neutral', 'Sadness', 'Surprise']

# =========================
# Check model
# =========================
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

print("✔ Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

# =========================
# Check image path argument
# =========================
# if len(sys.argv) < 2:
#     print("Usage: python predict_single_image.py path_to_image")
#     sys.exit(1)

image_path = "broke_up.jpg" 
# image_path = sys.argv[1]

if not os.path.exists(image_path):
    raise FileNotFoundError(f"Image not found: {image_path}")

# =========================
# Preprocess Image
# =========================
img_bgr = cv2.imread(image_path)

if img_bgr is None:
    raise ValueError("Unable to read image.")

img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
img_resized = cv2.resize(img_gray, (IMG_WIDTH, IMG_HEIGHT))
img_normalized = img_resized.astype("float32") / 255.0

# (48,48) → (48,48,1)
img_input = np.expand_dims(img_normalized, axis=-1)
# (1,48,48,1)
img_input = np.expand_dims(img_input, axis=0)

# =========================
# Predict
# =========================
prediction = model.predict(img_input)
pred_idx = int(np.argmax(prediction[0]))
confidence = float(np.max(prediction[0]))

print("Predicted Emotion:", CLASS_LABELS[pred_idx])
print("Confidence:", round(confidence * 100, 2), "%")

# =========================
# Visualization
# =========================
plt.figure(figsize=(10, 4))

# 左邊顯示圖片
plt.subplot(1, 2, 1)
plt.imshow(img_gray, cmap='gray')
plt.title(f"Pred: {CLASS_LABELS[pred_idx]}\nConf: {confidence*100:.2f}%")
plt.axis("off")

# 右邊顯示機率分布
plt.subplot(1, 2, 2)
plt.bar(CLASS_LABELS, prediction[0])
plt.xticks(rotation=45)
plt.title("Probability Distribution")
plt.ylim([0, 1])

plt.tight_layout()
plt.show()