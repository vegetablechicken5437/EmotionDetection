import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from datetime import datetime
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, MaxPool2D, Flatten, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# =========================
# Paths / Hyperparams
# =========================
train_dir = "train"
test_dir  = "test"

SEED = 12
IMG_HEIGHT = 48
IMG_WIDTH  = 48
BATCH_SIZE = 32
EPOCHS = 30
LR = 0.001
NUM_CLASSES = 7
CLASS_LABELS = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sadness', "Surprise"]

# =========================
# Data Generators
# =========================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    validation_split=0.2
)

# train_dir 的 0.2 當 validation）
test_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    directory=train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    shuffle=True,
    color_mode="grayscale",
    class_mode="categorical",
    subset="training",
    seed=SEED
)

validation_generator = test_datagen.flow_from_directory(
    directory=train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    shuffle=False,
    color_mode="grayscale",
    class_mode="categorical",
    subset="validation",
    seed=SEED
)

test_generator = test_datagen.flow_from_directory(
    directory=test_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    shuffle=False,
    color_mode="grayscale",
    class_mode="categorical",
    seed=SEED
)

# =========================
# (Optional) Preview Images
# =========================
def display_one_image(image, title, subplot, color):
    plt.subplot(subplot)
    plt.axis('off')
    plt.imshow(image)
    plt.title(title, fontsize=16, color=color)

def display_nine_images(images, titles, title_colors=None):
    plt.figure(figsize=(13, 13))
    for i in range(9):
        color = 'black' if title_colors is None else title_colors[i]
        display_one_image(images[i], titles[i], 331 + i, color)

img_datagen = ImageDataGenerator(rescale=1./255)
img_generator = img_datagen.flow_from_directory(
    directory=train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    shuffle=True,
    color_mode="rgb",
    class_mode="categorical",
    seed=SEED
)

images, classes = next(img_generator)
class_idxs = np.argmax(classes, axis=-1)
labels = [CLASS_LABELS[idx] for idx in class_idxs]
# display_nine_images(images, labels)
# plt.show()

# =========================
# Models
# =========================
def SimpleModel():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(7, activation='softmax'))
    return model

def ImprovedModel():
    model = Sequential([
        Conv2D(64, (3, 3), activation='relu', input_shape=(48, 48, 1), padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPool2D(2, 2),
        Dropout(0.25),

        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPool2D(2, 2),
        Dropout(0.25),

        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(7, activation='softmax')
    ])
    return model

# =========================
# Build / Compile
# =========================
model = ImprovedModel()
# model = SimpleModel()

model.summary()

model.compile(
    loss='categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
    metrics=["accuracy"]
)

# =========================
# Auto Load or Train
# =========================
SAVE_DIR = "saved_models"
os.makedirs(SAVE_DIR, exist_ok=True)

LATEST_MODEL_PATH = os.path.join(SAVE_DIR, "emotion_cnn_final_20260225_161636.keras")
BEST_WEIGHTS_PATH = os.path.join(SAVE_DIR, "emotion_cnn_best_20260225_161636.weights.h5")

if os.path.exists(LATEST_MODEL_PATH):
    print("✔ Found existing model. Loading model...")
    model = tf.keras.models.load_model(LATEST_MODEL_PATH)
else:
    print("⚠ No saved model found. Start training...")

    callbacks = [
        ModelCheckpoint(
            filepath=BEST_WEIGHTS_PATH,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=6,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
    ]

    record = model.fit(
        x=train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator,
        callbacks=callbacks
    )

    # 存完整模型
    model.save(LATEST_MODEL_PATH)
    print("✔ Model saved to:", LATEST_MODEL_PATH)

# =========================
# Load best weights if exists
# =========================
if os.path.exists(BEST_WEIGHTS_PATH):
    print("✔ Loading best weights...")
    model.load_weights(BEST_WEIGHTS_PATH)

# =========================
# Evaluate
# =========================
print("Testing...")
model.evaluate(test_generator)

# =========================
# Visualization Helper
# =========================
def visualize_test_results(generator, model, num_images=12):
    test_images, test_labels = next(generator)
    predictions = model.predict(test_images)

    plt.figure(figsize=(20, 10))
    for i in range(num_images):
        plt.subplot(3, 4, i + 1)

        pred_idx = int(np.argmax(predictions[i+10]))
        true_idx = int(np.argmax(test_labels[i+10]))
        color = 'green' if pred_idx == true_idx else 'red'

        plt.imshow(test_images[i].reshape(48, 48), cmap='gray')
        plt.title(f"Pred: {CLASS_LABELS[pred_idx]}\nTrue: {CLASS_LABELS[true_idx]}", color=color, fontsize=10)
        plt.axis('off')

    plt.tight_layout()
    plt.show()

# Run visualization
visualize_test_results(test_generator, model)