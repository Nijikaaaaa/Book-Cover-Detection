import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    average_precision_score
)
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.callbacks import EarlyStopping

DATA_DIR = "mini"
CSV_FILES = [
    "Graphic-Novels-Anime-Manga.csv",
    "Romance.csv",                     
    "Teaching-Resources-Education.csv"
]
IMG_SIZE = (128, 128)
EPOCHS = 100
BATCH_SIZE = 32

# Loading CSVs
dfs = []
label_map = {0: "Normal", 1: "Anime/Comic"}

for csv_file in CSV_FILES:
    path = os.path.join(DATA_DIR, csv_file)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(path)
    label_val = 1 if "Graphic-Novels-Anime-Manga" in csv_file else 0
    df["label"] = label_val
    dfs.append(df)

df_all = pd.concat(dfs, ignore_index=True)
print("Label Map:", label_map)
print("CSV Columns:", df_all.columns.tolist())
print("First row example:\n", df_all.iloc[0])

image_col = "img_paths"
if image_col not in df_all.columns:
    raise ValueError(f"Column '{image_col}' not found in CSVs.")

# Preprocess function
def preprocess_image(img_path):
    if not os.path.exists(img_path):
        print(f"Missing file: {img_path}")
        return None
    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to load: {img_path}")
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, IMG_SIZE)
    img_resized = img_resized.astype("float32") / 255.0
    return img_resized

# Load images
images, labels = [], []
for _, row in df_all.iterrows():
    img_path = str(row[image_col]).replace("dataset/", "")
    if not os.path.isabs(img_path):
        img_path = os.path.join(DATA_DIR, img_path)

    img = preprocess_image(img_path)
    if img is not None:
        images.append(img)
        labels.append(row["label"])

images = np.array(images)
labels = np.array(labels)
print(f"Loaded {len(images)} processed images.")

if len(images) == 0:
    raise RuntimeError("No images were loaded. Check if img_paths in CSVs are correct and point to local files.")

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(
    images, labels, test_size=0.2, stratify=labels, random_state=42
)

# Model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

early_stop = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

# Train
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop],
    verbose=1
)

# Save model
model.save("book_mobilenet_binary.h5")
print("Model saved as book_mobilenet_binary.h5")

# Accuracy curves
plt.figure(figsize=(8, 5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training vs Validation Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# Predictions
y_prob = model.predict(X_val)
y_pred = (y_prob > 0.5).astype("int32")

# Confusion matrix
cm = confusion_matrix(y_val, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Anime/Comic"])
disp.plot(cmap=plt.cm.Blues)
plt.show()

# Extra metrics
precision = precision_score(y_val, y_pred)
recall = recall_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)
mAP = average_precision_score(y_val, y_prob)

print("\nClassification Report:")
print(classification_report(y_val, y_pred, target_names=["Normal", "Anime/Comic"]))
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")
print(f"mAP:       {mAP:.4f}")