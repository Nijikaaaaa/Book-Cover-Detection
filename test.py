import os
import cv2
import numpy as np
import tensorflow as tf
import argparse
import pandas as pd
import pytesseract
from PIL import Image
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

MODEL_PATH = "book_mobilenet_binary.h5"
IMG_SIZE = (128, 128)
LABEL_MAP = {0: "Normal", 1: "Anime/Comic"}

model = tf.keras.models.load_model(MODEL_PATH)

def preprocess_image(img_path):
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Failed to load image: {img_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, IMG_SIZE)
    img_resized = img_resized.astype("float32") / 255.0
    return np.expand_dims(img_resized, axis=0)

def classify_image(img_path):
    img = preprocess_image(img_path)
    preds = model.predict(img, verbose=0)
    preds = np.array(preds)
    if preds.ndim == 2 and preds.shape[1] == 1:
        prob = float(preds[0, 0])
        pred_label = 1 if prob >= 0.5 else 0
        confidence = prob if pred_label == 1 else 1.0 - prob
    elif preds.ndim == 2 and preds.shape[1] == 2:
        prob = float(preds[0, 1])
        pred_label = 1 if prob >= 0.5 else 0
        confidence = prob if pred_label == 1 else 1.0 - prob
    else:
        pred_label = int(np.argmax(preds, axis=1)[0])
        confidence = float(np.max(preds))
    return LABEL_MAP[pred_label], confidence

def extract_title(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return "[Image load error]"
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 9)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(thresh)
    min_height = img.shape[0] * 0.05
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h >= min_height:
            cv2.drawContours(mask, [cnt], -1, 255, -1)
    filtered = cv2.bitwise_and(thresh, mask)
    temp_path = "temp_ocr.png"
    cv2.imwrite(temp_path, filtered)
    text = pytesseract.image_to_string(Image.open(temp_path), config="--oem 3 --psm 6").strip()
    if os.path.exists(temp_path):
        os.remove(temp_path)
    return text if text else "[No title detected]"

def draw_results(img_path, pred_class, conf, title, output_path):
    img = cv2.imread(img_path)
    if img is None:
        return
    cv2.putText(img, f"{pred_class} ({conf:.2f})", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.putText(img, f"Title: {title}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.imwrite(output_path, img)

def main():
    parser = argparse.ArgumentParser(description="Classify book cover and extract title")
    parser.add_argument("image", type=str, help="Path to input image or folder")
    parser.add_argument("--output", type=str, default="result.jpg", help="Path to save annotated image (single image)")
    parser.add_argument("--csv", type=str, default="results.csv", help="Path to save CSV log")
    args = parser.parse_args()
    results = []
    if os.path.isdir(args.image):
        files = [os.path.join(args.image, f) for f in sorted(os.listdir(args.image)) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    else:
        files = [args.image]
    for img_path in files:
        try:
            pred_class, conf = classify_image(img_path)
        except Exception as e:
            print(f"[ERROR] classification failed for {img_path}: {e}")
            pred_class, conf = "[ERROR]", 0.0
        try:
            title = extract_title(img_path)
        except Exception as e:
            print(f"[ERROR] OCR failed for {img_path}: {e}")
            title = "[OCR Error]"
        print(f"[{os.path.basename(img_path)}] {pred_class} ({conf:.2f}) | Title: {title}")
        results.append({"filename": os.path.basename(img_path), "predicted_class": pred_class, "confidence": conf, "title": title})
        if len(files) == 1:
            draw_results(img_path, pred_class, conf, title, args.output)
    df = pd.DataFrame(results)
    df.to_csv(args.csv, index=False)
    print(f"[INFO] Results saved to {args.csv}")

if __name__ == "__main__":
    main()