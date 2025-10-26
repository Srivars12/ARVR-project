# robust_inference.py
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
import pandas as pd
import argparse
import os
from collections import deque

# ----- CONFIG -----
MODEL_PATH = "temple_object_model.h5"
CAPTIONS_CSV = "captions.csv"
IMG_SIZE = (224, 224)
CONF_THRESHOLD = 0.60      # if max prob < threshold => Unknown
SMOOTH_WINDOW = 7          # number of frames for majority voting
CLASS_ORDER = {0: "Sculptures", 1: "Yazhi", 2: "Deity", 3: "Vehicle", 4: "Gopuram"}
# ------------------

def load_model(path):
    return tf.keras.models.load_model(path)

def load_captions(csv_path):
    df = pd.read_csv(csv_path)
    # try to support common column names
    if 'class' in df.columns and 'caption' in df.columns:
        return dict(zip(df['class'], df['caption']))
    elif 'class_name' in df.columns and 'caption' in df.columns:
        return dict(zip(df['class_name'], df['caption']))
    else:
        # fallback assume first two cols are label and caption
        return dict(zip(df.iloc[:,0], df.iloc[:,1]))

def apply_clahe_rgb(img):
    # img: numpy RGB
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    lab = cv2.merge((l,a,b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

def gamma_correction(img, gamma=1.0):
    inv = 1.0 / gamma
    table = np.array([((i/255.0) ** inv) * 255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(img, table)

def preprocess(img_rgb):
    # apply CLAHE + gamma modest boost
    img = apply_clahe_rgb(img_rgb)
    img = gamma_correction(img, gamma=1.1)
    img = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_LINEAR)
    x = img.astype(np.float32) / 255.0
    return np.expand_dims(x, 0)

def tta_predictions(model, img_rgb):
    # generate several small variations and average probs
    crops = []
    h, w, _ = img_rgb.shape
    # center crop
    center = img_rgb
    crops.append(center)
    # five crops: center + four corners (resized)
    smaller = cv2.resize(img_rgb, (int(w*0.9), int(h*0.9)))
    crops.append(smaller)
    crops.append(cv2.flip(center, 1))
    crops.append(gamma_correction(center, 0.9))
    probs = []
    for c in crops:
        x = preprocess(c)
        p = model.predict(x)[0]
        probs.append(p)
    avg = np.mean(probs, axis=0)
    return avg

def predict_frame(model, frame):
    probs = tta_predictions(model, frame)
    max_idx = int(np.argmax(probs))
    max_conf = float(np.max(probs))
    if max_conf < CONF_THRESHOLD:
        return "Unknown", max_conf, probs
    return CLASS_ORDER.get(max_idx, str(max_idx)), max_conf, probs

def run_camera():
    model = load_model(MODEL_PATH)
    captions = load_captions(CAPTIONS_CSV)
    cap = cv2.VideoCapture(0)
    hist = deque(maxlen=SMOOTH_WINDOW)

    if not cap.isOpened():
        print("Cannot open camera")
        return

    print("Press q to quit")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        label, conf, probs = predict_frame(model, frame_rgb)
        hist.append(label)
        # majority vote
        final_label = max(set(hist), key=hist.count)

        caption_text = captions.get(final_label, "No caption available.")
        # overlay
        cv2.putText(frame, f"{final_label} {conf:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
        cv2.putText(frame, caption_text, (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
        cv2.imshow("Robust Camera", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_camera()
