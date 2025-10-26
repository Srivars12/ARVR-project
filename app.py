import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image

st.set_page_config(page_title="Temple Object Detection", layout="centered")
st.title("🛕 Temple Caption Display")

# -------------------------------
# Load CSV
# -------------------------------
@st.cache_data
def load_csv(csv_path="captions.csv"):
    df = pd.read_csv(csv_path)
    return df

df = load_csv()

DEFAULT_CLASS = "Unknown"
DEFAULT_CAPTION = "No caption available for this image"

# -------------------------------
# Get caption based on filename
# -------------------------------
def get_caption_from_csv(uploaded_file, df):
    uploaded_name = uploaded_file.name
    row = df[df['image_path'].str.endswith(uploaded_name, na=False)]
    if not row.empty:
        # Use the class name inferred from filename if needed
        file_path = row.iloc[0]['image_path']
        class_name = None
        for cls in ["Gopuram", "Sculpture", "Yazhi", "Vehicle", "Deity"]:
            if cls.lower() in file_path.lower():
                class_name = cls
                break
        if class_name is None:
            class_name = DEFAULT_CLASS
        return class_name, row.iloc[0]['caption']
    else:
        return DEFAULT_CLASS, DEFAULT_CAPTION

# -------------------------------
# Streamlit UI
# -------------------------------
mode = st.radio("Select Mode", ["Upload Image", "Webcam Capture"])

# ---------- Upload Image ----------
if mode == "Upload Image":
    file = st.file_uploader("Upload Temple Image", type=["jpg","jpeg","png"])
    if file:
        image = Image.open(file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        class_name, caption = get_caption_from_csv(file, df)
        
        st.success(f"✅ Class: {class_name}")
        st.info(f"📌 Caption: {caption}")

# ---------- Webcam Capture ----------
else:
    if 'cap' not in st.session_state:
        st.session_state.cap = None
    if 'captured_frame' not in st.session_state:
        st.session_state.captured_frame = None

    col1, col2, col3 = st.columns(3)
    with col1:
        start = st.button("Start Webcam")
    with col2:
        capture = st.button("Capture Photo")
    with col3:
        stop = st.button("Stop Webcam")

    # Start webcam
    if start:
        if st.session_state.cap is None:
            st.session_state.cap = cv2.VideoCapture(0)
            if not st.session_state.cap.isOpened():
                st.error("❌ Webcam not accessible.")
            else:
                st.success("✅ Webcam Started")

    # Stop webcam
    if stop:
        if st.session_state.cap:
            st.session_state.cap.release()
            st.session_state.cap = None
            st.session_state.captured_frame = None
            st.success("🛑 Webcam Stopped")

    # Read a single frame
    if st.session_state.cap:
        ret, frame = st.session_state.cap.read()
        if ret:
            st.session_state.captured_frame = frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st.image(frame_rgb, channels="RGB", caption="Live Webcam Feed")

    # Capture photo and get caption
    if capture and st.session_state.captured_frame is not None:
        frame = st.session_state.captured_frame
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        st.image(img_pil, caption="Captured Image", use_column_width=True)

        # Use a placeholder filename since webcam images are new
        # Here we try to match with CSV by image similarity if possible
        # For simplicity, we show default
        class_name = DEFAULT_CLASS
        caption = DEFAULT_CAPTION

        st.success(f"✅ Class: {class_name}")
        st.info(f"📌 Caption: {caption}")
