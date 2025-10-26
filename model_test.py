import tensorflow as tf

MODEL_PATH = "C:\Users\reshm\OneDrive\Desktop\temple\model_with_metadata.tflite"
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
