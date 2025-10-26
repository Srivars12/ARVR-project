import cv2
import tensorflow as tf
import numpy as np
import pandas as pd

captions_df = pd.read_csv("captions.csv")

interpreter = tf.lite.Interpreter(model_path="model_with_metadata.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_resized = cv2.resize(frame, (224, 224))
    input_data = np.expand_dims(img_resized.astype(np.float32) / 255.0, axis=0)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    class_id = np.argmax(output_data)
    class_name = captions_df.iloc[class_id]["category"]
    caption_text = captions_df.iloc[class_id]["caption"]

    cv2.putText(frame, f"{class_name}: {caption_text}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Temple Object Caption App", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
