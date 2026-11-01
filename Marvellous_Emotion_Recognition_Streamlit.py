"""
Marvellous Emotion Recognition - Streamlit App
Uses a trained FER2013 CNN model and OpenCV Haar Cascade face detection.

Expected files in the same folder:
- Emotion_Detection_CNN.keras
- emotion_labels.txt (optional; defaults to standard FER2013 labels)

Run:
    streamlit run Marvellous_Emotion_Recognition_Streamlit.py
"""

from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

IMAGE_SIZE = 48
MODEL_PATH = Path(__file__).resolve().parent / "Emotion_Detection_CNN.keras"
LABELS_PATH = Path(__file__).resolve().parent / "emotion_labels.txt"
DEFAULT_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

st.set_page_config(
    page_title="Marvellous Emotion Recognition",
    page_icon="😊",
    layout="wide",
)

st.title("😊 Real-Time Emotion Recognition")
st.caption("CNN + FER2013 + OpenCV | Upload an image or capture a photo with your camera")

@st.cache_resource(show_spinner="Loading the trained CNN model...")
def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}\n"
            "Train the model first, then copy Emotion_Detection_CNN.keras "
            "into this app's folder."
        )
    return tf.keras.models.load_model(MODEL_PATH)

def load_labels():
    if LABELS_PATH.is_file():
        labels = [
            line.strip()
            for line in LABELS_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(labels) == 7:
            return labels
    return DEFAULT_LABELS

def find_faces_and_predict(rgb_image, model, labels):
    """Return annotated RGB image and prediction details for detected faces."""
    # OpenCV uses BGR; the incoming image is RGB.
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        raise RuntimeError(
            "OpenCV could not load haarcascade_frontalface_default.xml. "
            "Reinstall opencv-contrib-python or opencv-python version 4.x."
        )

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
    )

    results = []
    annotated = bgr_image.copy()

    for (x, y, w, h) in faces:
        face_gray = gray[y:y + h, x:x + w]
        if face_gray.size == 0:
            continue

        face_resized = cv2.resize(face_gray, (IMAGE_SIZE, IMAGE_SIZE))
        face_array = face_resized.astype(np.float32) / 255.0
        face_array = face_array.reshape(1, IMAGE_SIZE, IMAGE_SIZE, 1)

        probabilities = model.predict(face_array, verbose=0)[0]
        predicted_index = int(np.argmax(probabilities))
        emotion = labels[predicted_index] if predicted_index < len(labels) else DEFAULT_LABELS[predicted_index]
        confidence = float(probabilities[predicted_index]) * 100

        # Draw the detected face and prediction label.
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (40, 220, 40), 2)
        label_text = f"{emotion}: {confidence:.1f}%"
        text_y = max(25, y - 10)
        cv2.putText(
            annotated,
            label_text,
            (x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (40, 220, 40),
            2,
            cv2.LINE_AA,
        )

        results.append({
            "Face": len(results) + 1,
            "Predicted emotion": emotion,
            "Model confidence": f"{confidence:.1f}%",
        })

    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    return annotated_rgb, results

with st.sidebar:
    st.header("About")
    st.write("This app uses your already-trained FER2013 CNN model.")
    st.write("The seven classes are Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral.")
    st.info(
        "Prediction confidence is the model's score, not a scientifically verified "
        "measurement of a person's actual feelings."
    )
    input_method = st.radio(
        "Choose input",
        ["Upload image", "Take a camera photo"],
        index=0,
    )

try:
    model = load_model()
    labels = load_labels()
except Exception as exc:
    st.error(str(exc))
    st.stop()

uploaded_image = None
if input_method == "Upload image":
    uploaded_file = st.file_uploader(
        "Upload a face image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        help="Use a clear, front-facing image with a visible face.",
    )
    if uploaded_file is not None:
        try:
            uploaded_image = Image.open(uploaded_file).convert("RGB")
        except Exception as exc:
            st.error(f"Could not open image: {exc}")
else:
    camera_file = st.camera_input(
        "Take a photo",
        help="Allow browser camera access when prompted.",
    )
    if camera_file is not None:
        try:
            uploaded_image = Image.open(camera_file).convert("RGB")
        except Exception as exc:
            st.error(f"Could not read camera photo: {exc}")

if uploaded_image is not None:
    rgb_image = np.array(uploaded_image)
    left, right = st.columns(2)

    with left:
        st.subheader("Input image")
        st.image(rgb_image, use_container_width=True)

    if st.button("Detect emotion", type="primary"):
        with st.spinner("Detecting faces and predicting emotion..."):
            try:
                annotated_image, results = find_faces_and_predict(rgb_image, model, labels)
                with right:
                    st.subheader("Prediction")
                    st.image(annotated_image, use_container_width=True)

                if results:
                    st.success(f"Detected {len(results)} face(s).")
                    st.dataframe(results, use_container_width=True, hide_index=True)
                else:
                    st.warning(
                        "No face detected. Try a clear, front-facing photo with good lighting."
                    )
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
else:
    st.info("Choose an image or take a camera photo to begin.")

st.divider()
st.caption("Note: This app processes a still image/photo per prediction; it is not continuous live video.")
