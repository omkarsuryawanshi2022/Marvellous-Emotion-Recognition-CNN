"""
Real-Time Emotion Detection using CNN + OpenCV (NO STREAMLIT)

Dataset supported:
1) Standard FER2013 CSV: fer2013/fer2013.csv with columns emotion and pixels.
2) Image folders:
   fer2013/train/angry, disgust, fear, happy, sad, surprise, neutral
   (test/ folders are optional; this script creates a stratified split.)

Run from Windows CMD:
  pip install tensorflow opencv-python numpy pandas scikit-learn matplotlib
  python Marvellous_Real_Time_Emotion_Detection.py --mode train --dataset fer2013
  python Marvellous_Real_Time_Emotion_Detection.py --mode webcam
  python Marvellous_Real_Time_Emotion_Detection.py --mode all --dataset fer2013

Press Q or ESC to close the webcam window.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix, classification_report
except ImportError:
    train_test_split = None
    confusion_matrix = None
    classification_report = None

IMAGE_SIZE = 48
BATCH_SIZE = 64
DEFAULT_EPOCHS = 10
MODEL_PATH = "Emotion_Detection_CNN.keras"
LABELS_PATH = "emotion_labels.txt"

# Standard FER2013 numeric mapping.
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
ALIASES = {
    "angry": 0, "anger": 0,
    "disgust": 1, "disgusted": 1,
    "fear": 2, "fearful": 2,
    "happy": 3, "happiness": 3,
    "sad": 4, "sadness": 4,
    "surprise": 5, "surprised": 5,
    "neutral": 6,
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def build_cnn():
    """CNN architecture for 48x48 grayscale facial-expression images."""
    model = models.Sequential([
        layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1)),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),

        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),

        layers.Conv2D(128, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(128, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Dropout(0.30),

        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.50),
        layers.Dense(len(EMOTIONS), activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def read_csv_dataset(csv_path):
    if pd is None:
        raise ImportError("CSV dataset needs pandas. Run: pip install pandas")
    df = pd.read_csv(csv_path)
    if not {"emotion", "pixels"}.issubset(df.columns):
        raise ValueError("CSV must contain 'emotion' and 'pixels' columns.")

    images, labels = [], []
    skipped = 0
    for row in df[["emotion", "pixels"]].itertuples(index=False, name=None):
        try:
            label = int(row[0])
            pixels = np.fromstring(str(row[1]), sep=" ", dtype=np.float32)
            if 0 <= label < 7 and pixels.size == IMAGE_SIZE * IMAGE_SIZE:
                images.append(pixels.reshape(IMAGE_SIZE, IMAGE_SIZE, 1) / 255.0)
                labels.append(label)
            else:
                skipped += 1
        except (ValueError, TypeError):
            skipped += 1

    if not images:
        raise ValueError("No valid FER2013 images were read from the CSV.")
    print(f"Read {len(images):,} images from {csv_path}; skipped {skipped} invalid rows.")
    return np.asarray(images, dtype=np.float32), np.asarray(labels, dtype=np.int64)


def read_folder_dataset(dataset_dir):
    images, labels = [], []
    # Support class folders directly under root and under train/training/test/testing.
    possible_parents = [dataset_dir, dataset_dir / "train", dataset_dir / "training",
                        dataset_dir / "test", dataset_dir / "testing"]
    seen_class_folders = set()

    for parent in possible_parents:
        if not parent.is_dir():
            continue
        for class_dir in parent.iterdir():
            if not class_dir.is_dir() or class_dir.name.lower() not in ALIASES:
                continue
            resolved = str(class_dir.resolve())
            if resolved in seen_class_folders:
                continue
            seen_class_folders.add(resolved)
            label = ALIASES[class_dir.name.lower()]
            for file_path in class_dir.rglob("*"):
                if file_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                image = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
                if image is None:
                    continue
                image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
                images.append(image.astype(np.float32) / 255.0)
                labels.append(label)

    if not images:
        raise ValueError(
            "No images found. Expected fer2013.csv or folders such as "
            "fer2013/train/angry, fer2013/train/happy, etc."
        )
    print(f"Read {len(images):,} images from image folders.")
    return (np.asarray(images, dtype=np.float32).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1),
            np.asarray(labels, dtype=np.int64))


def load_dataset(dataset_path):
    root = Path(dataset_path).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"Dataset folder not found: {root.resolve()}")
    # Look for standard CSV in root or one level below.
    csv_candidates = [root / "fer2013.csv", root / "FER2013.csv"]
    csv_candidates.extend(root.glob("*.csv"))
    csv_path = next((p for p in csv_candidates if p.is_file()), None)
    if csv_path:
        return read_csv_dataset(csv_path)
    return read_folder_dataset(root)


def train_model(dataset_path, model_path=MODEL_PATH, epochs=DEFAULT_EPOCHS):
    np.random.seed(42)
    tf.random.set_seed(42)
    x, y = load_dataset(dataset_path)
    if len(np.unique(y)) < 2:
        raise ValueError("Training requires at least two emotion classes.")

    if train_test_split is not None:
        x_train, x_temp, y_train, y_temp = train_test_split(
            x, y, test_size=0.20, random_state=42, stratify=y
        )
        x_val, x_test, y_val, y_test = train_test_split(
            x_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
        )
    else:
        order = np.random.permutation(len(x))
        x, y = x[order], y[order]
        a, b = int(len(x) * 0.8), int(len(x) * 0.9)
        x_train, y_train = x[:a], y[:a]
        x_val, y_val = x[a:b], y[a:b]
        x_test, y_test = x[b:], y[b:]

    print(f"Training: {len(x_train):,} | Validation: {len(x_val):,} | Test: {len(x_test):,}")
    model = build_cnn()
    model.summary()
    cb = [
        callbacks.ModelCheckpoint(model_path, monitor="val_accuracy",
                                  save_best_only=True, mode="max", verbose=1),
        callbacks.EarlyStopping(monitor="val_loss", patience=7,
                                restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                    patience=3, min_lr=1e-6, verbose=1),
    ]

    history = model.fit(
        x_train, y_train, validation_data=(x_val, y_val),
        epochs=epochs, batch_size=BATCH_SIZE, callbacks=cb, verbose=1
    )
    model.save(model_path)
    Path(LABELS_PATH).write_text("\n".join(EMOTIONS) + "\n", encoding="utf-8")
    print(f"Saved model: {Path(model_path).resolve()}")

    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test loss: {loss:.4f}")
    print(f"Test accuracy: {accuracy * 100:.2f}%")
    predicted = np.argmax(model.predict(x_test, batch_size=BATCH_SIZE, verbose=0), axis=1)
    if confusion_matrix is not None:
        print("\nConfusion matrix (label order 0..6):")
        print(confusion_matrix(y_test, predicted, labels=list(range(7))))
    if classification_report is not None:
        print("\nClassification report:")
        print(classification_report(
            y_test, predicted, labels=list(range(7)),
            target_names=EMOTIONS, zero_division=0
        ))

    try:
        import matplotlib.pyplot as plt
        for metric, title, filename in [
            ("accuracy", "Training and Validation Accuracy", "emotion_training_accuracy.png"),
            ("loss", "Training and Validation Loss", "emotion_training_loss.png"),
        ]:
            plt.figure(figsize=(9, 5))
            plt.plot(history.history.get(metric, []), label="Training")
            plt.plot(history.history.get("val_" + metric, []), label="Validation")
            plt.xlabel("Epoch")
            plt.ylabel(metric.capitalize())
            plt.title(title)
            plt.legend()
            plt.tight_layout()
            plt.savefig(filename, dpi=150)
            plt.close()
        print("Saved training graphs: emotion_training_accuracy.png and emotion_training_loss.png")
    except ImportError:
        print("Matplotlib unavailable; skipped training graphs.")
    return model


def load_trained_model(model_path):
    model_file = Path(model_path)
    if not model_file.is_file():
        raise FileNotFoundError(
            f"Model not found: {model_file.resolve()}\n"
            "Train first with --mode train --dataset fer2013"
        )
    model = tf.keras.models.load_model(model_file)
    label_file = Path(LABELS_PATH)
    labels = ([s.strip() for s in label_file.read_text(encoding="utf-8").splitlines() if s.strip()]
              if label_file.exists() else EMOTIONS)
    return model, labels


def run_webcam(model_path=MODEL_PATH, camera_index=0):
    model, labels = load_trained_model(model_path)
    cascade_file = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_detector = cv2.CascadeClassifier(cascade_file)
    if face_detector.empty():
        raise RuntimeError("OpenCV Haar face detector could not be loaded.")

    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        raise RuntimeError(f"Cannot open camera index {camera_index}; try --camera 1.")

    print("Webcam running. Press Q or ESC to exit.")
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("Could not read webcam frame.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(
                gray, scaleFactor=1.2, minNeighbors=5, minSize=(40, 40)
            )
            for (x, y, w, h) in faces:
                face = cv2.resize(gray[y:y+h, x:x+w], (IMAGE_SIZE, IMAGE_SIZE))
                face = (face.astype(np.float32) / 255.0).reshape(1, IMAGE_SIZE, IMAGE_SIZE, 1)
                probabilities = model.predict(face, verbose=0)[0]
                idx = int(np.argmax(probabilities))
                emotion = labels[idx] if idx < len(labels) else EMOTIONS[idx]
                confidence = float(probabilities[idx]) * 100
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f"{emotion}: {confidence:.1f}%",
                            (x, max(y-10, 25)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.putText(frame, "Q / ESC: Quit", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("Real-Time Emotion Detection - CNN + OpenCV", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="FER2013 real-time emotion recognition (no Streamlit).")
    parser.add_argument("--mode", choices=["train", "webcam", "all"], default="webcam")
    parser.add_argument("--dataset", default="fer2013", help="Path to extracted FER2013 dataset")
    parser.add_argument("--model", default=MODEL_PATH, help="Path to saved .keras model")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index")
    args = parser.parse_args()
    try:
        if args.mode in ("train", "all"):
            train_model(args.dataset, args.model, args.epochs)
        if args.mode in ("webcam", "all"):
            run_webcam(args.model, args.camera)
    except (FileNotFoundError, ValueError, ImportError, RuntimeError) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
