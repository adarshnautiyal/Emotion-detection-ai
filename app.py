import os
import json
import h5py
import cv2
from flask import Flask, render_template, request, jsonify, Response
import numpy as np
from PIL import Image
import random
from tensorflow.keras.models import load_model, model_from_json
from collections import deque

app = Flask(__name__)

MODEL_CANDIDATES = ("model.keras", "model.hdf5")


# ✅ FIXED MODEL LOADER
def load_emotion_model():
    for model_path in MODEL_CANDIDATES:
        if os.path.exists(model_path):
            try:
                return load_model(model_path, compile=False)

            except ValueError as e:
                if "batch_shape" not in str(e):
                    raise

                print("⚠️ Fixing model compatibility...")

                with h5py.File(model_path, "r") as f:
                    model_config = f.attrs.get("model_config")

                    if isinstance(model_config, bytes):
                        model_config = model_config.decode("utf-8")

                    # 🔥 Fix old Keras format
                    model_config = model_config.replace("batch_shape", "batch_input_shape")

                    model = model_from_json(model_config)
                    model.load_weights(model_path)

                return model

    raise FileNotFoundError(
        "No model file found. Expected one of: " + ", ".join(MODEL_CANDIDATES)
    )


# ✅ LOAD MODEL
model = load_emotion_model()

labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']


# ✅ FACE DETECTION MODEL
net = cv2.dnn.readNetFromCaffe(
    "models/deploy.prototxt",
    "models/res10_300x300_ssd_iter_140000.caffemodel"
)

emotion_data_map = {
    "Angry": {
        "quotes": ["Calm mind brings strength.", "Pause before reacting.", "Anger fades with patience."],
        "activity": "Take deep breaths for 2 minutes."
    },
    "Disgust": {
        "quotes": ["Focus on what matters.", "Let go of negativity.", "Shift your attention."],
        "activity": "Listen to calming music."
    },
    "Fear": {
        "quotes": ["You are stronger than fear.", "Fear is temporary.", "Face it step by step."],
        "activity": "Try grounding technique (5-4-3-2-1)."
    },
    "Happy": {
        "quotes": ["Happiness is a choice.", "Smile and spread joy.", "Enjoy the moment."],
        "activity": "Share your happiness 😄"
    },
    "Neutral": {
        "quotes": ["Stay balanced.", "Peace begins within.", "Keep going steadily."],
        "activity": "Relax and breathe deeply."
    },
    "Sad": {
        "quotes": ["This too shall pass.", "Healing takes time.", "Better days are coming."],
        "activity": "Talk to someone you trust."
    },
    "Surprise": {
        "quotes": ["Life is full of surprises.", "Embrace the unknown.", "Stay curious."],
        "activity": "Reflect on the moment."
    }
}


@app.route("/")
def home():
    return render_template("index.html")


# ✅ IMAGE PREDICTION
@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("image")

    if file is None or not file.filename:
        return jsonify({"error": "Please upload an image file."}), 400

    try:
        img = Image.open(file.stream).convert("RGB")
    except Exception:
        return jsonify({"error": "Unable to read the uploaded image."}), 400

    img = img.resize((96, 96))
    img = np.asarray(img, dtype="float32") / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img, verbose=0)[0]
    index = np.argmax(prediction)

    emotion = labels[index]
    confidence = float(prediction[index]) * 100

    return jsonify({
        "emotion": emotion,
        "confidence": round(confidence, 2),
        "quote": random.choice(emotion_data_map[emotion]["quotes"]),
        "activity": emotion_data_map[emotion]["activity"]
    })


# ✅ VIDEO STREAM
def generate_frames():
    cap = cv2.VideoCapture(0)
    pred_buffer = deque(maxlen=10)

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)),
            1.0,
            (300, 300),
            (104.0, 177.0, 123.0)
        )

        net.setInput(blob)
        detections = net.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > 0.6:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")

                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                face = frame[y1:y2, x1:x2]

                if face.size > 0:
                    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    face_pil = Image.fromarray(face_rgb).resize((96, 96))

                    face_array = np.asarray(face_pil, dtype="float32") / 255.0
                    face_array = np.expand_dims(face_array, axis=0)

                    prediction = model.predict(face_array, verbose=0)[0]
                    pred_buffer.append(prediction)

                    avg_pred = np.mean(pred_buffer, axis=0)
                    idx = np.argmax(avg_pred)

                    emotion = labels[idx]
                    prob = avg_pred[idx]

                    color = (0, 255, 0) if prob > 0.5 else (0, 165, 255)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        frame,
                        f"{emotion} ({prob*100:.1f}%)",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2
                    )

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
        )

    cap.release()


@app.route('/video_feed')
def video_feed():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
