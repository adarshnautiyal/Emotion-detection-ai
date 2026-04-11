from flask import Flask, render_template, request, jsonify
import numpy as np
import cv2
import random
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load model
model = load_model("model.hdf5")

# Face detector (optional use)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

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


# 🔥 IMAGE UPLOAD PREDICTION (VERCEL SAFE)
@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]

    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    # Resize to model input size
    img = cv2.resize(img, (96, 96))
    img = img.astype("float32") / 255.0
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


if __name__ == "__main__":
    app.run(debug=True)