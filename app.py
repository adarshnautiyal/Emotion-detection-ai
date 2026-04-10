from flask import Flask, render_template, Response
import cv2
import numpy as np
import random
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load model
model = load_model("model.hdf5")

# Face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

camera = cv2.VideoCapture(0)

# ---------------- EMOTION DATA ---------------- #

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

# ---------------- GLOBAL VARIABLES ---------------- #

current_emotion = "Neutral"
current_confidence = 0.0
current_quote = ""
current_activity = ""

# ---------------- VIDEO STREAM ---------------- #

def generate_frames():
    global current_emotion, current_confidence, current_quote, current_activity

    while True:
        success, frame = camera.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]

            face = cv2.resize(face, (96, 96))
            face = face.astype("float32") / 255.0
            face = np.expand_dims(face, axis=0)

            prediction = model.predict(face, verbose=0)[0]
            index = np.argmax(prediction)

            current_emotion = labels[index]
            current_confidence = float(prediction[index]) * 100

            current_quote = random.choice(
                emotion_data_map[current_emotion]["quotes"]
            )
            current_activity = emotion_data_map[current_emotion]["activity"]

            # Draw rectangle + label
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame,
                        f"{current_emotion} {current_confidence:.1f}%",
                        (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# ---------------- ROUTES ---------------- #

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/emotion_data')
def emotion_data():
    return {
        "emotion": current_emotion,
        "confidence": round(current_confidence, 2),
        "quote": current_quote,
        "activity": current_activity
    }


# ---------------- RUN ---------------- #

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)