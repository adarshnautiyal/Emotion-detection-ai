from flask import Flask, render_template, request
import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load model
model = load_model('model.hdf5', compile=False)

# Load Haar Cascade
cascade_path = os.path.join(os.getcwd(), 'haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier(cascade_path)

# Check cascade
if face_cascade.empty():
    print("❌ Error: Haarcascade file not loaded!")

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return render_template('index.html', prediction="No file uploaded")

    file = request.files['image']

    if file.filename == '':
        return render_template('index.html', prediction="No file selected")

    try:
        # Convert file to OpenCV image
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img is None:
            return render_template('index.html', prediction="Invalid image")

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # If no face found
        if len(faces) == 0:
            return render_template('index.html', prediction="No face detected")

        # Pick largest face
        face = max(faces, key=lambda rect: rect[2] * rect[3])
        (x, y, w, h) = face

        # Crop face
        face_img = gray[y:y+h, x:x+w]

        # Resize to 64x64 (IMPORTANT)
        face_img = cv2.resize(face_img, (64, 64))

        # Improve contrast
        face_img = cv2.equalizeHist(face_img)

        # Normalize
        face_img = face_img / 255.0

        # Reshape (IMPORTANT)
        face_img = face_img.reshape(1, 64, 64, 1)

        # Predict
        prediction = model.predict(face_img)
        confidence = np.max(prediction) * 100
        emotion = emotion_labels[np.argmax(prediction)]

        result = f"{emotion} ({confidence:.2f}%)"

        return render_template('index.html', prediction=result)

    except Exception as e:
        return render_template('index.html', prediction=f"Error: {str(e)}")

# Run app
if __name__ == '__main__':
    app.run(debug=True)