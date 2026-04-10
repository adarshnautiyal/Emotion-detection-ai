import cv2
import numpy as np
from collections import deque
from tensorflow.keras.models import load_model

model = load_model("model.hdf5")

labels = ['angry','disgust','fear','happy','neutral','sad','surprise']

pred_buffer = deque(maxlen=15)

cap = cv2.VideoCapture(0)

net = cv2.dnn.readNetFromCaffe(
    "models/deploy.prototxt",
    "models/res10_300x300_ssd_iter_140000.caffemodel"
)

while True:
    ret, frame = cap.read()
    if not ret:
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

        if confidence < 0.6:
            continue

        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        x1, y1, x2, y2 = box.astype("int")

        if x1 < 0 or y1 < 0:
            continue

        face = frame[y1:y2, x1:x2]

        if face.shape[0] < 80 or face.shape[1] < 80:
            continue

        face = cv2.resize(face, (96, 96))
        face = face.astype("float32") / 255.0
        face = np.expand_dims(face, axis=0)

        pred = model.predict(face, verbose=0)[0]

        pred_buffer.append(pred)
        avg_pred = np.mean(pred_buffer, axis=0)

        emotion = labels[np.argmax(avg_pred)]
        conf = np.max(avg_pred)

        color = (0,255,0) if conf > 0.55 else (0,0,255)

        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
        cv2.putText(frame,
                    f"{emotion} {conf:.2f}",
                    (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2)

    cv2.imshow("Emotion AI - LEVEL 3", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()