import os
os.environ["KERAS_BACKEND"] = "torch"
import cv2
import tempfile
import h5py
from collections import deque
from keras.models import load_model

MODEL_CANDIDATES = ("model.keras", "model.hdf5")


def load_emotion_model():
    for model_path in MODEL_CANDIDATES:
        if os.path.exists(model_path):
            try:
                return load_model(model_path)
            except TypeError as exc:
                if "batch_shape" not in str(exc):
                    raise
                with h5py.File(model_path, "r") as source:
                    model_config = source.attrs.get("model_config")
                    if model_config is None:
                        raise
                    if isinstance(model_config, bytes):
                        model_config = model_config.decode("utf-8")
                    patched = model_config.replace('"batch_shape"', '"batch_input_shape"')
                    with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=False) as tmp:
                        tmp_path = tmp.name
                    try:
                        with h5py.File(tmp_path, "w") as target:
                            for key, value in source.attrs.items():
                                if key == "model_config":
                                    target.attrs[key] = patched
                                else:
                                    target.attrs[key] = value
                            for name in source.keys():
                                source.copy(name, target)
                        return load_model(tmp_path)
                    finally:
                        if os.path.exists(tmp_path):
                            os.remove(tmp_path)
    raise FileNotFoundError(
        "No model file found. Expected one of: " + ", ".join(MODEL_CANDIDATES)
    )


model = load_emotion_model()

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
