import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# 📁 Dataset paths
train_dir = "train"
test_dir = "test"

# 🎯 Image settings
img_size = (96, 96)
batch_size = 32

# 🔄 Data Augmentation + Validation Split
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

# 📦 Training data
train_data = datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

# 📦 Validation data
val_data = datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

print("Classes found:", train_data.class_indices)

# 🧠 Base Model (MobileNetV2)
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(96, 96, 3)
)

# 🔥 IMPORTANT: Fine-tuning enabled
base_model.trainable = True

# ❄ Freeze first 100 layers (important for stability)
for layer in base_model.layers[:100]:
    layer.trainable = False

# 🔧 Custom Classification Head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(train_data.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

# ⚙ Compile (LOW learning rate for fine-tuning)
model.compile(
    optimizer=Adam(learning_rate=0.00001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 🚀 Train Model
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=20
)

# 💾 Save Model (recommended format)
model.save("model.hdf5")

print("✅ Training completed and improved model saved!")