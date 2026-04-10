import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau

# 📁 Dataset paths
train_dir = "train"
test_dir = "test"

# 🎯 Settings
img_size = (96, 96)
batch_size = 32

# 🔄 Data Augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    zoom_range=0.25,
    horizontal_flip=True,
    validation_split=0.2
)

# 📦 Train data
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

print("Classes:", train_data.class_indices)

# 🧠 Base Model
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(96, 96, 3)
)

# 🔥 Fine-tuning (improved)
base_model.trainable = True

# freeze early layers only
for layer in base_model.layers[:75]:
    layer.trainable = False

# 🔧 Head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.4)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)

output = Dense(train_data.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

# ⚙ Optimizer (low LR for stability)
model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 📉 Learning rate scheduler
lr_reduce = ReduceLROnPlateau(
    monitor='val_accuracy',
    factor=0.5,
    patience=2,
    min_lr=1e-7,
    verbose=1
)

# 🚀 Train
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=25,
    callbacks=[lr_reduce]
)

# 💾 Save modern format
model.save("model.keras")

print("✅ Training completed successfully!")