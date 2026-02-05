import tensorflow as tf
from tensorflow.keras import layers, models

# Image parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Dataset paths
train_dir = "dataset/train"
val_dir = "dataset/val"

# Load datasets (grayscale for wafer images)
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="grayscale"
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="grayscale"
)

num_classes = len(train_ds.class_names)

# CNN model
model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(224,224,1)),

    layers.Conv2D(32,3,activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(64,3,activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(128,3,activation='relu'),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(128,activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes,activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("Training model...")

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20
)

# Save model
model.save("simple_cnn.h5")
print("Model saved")

