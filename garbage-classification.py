# %% [markdown]
# # Proyek Klasifikasi Gambar: [Garbage Classification](https://www.kaggle.com/datasets/mostafaabla/garbage-classification/data)
# - **Nama:** Krisna Santosa
# - **Email:** mamang.krisna15@gmail.com
# - **ID Dicoding:** [krisna_santosa](https://www.dicoding.com/users/krisna_santosa/)
#
# ## Deskripsi Proyek
#
# Sampah merupakan masalah yang sering dihadapi oleh masyarakat di berbagai negara. Sampah yang tidak dikelola dengan baik akan menimbulkan masalah lingkungan yang serius. Oleh karena itu, diperlukan suatu sistem yang dapat mengelompokkan sampah menjadi beberapa kategori agar dapat dikelola dengan baik. Pada proyek ini, akan dilakukan klasifikasi sampah menjadi 12 kategori berbeda menggunakan teknik deep learning. Dataset yang digunakan adalah dataset [Garbage Classification](https://www.kaggle.com/mostafaabla/garbage-classification) yang terdiri dari 12 kategori sampah berbeda seperti kertas, kardus, botol, kaleng, gelas, dan lain-lain. Jumlah total gambar pada dataset ini adalah 15.000+ gambar dengan resolusi yang bervariasi.

# %% [markdown]
# ## Import Semua Packages/Library yang Digunakan

# %%
from sklearn.model_selection import train_test_split
import shutil
from PIL import Image
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, GlobalAveragePooling2D, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import random
from pathlib import Path
import pandas as pd
import os
from typing import Dict, List, Union
import requests
import logging

# %%
# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)

# %% [markdown]
# ## Data Preparation

# %% [markdown]
# ### Data Loading

# %%
data_dir = '/kaggle/input/garbage-classification/garbage_classification'

# %%
# Exploration


def print_images_resolution(directory):
    unique_sizes = set()
    total_images = 0

    for subdir in os.listdir(directory):
        subdir_path = os.path.join(directory, subdir)
        image_files = os.listdir(subdir_path)
        num_images = len(image_files)
        print(f"{subdir}: {num_images}")
        total_images += num_images

        for img_file in image_files:
            img_path = os.path.join(subdir_path, img_file)
            with Image.open(img_path) as img:
                unique_sizes.add(img.size)

        for size in unique_sizes:
            print(f"- {size}")
        print("---------------")

    print(f"\nTotal: {total_images}")


# %%
print_images_resolution(data_dir)

# %%
classes = os.listdir(data_dir)
print(f"Classes: {classes}")

for class_name in classes:
    num_images = len(os.listdir(os.path.join(data_dir, class_name)))
    print(f"{class_name}: {num_images} images")


# %%
image_dir = Path(data_dir)

# Get filepaths and labels
filepaths = list(image_dir.glob(r'**/*.JPG')) + list(image_dir.glob(r'**/*.jpg')) + \
    list(image_dir.glob(r'**/*.png')) + list(image_dir.glob(r'**/*.PNG'))

labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], filepaths))

filepaths = pd.Series(filepaths, name='Filepath').astype(str)
labels = pd.Series(labels, name='Label')

# Concatenate filepaths and labels
image_df = pd.concat([filepaths, labels], axis=1)

# %%
# Display 16 picture of the dataset with their labels
random_index = np.random.randint(0, len(image_df), 20)
fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(10, 10),
                         subplot_kw={'xticks': [], 'yticks': []})

for i, ax in enumerate(axes.flat):
    ax.imshow(plt.imread(image_df.Filepath[random_index[i]]))
    ax.set_title(image_df.Label[random_index[i]])
plt.tight_layout()
plt.show()

# %%
# Constants
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNEL = 3
BATCH_SIZE = 64
EPOCHS = 20

# %% [markdown]
# ### Data Preprocessing

# %% [markdown]
# #### Split Dataset

# %%

base_dir = "/kaggle/working"
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")
test_dir = os.path.join(base_dir, "test")

for folder in [train_dir, val_dir, test_dir]:
    os.makedirs(folder, exist_ok=True)

# Divide for each class
for class_name in classes:
    class_path = os.path.join(data_dir, class_name)
    images = np.array(os.listdir(class_path))

    # Split data: 80% train, 10% val, 10% test
    train_images, temp_images = train_test_split(
        images, test_size=0.2, random_state=42)
    val_images, test_images = train_test_split(
        temp_images, test_size=0.5, random_state=42)

    for split_name, split_images in zip(["train", "val", "test"], [train_images, val_images, test_images]):
        split_class_dir = os.path.join(base_dir, split_name, class_name)
        os.makedirs(split_class_dir, exist_ok=True)

        for img in split_images:
            src_path = os.path.join(class_path, img)
            dst_path = os.path.join(split_class_dir, img)
            shutil.copy(src_path, dst_path)

print("Dataset successfully divided into train, validation and test set!")


# %%
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

validation_generator = val_test_datagen.flow_from_directory(
    val_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

test_generator = val_test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

print("Data Loaded Successfully!")


# %% [markdown]
# ## Modelling

# %%
base_model = MobileNetV2(input_shape=(
    IMG_HEIGHT, IMG_WIDTH, IMG_CHANNEL), include_top=False, weights='imagenet')
base_model.trainable = False

for layer in base_model.layers[:20]:
    layer.trainable = True

# Sequential Model
with tf.device('/device:GPU:0'):
    model = Sequential([
        base_model,
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(2, 2),
        GlobalAveragePooling2D(),
        Dropout(0.3),
        Dense(len(train_generator.class_indices), activation='softmax')
    ])


initial_learning_rate = 1e-3
optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)

model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# %%
# Config Callbaks
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.1,
    patience=2,
    min_lr=1e-6,
    verbose=1
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    'best_model.keras',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

# %%
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks=[early_stopping, reduce_lr, checkpoint]
)

# %% [markdown]
# ## Evaluasi dan Visualisasi

# %%
# Evaluate the model
evaluation = model.evaluate(
    test_generator,
    steps=test_generator.samples // test_generator.batch_size
)

print("Loss:", evaluation[0])
print("Accuracy:", evaluation[1])

# %%
# Plot training history
plt.figure(figsize=(12, 4))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# %%
# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Konversi Model

# %%
# Save model in SavedModel format
tf.saved_model.save(model, 'saved_model')

# %%
# TF Lite conversion

converter = tf.lite.TFLiteConverter.from_saved_model('saved_model')

converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]

tflite_model = converter.convert()

with tf.io.gfile.GFile('tflite/model.tflite', 'wb') as f:
    f.write(tflite_model)


# %%
# Labels for TF Lite
LABEL_MAP = {
    0: 'battery',
    1: 'biological',
    2: 'brown-glass',
    3: 'cardboard',
    4: 'clothes',
    5: 'green-glass',
    6: 'metal',
    7: 'paper',
    8: 'plastic',
    9: 'shoes',
    10: 'trash',
    11: 'white-glass'
}

with open("tflite/labels.txt", "w") as f:
    for i in range(len(LABEL_MAP)):
        f.write(f"{LABEL_MAP[i]}\n")


# %%
!pip install tensorflowjs

# %%
# Convert to TFJS
!tensorflowjs_converter \
    - -input_format = tf_saved_model \
    / kaggle/working/saved_model / \
    /kaggle/working/tfjs_model

# %% [markdown]
# ## Inference (Optional)

# %%
# Logging Config
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Label Constants
LABEL_MAP = {
    0: 'battery',
    1: 'biological',
    2: 'brown-glass',
    3: 'cardboard',
    4: 'clothes',
    5: 'green-glass',
    6: 'metal',
    7: 'paper',
    8: 'plastic',
    9: 'shoes',
    10: 'trash',
    11: 'white-glass'
}


def load_and_preprocess_image(file_path: Union[str, Path], target_size: tuple = (224, 224)) -> List[List[List[List[float]]]]:
    """
    Load and preprocess an image for classification.

    Args:
        file_path: Path to the image file
        target_size: Tuple of (height, width) for resizing

    Returns:
        Preprocessed image tensor as a nested list
    """
    try:
        file_path = Path(file_path)
        if not file_path.is_file():
            raise FileNotFoundError(f"Image file not found: {file_path}")

        with open(file_path, 'rb') as f:
            image = tf.io.decode_image(f.read(), channels=3)

        image = tf.image.resize(image, target_size)
        image = image / 255.0
        image_tensor = tf.expand_dims(image, 0)

        return image_tensor.numpy().tolist()

    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise


def predict_image(image_tensor: List[List[List[List[float]]]], endpoint: str) -> Dict[str, Union[str, float, List[Dict[str, float]]]]:
    """
    Send image to classification endpoint and get predictions.

    Args:
        image_tensor: Preprocessed image tensor
        endpoint: URL of the classification API endpoint

    Returns:
        Dictionary containing:
        - confidence_scores: Raw prediction scores
        - predicted_class: The class with highest confidence
        - all_predictions: List of all classes with their confidence scores
    """
    try:
        request_data = {"instances": image_tensor}
        response = requests.post(endpoint, json=request_data)
        response.raise_for_status()

        predictions = response.json()['predictions'][0]
        predicted_class_idx = tf.argmax(predictions).numpy()

        all_predictions = [
            {
                'label': LABEL_MAP[idx],
                'confidence': float(score)
            }
            for idx, score in enumerate(predictions)
        ]

        # Sort predictions by confidence
        all_predictions.sort(key=lambda x: x['confidence'], reverse=True)

        return {
            'confidence_scores': predictions,
            'predicted_class': LABEL_MAP[predicted_class_idx],
            'all_predictions': all_predictions
        }

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise


def process_and_predict(image_path: str, endpoint: str) -> Dict[str, Union[str, float, List[Dict[str, float]]]]:
    """
    Process an image and get predictions in one step.

    Args:
        image_path: Path to the image file
        endpoint: URL of the classification API endpoint

    Returns:
        Dictionary containing prediction results and confidence scores for all classes
    """
    logger.info(f"Processing image: {image_path}")
    image_tensor = load_and_preprocess_image(image_path)

    logger.info("Getting predictions...")
    result = predict_image(image_tensor, endpoint)

    logger.info(f"Predicted class: {result['predicted_class']}")
    return result


# %%
image_path = 'metal118.jpg'
endpoint = "http://localhost:8501/v1/models/garbage_classifier:predict"

try:
    result = process_and_predict(image_path, endpoint)
    print(f"\nTop prediction: {result['predicted_class']}")
    print("\nAll predictions sorted by confidence:")
    for pred in result['all_predictions']:
        print(f"{pred['label']}: {pred['confidence']:.6f}")
    print(f"\nAll Confidence Score:\n{result['confidence_scores']}")
except Exception as e:
    logger.error(f"Process failed: {str(e)}")
