import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import (
    load_img,
    img_to_array,
)

from config import IMG_SIZE, BATCH_SIZE, TRAIN_DIR, VAL_DIR


def create_datasets():
    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        TRAIN_DIR,
        label_mode="categorical",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=42,
    )

    def augment(image, label):
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
        image = tf.image.random_flip_left_right(image)
        image = tf.image.rot90(
            image, tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
        )
        return image, label

    train_dataset = train_dataset.map(augment, num_parallel_calls=1)
    train_dataset = train_dataset.prefetch(1)

    val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        VAL_DIR,
        label_mode="categorical",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    val_dataset = val_dataset.cache().prefetch(1)

    return train_dataset, val_dataset


def load_and_prepare_image(image_path):
    """Загружает изображение, приводит к нужному размеру."""
    img = load_img(image_path, target_size=IMG_SIZE)
    img_array = img_to_array(img)
    return np.expand_dims(img_array, axis=0)
