from keras import Sequential
from keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
)
from keras.applications import MobileNetV2
from keras.models import Model
from config import IMG_SIZE, NUM_CLASSES


def create_cnn_model():
    """Собственная нейросеть."""
    model = Sequential(
        [
            Conv2D(32, (3, 3), activation="relu", input_shape=(*IMG_SIZE, 3)),
            MaxPooling2D(2, 2),
            Conv2D(64, (3, 3), activation="relu"),
            MaxPooling2D(2, 2),
            Conv2D(128, (3, 3), activation="relu"),
            MaxPooling2D(2, 2),
            Flatten(),
            Dense(256, activation="relu"),
            Dropout(0.5),
            Dense(NUM_CLASSES, activation="softmax"),
        ]
    )
    return model


def create_mobilenet_model():
    """Модель на основе предобученного MobileNetV2."""
    base_model = MobileNetV2(
        weights="imagenet", include_top=False, input_shape=(*IMG_SIZE, 3)
    )
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)
    predictions = Dense(NUM_CLASSES, activation="softmax", dtype="float32")(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    return model
