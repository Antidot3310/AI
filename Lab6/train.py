import os
import tensorflow as tf

from config import MODELS_DIR, EPOCHS_CNN, EPOCHS_MOBILENET
from utils import create_datasets
from models import create_cnn_model, create_mobilenet_model

physical_devices = tf.config.list_physical_devices("GPU")
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)
tf.keras.mixed_precision.set_global_policy("mixed_float16")

def train_and_save():
    train_ds, val_ds = create_datasets()

    # ----- MobileNetV2 -----
    print("--- Обучение MobileNetV2 ---")
    mobilenet_model = create_mobilenet_model()
    mobilenet_model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    history_mobilenet = mobilenet_model.fit(
        train_ds, validation_data=val_ds, epochs=EPOCHS_MOBILENET, verbose=1
    )
    mobilenet_model.save(os.path.join(MODELS_DIR, "gesture_mobilenet.h5"))

    # ----- CNN -----
    print("\n--- Обучение собственной CNN ---")
    cnn_model = create_cnn_model()
    cnn_model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    history_cnn = cnn_model.fit(
        train_ds, validation_data=val_ds, epochs=EPOCHS_CNN, verbose=1
    )
    cnn_model.save(os.path.join(MODELS_DIR, "gesture_cnn.h5"))

    # Оценка
    loss_cnn, acc_cnn = cnn_model.evaluate(val_ds, verbose=0)
    loss_mob, acc_mob = mobilenet_model.evaluate(val_ds, verbose=0)
    print(f"\nТочность CNN: {acc_cnn*100:.2f}%")
    print(f"Точность MobileNetV2: {acc_mob*100:.2f}%")


if __name__ == "__main__":
    train_and_save()
