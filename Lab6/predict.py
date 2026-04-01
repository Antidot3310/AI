import os
import glob
import tensorflow as tf

from config import MODELS_DIR, CLASS_NAMES
from utils import load_and_prepare_image


def predict_on_folder(model, folder_path):
    """Предсказывает классы для всех изображений в папке и выводит результат."""
    image_paths = glob.glob(os.path.join(folder_path, "*.jpg"))

    print(f"\nНайдено {len(image_paths)} изображений. Обработка...\n")
    print("-" * 60)
    print(f"{'Файл':<40} {'Предсказание':<20} {'Уверенность':<10}")
    print("-" * 60)

    for img_path in image_paths:
        img_array = load_and_prepare_image(img_path)
        predictions = model.predict(img_array, verbose=0)
        pred_class_idx = tf.argmax(predictions[0]).numpy()
        confidence = predictions[0][pred_class_idx]
        pred_class = CLASS_NAMES[pred_class_idx]

        filename = os.path.basename(img_path)
        print(f"{filename:<40} {pred_class:<20} {confidence:.4f}")
    print("-" * 60)


def main():
    test_dir = "test"
    model_path = os.path.join(MODELS_DIR, "gesture_cnn.h5")
    model = tf.keras.models.load_model(model_path)
    predict_on_folder(model, test_dir)


if __name__ == "__main__":
    main()
