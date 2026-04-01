import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "dataset")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
MODELS_DIR = os.path.join(BASE_DIR, "trained_models")

IMG_SIZE = (224, 224)
BATCH_SIZE = 256
EPOCHS_CNN = 10
EPOCHS_MOBILENET = 5
NUM_CLASSES = 3

CLASS_NAMES = ["like", "palm", "spiderman"]
