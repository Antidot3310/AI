import numpy as np
import keras
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

X = np.loadtxt("dataIn.txt").T
y = np.loadtxt("dataOut.txt").T

print("Форма X:", X.shape)
print("Форма y:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = keras.Sequential(
    [
        layers.Dense(64, activation="sigmoid", input_shape=(12,)),  # скрытый слой
        layers.Dense(2, activation="softmax"),  # выходной слой
    ]
)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
history = model.fit(
    X_train,
    y_train,
    epochs=150,
    batch_size=16,
    validation_data=(X_test, y_test),
    verbose=1,
)

# Предсказание классов
y_pred_proba = model.predict(X_test)
y_pred = np.argmax(y_pred_proba, axis=1)
y_true = np.argmax(y_test, axis=1)

accuracy = accuracy_score(y_true, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")

print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))
print("\nClassification Report:")
print(
    classification_report(y_true, y_pred, target_names=["Правящая партия", "Оппозиция"])
)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss over epochs")

plt.subplot(1, 2, 2)
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Val Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Accuracy over epochs")

plt.tight_layout()
plt.show()

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Логистическая регрессия
lr = LogisticRegression()
lr.fit(X_train, np.argmax(y_train, axis=1))
y_pred_lr = lr.predict(X_test)
print("Logistic Regression Accuracy:", accuracy_score(y_true, y_pred_lr))

# Случайный лес
rf = RandomForestClassifier(n_estimators=50)
rf.fit(X_train, np.argmax(y_train, axis=1))
y_pred_rf = rf.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_true, y_pred_rf))
