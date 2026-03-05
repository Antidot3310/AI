import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier

train_df = pd.read_csv("train_processed.csv")
test_df = pd.read_csv("test_processed.csv")

print("Train dimensionality:", train_df.shape)
print("Test dimensionality:", test_df.shape)

# предсказание Survived
X_train_clf = train_df.drop("Survived", axis=1)
y_train_clf = train_df["Survived"]

X_test_clf = test_df.drop("Survived", axis=1)
y_test_clf = test_df["Survived"]

logreg = LogisticRegression(random_state=42, class_weight="balanced")
logreg.fit(X_train_clf, y_train_clf)
y_pred_clf = logreg.predict(X_test_clf)

print("\nLogistic Regression:")
print(f"Accuracy: {accuracy_score(y_test_clf, y_pred_clf):.3f}")
print("Confusion Matrix:\n", confusion_matrix(y_test_clf, y_pred_clf))
print("Classification Report:\n", classification_report(y_test_clf, y_pred_clf))

# улучшение
rf_clf = RandomForestClassifier(
    n_estimators=100, random_state=42, class_weight="balanced"
)
rf_clf.fit(X_train_clf, y_train_clf)
y_pred_rf = rf_clf.predict(X_test_clf)

print("RandomForest Classifier:")
print("Accuracy:", accuracy_score(y_test_clf, y_pred_rf))
print("Confusion Matrix:\n", confusion_matrix(y_test_clf, y_pred_rf))
print("Classification Report:\n", classification_report(y_test_clf, y_pred_rf))
