import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

train_df = pd.read_csv("train_processed.csv")
test_df = pd.read_csv("test_processed.csv")

X_train_clf = train_df.drop("Survived", axis=1)
y_train_clf = train_df["Survived"]

X_test_clf = test_df.drop("Survived", axis=1)
y_test_clf = test_df["Survived"]

dt_clf = DecisionTreeClassifier(ccp_alpha=0.005, max_depth=5, random_state=42)
dt_clf.fit(X_train_clf, y_train_clf)

y_pred_clf = dt_clf.predict(X_test_clf)

print(export_text(dt_clf, feature_names=X_test_clf.columns.tolist()))
print("Decision Tree Classifier:")
print("Accuracy:", accuracy_score(y_test_clf, y_pred_clf))
print("Confusion Matrix:\n", confusion_matrix(y_test_clf, y_pred_clf))
print("Classification Report:\n", classification_report(y_test_clf, y_pred_clf))

from sklearn.metrics import roc_curve, auc

y_proba = dt_clf.predict_proba(X_test_clf)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test_clf, y_proba)
roc_auc = auc(fpr, tpr)
print(f"ROC-AUC: {roc_auc:.3f}")


plt.figure()
plt.plot(fpr, tpr, color="orange", label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color="navy")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()

from sklearn.tree import plot_tree

plt.figure(figsize=(20, 10))
plot_tree(
    dt_clf,
    feature_names=X_train_clf.columns.tolist(),
    fontsize=10,
)
plt.title("Decision Tree Visualization (max_depth=5)")
plt.show()
