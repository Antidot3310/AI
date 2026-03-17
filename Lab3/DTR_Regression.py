import pandas as pd
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.metrics import mean_squared_error, mean_absolute_error

train_df = pd.read_csv("train_processed.csv")
test_df = pd.read_csv("test_processed.csv")

X_train_reg = train_df.drop("Age", axis=1)
y_train_reg = train_df["Age"]

X_test_reg = test_df.drop("Age", axis=1)
y_test_reg = test_df["Age"]


dt_reg = DecisionTreeRegressor(ccp_alpha=0.0005, max_depth=5, random_state=42)
dt_reg.fit(X_train_reg, y_train_reg)

y_pred_reg = dt_reg.predict(X_test_reg)

print(export_text(dt_reg, feature_names=X_test_reg.columns.tolist()))
print("Decision Tree Regressor:")
print(f"MSE: {mean_squared_error(y_test_reg, y_pred_reg):.3f}")
print(f"MAE: {mean_absolute_error(y_test_reg, y_pred_reg):.3f}")

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(20, 10))
plot_tree(
    dt_reg,
    feature_names=X_train_reg.columns.tolist(),
    fontsize=10,
)
plt.title("Decision Tree Visualization (max_depth=5)")
plt.show()
