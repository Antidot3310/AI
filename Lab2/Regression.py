import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

train_df = pd.read_csv("train_processed.csv")
test_df = pd.read_csv("test_processed.csv")

print("Train dimensionality:", train_df.shape)
print("Test dimensionality:", test_df.shape)

# предсказание Age
X_train_reg = train_df.drop("Age", axis=1)
y_train_reg = train_df["Age"]

X_test_reg = test_df.drop("Age", axis=1)
y_test_reg = test_df["Age"]


lr = LinearRegression()
lr.fit(X_train_reg, y_train_reg)
y_pred_reg = lr.predict(X_test_reg)

print("Linear Regression:")
print(f"MSE: {mean_squared_error(y_test_reg, y_pred_reg):.6f}")
print(f"MAE: {mean_absolute_error(y_test_reg, y_pred_reg):.6f}\n\n")

# Улучшение
ridge = Ridge(alpha=1)
ridge.fit(X_train_reg, y_train_reg)
y_pred_ridge = ridge.predict(X_test_reg)

print("Ridge Regression:")
print(f"MSE: {mean_squared_error(y_test_reg, y_pred_ridge):.6f}")
print(f"MAE: {mean_absolute_error(y_test_reg, y_pred_ridge):.6f}")


rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(X_train_reg, y_train_reg)
y_pred_rf = rf_reg.predict(X_test_reg)

print("RandomForest Regression:")
print(f"MSE: {mean_squared_error(y_test_reg, y_pred_rf):.6f}")
print(f"MAE: {mean_absolute_error(y_test_reg, y_pred_rf):.6f}")
