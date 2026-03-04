import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("train.csv")

# Первичный анализ
print("Первые 5 строк:")
print(df.head())
print("\nИнформация о датасете:")
print(df.info())
print("\nСтатистика числовых признаков:")
print(df.describe())

# Разделение на train / test
train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)

# Анализ пропусков
missing_train = train_df.isnull().sum()
print("\nПропуски в train выборке:")
print(missing_train[missing_train > 0])

# Заполнение пропусков (используем только train)
median_age = train_df["Age"].median()
mode_embarked = train_df["Embarked"].mode()[0]

train_df["Age"] = train_df["Age"].fillna(median_age)
train_df["Embarked"] = train_df["Embarked"].fillna(mode_embarked)
train_df["Cabin"] = train_df["Cabin"].fillna("U")  # временно

# Применяем к test
test_df["Age"] = test_df["Age"].fillna(median_age)
test_df["Embarked"] = test_df["Embarked"].fillna(mode_embarked)
test_df["Cabin"] = test_df["Cabin"].fillna("U")

# Проверка, что пропусков нет
print("\nПосле заполнения пропусков в train:")
print(train_df.isnull().sum())

#  Нормализация числовых признаков
num_cols = ["Age", "Fare", "SibSp", "Parch"]
scaler = MinMaxScaler()

train_df[num_cols] = scaler.fit_transform(train_df[num_cols])
test_df[num_cols] = scaler.transform(test_df[num_cols])

# Кодирование категориальных признаков

# Sex
train_df["Sex"] = train_df["Sex"].map({"male": 0, "female": 1})
test_df["Sex"] = test_df["Sex"].map({"male": 0, "female": 1})

# Embarked (OHE)
embarked_dummies_train = pd.get_dummies(
    train_df["Embarked"], prefix="Embarked", drop_first=True
)
train_df = pd.concat([train_df, embarked_dummies_train], axis=1)
train_df.drop("Embarked", axis=1, inplace=True)

# Для test то же самое
embarked_dummies_test = pd.get_dummies(
    test_df["Embarked"], prefix="Embarked", drop_first=True
)
test_df = pd.concat([test_df, embarked_dummies_test], axis=1)
test_df.drop("Embarked", axis=1, inplace=True)

# Добавляем в test отсутствующие столбцы
for col in embarked_dummies_train.columns:
    if col not in test_df.columns:
        test_df[col] = 0

# Pclass (аналогично)
pclass_dummies_train = pd.get_dummies(
    train_df["Pclass"], prefix="Pclass", drop_first=True
)
train_df = pd.concat([train_df, pclass_dummies_train], axis=1)
train_df.drop("Pclass", axis=1, inplace=True)

pclass_dummies_test = pd.get_dummies(
    test_df["Pclass"], prefix="Pclass", drop_first=True
)
test_df = pd.concat([test_df, pclass_dummies_test], axis=1)
test_df.drop("Pclass", axis=1, inplace=True)

for col in pclass_dummies_train.columns:
    if col not in test_df.columns:
        test_df[col] = 0

# Удаление ненужных столбцов
cols_to_drop = ["Name", "Ticket", "Cabin", "PassengerId"]
train_df.drop(cols_to_drop, axis=1, inplace=True, errors="ignore")
test_df.drop(cols_to_drop, axis=1, inplace=True, errors="ignore")

# Проверка
print("\nИтоговый train_df:")
print(train_df.head())
print("\nИнформация о train_df:")
print(train_df.info())

print("\nИтоговый test_df:")
print(test_df.head())
print("\nИнформация о test_df:")
print(test_df.info())
