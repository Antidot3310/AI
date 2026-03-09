import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def fill_missing(train: pd.DataFrame, test: pd.DataFrame):
    fill_values = {
        "Age": train["Age"].median(),
        "Embarked": train["Embarked"].mode()[0],
        "Cabin": "U",
    }
    train.fillna(fill_values, inplace=True)
    test.fillna(fill_values, inplace=True)
    return train, test


def normalize(train: pd.DataFrame, test: pd.DataFrame, columns):
    scaler = MinMaxScaler()
    train[columns] = scaler.fit_transform(train[columns])
    test[columns] = scaler.transform(test[columns])
    return train, test


def one_encode(train: pd.DataFrame, test: pd.DataFrame, column):
    dummies_train = pd.get_dummies(train[column], prefix=column, drop_first=True)
    dummies_test = pd.get_dummies(test[column], prefix=column, drop_first=True)

    # в test появятся все столбцы из train, недостающие заполнятся 0
    dummies_test = dummies_test.reindex(columns=dummies_train.columns, fill_value=0)

    train = pd.concat([train, dummies_train], axis=1).drop(column, axis=1)
    test = pd.concat([test, dummies_test], axis=1).drop(column, axis=1)
    return train, test


def drop_columns(train: pd.DataFrame, test: pd.DataFrame, columns):
    train.drop(columns=columns, inplace=True, errors="ignore")
    test.drop(columns=columns, inplace=True, errors="ignore")
    return train, test


df = pd.read_csv("train.csv")

# Первичный анализ
print("Первые 5 строк:")
print(df.head())
print("\nИнформация:")
print(df.info())
print("\nСтатистика:")
print(df.describe())

# Разделение на train / test
train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)

# Анализ пропусков
missing_train = train_df.isnull().sum()
print("\nПропуски в train выборке:")
print(missing_train[missing_train > 0])

# Заполнение пропусков (используем только train)
fill_missing(train_df, test_df)

# Проверка
assert train_df.isnull().sum().sum() == 0

#  Нормализация числовых признаков
num_cols = ["Age", "Fare", "SibSp", "Parch"]
normalize(train_df, test_df, num_cols)

# Кодирование категориальных признаков
# Sex
train_df["Sex"] = train_df["Sex"].map({"male": 0, "female": 1})
test_df["Sex"] = test_df["Sex"].map({"male": 0, "female": 1})

cols = {"Embarked", "Pclass"}
for item in cols:
    train_df, test_df = one_encode(train_df, test_df, item)

# Удаление ненужных столбцов
cols_to_drop = ["Name", "Ticket", "Cabin", "PassengerId"]
drop_columns(train_df, test_df, cols_to_drop)

# Проверка
print("\nИтоговый train_df:")
print(train_df.head())
print("\nИнформация о train_df:")
print(train_df.info())

print("\nИтоговый test_df:")
print(test_df.head())
print("\nИнформация о test_df:")
print(test_df.info())

# Сохранение обработанные данные в CSV
train_df.to_csv("train_processed.csv", index=False)
test_df.to_csv("test_processed.csv", index=False)
