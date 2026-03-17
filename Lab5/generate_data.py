import numpy as np

np.random.seed(42)

# Параметры
n_features = 12
n_samples = 5000
noise_level = 0.05

# Генерируем признаки
X = np.random.randint(0, 2, size=(n_samples, n_features))

# Выбираем случайное количество комбинаций (от 10 до 20)
n_combinations = np.random.randint(10, 20)
combs = []
for _ in range(n_combinations):
    size = np.random.randint(2, 5)  # размер комбинации от 2 до 4
    comb = tuple(sorted(np.random.choice(n_features, size, replace=False)))
    combs.append(comb)


# Для каждой комбинации случайно выбираем тип функции
def apply_func(x, func_type):
    if func_type == 0:  # AND
        return np.all(x, axis=1).astype(int)
    elif func_type == 1:  # OR
        return np.any(x, axis=1).astype(int)
    elif func_type == 2:  # XOR (чётность)
        return (np.sum(x, axis=1) % 2).astype(int)
    elif func_type == 3:  # NAND
        return 1 - np.all(x, axis=1).astype(int)
    elif func_type == 4:  # NOR
        return 1 - np.any(x, axis=1).astype(int)


# Случайные типы для каждой комбинации
func_types = np.random.randint(0, 5, size=n_combinations)

# Вычисляем значения для каждой комбинации
votes = np.zeros((n_samples, n_combinations))
for i, (comb, ftype) in enumerate(zip(combs, func_types)):
    votes[:, i] = apply_func(X[:, comb], ftype)

# Итоговая метка: победа правящей партии (класс 1), если сумма голосов > порога
# Порог подбираем так, чтобы классы были примерно сбалансированы
score = votes.sum(axis=1)
median_score = np.median(score)
y = (score > median_score).astype(int)

# Добавляем шум
noise_mask = np.random.random(n_samples) < noise_level
y[noise_mask] = 1 - y[noise_mask]

# One-hot encoding
Y_onehot = np.zeros((n_samples, 2))
Y_onehot[np.arange(n_samples), y] = 1

# Перемешиваем данные (важно)
indices = np.random.permutation(n_samples)
X = X[indices]
Y_onehot = Y_onehot[indices]

# Сохраняем
np.savetxt("dataIn.txt", X.T, fmt="%d")
np.savetxt("dataOut.txt", Y_onehot.T, fmt="%d")
