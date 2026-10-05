import os

import pandas as pd
import numpy as np

from evaluate_model import evaluate, sigmoid

RESULTS_DIR = 'results'
TEST_SIZE = 0.2  # Частка тестової вибірки
RANDOM_SEED = 42  # Фіксований seed, щоб розбиття було відтворюваним
TARGET = 'is_high_risk'
CATEGORICAL = ['vehicle_class', 'origin_group']

os.makedirs(RESULTS_DIR, exist_ok=True)

df = pd.read_csv('adac_ml_ready.csv')

# 1. One-Hot Encoding через pandas
# drop_first=True уникає мультиколінеарності
X_raw = pd.get_dummies(df[['age_years'] + CATEGORICAL], drop_first=True).astype(float)
y_all = df[TARGET].values
feature_names = X_raw.columns.tolist()


# 2. Стратифіковане розбиття на навчальну та тестову вибірки
# Класи незбалансовані (~73% / 27%), тому перемішуємо та ділимо кожен клас окремо,
# щоб частка is_high_risk в обох вибірках була однаковою
def stratified_split(y, test_size, seed):
    rng = np.random.default_rng(seed)
    train_idx, test_idx = [], []
    for cls in np.unique(y):
        idx = rng.permutation(np.where(y == cls)[0])
        n_test = int(round(len(idx) * test_size))
        test_idx.extend(idx[:n_test])
        train_idx.extend(idx[n_test:])
    return np.sort(train_idx), np.sort(test_idx)


train_idx, test_idx = stratified_split(y_all, TEST_SIZE, RANDOM_SEED)

X_train_raw, X_test_raw = X_raw.values[train_idx], X_raw.values[test_idx]
y_train, y_test = y_all[train_idx], y_all[test_idx]

# 3. Розрахунок параметрів масштабування ЛИШЕ на навчальній вибірці
# (тестова масштабується тими ж mu та sigma, інакше буде витік даних)
scale_params = {}
X_train_scaled = np.zeros_like(X_train_raw)

for i, col in enumerate(feature_names):
    mu = np.mean(X_train_raw[:, i])
    sigma = np.std(X_train_raw[:, i])
    # Захист від ділення на нуль (якщо колонка має однакові значення)
    if sigma == 0: sigma = 1e-8
    # Масштабуємо
    X_train_scaled[:, i] = (X_train_raw[:, i] - mu) / sigma
    # Зберігаємо mu та sigma для подальшого використання на тесті
    scale_params[col] = {'mean': mu, 'std': sigma}

# Додавання параметра x0 = 1 для обчислення вільного члена (theta_0)
X_final = np.c_[np.ones(X_train_scaled.shape[0]), X_train_scaled]

# 4. Логістична регресія (Градієнтний спуск без регуляризації)
m, n = X_final.shape
theta = np.zeros(n)  # Початкові ваги - нулі
alpha = 0.5  # Швидкість навчання
steps = 3000  # Кількість ітерацій

for _ in range(steps):
    z = X_final @ theta
    p = sigmoid(z)
    # Формула градієнта без L2/L1 штрафу: (1/m) * X.T * (p - y)
    grad = (X_final.T @ (p - y_train)) / m
    theta = theta - alpha * grad

print("=== РОЗБИТТЯ НА ВИБІРКИ ===")
print(f"{'Навчальна вибірка':.<45} {len(train_idx):>8d}")
print(f"{'Тестова вибірка':.<45} {len(test_idx):>8d}")

print("\n=== ПАРАМЕТРИ МАСШТАБУВАННЯ (Зберегти для Test set) ===")
for col, params in scale_params.items():
    print(f"{col:.<45} mean: {params['mean']:>8.4f} | std: {params['std']:>6.4f}")

print("\n=== КОЕФІЦІЄНТИ МОДЕЛІ (THETA) ===")
# theta[0] відповідає за x0=1, тому це вільний член
print(f"{'Вільний член (theta_0)':.<45} {theta[0]:>8.4f}")

# Решта theta відповідають за фічі
for name, weight in zip(feature_names, theta[1:]):
    print(f"{name:.<45} {weight:>8.4f}")

# 5. Збереження результатів у CSV
# Модель: theta разом з mu та sigma (для вільного члена масштабування немає)
model_df = pd.DataFrame({
    'feature': ['intercept'] + feature_names,
    'theta': theta,
    'mean': [np.nan] + [scale_params[c]['mean'] for c in feature_names],
    'std': [np.nan] + [scale_params[c]['std'] for c in feature_names],
})
model_df.to_csv(os.path.join(RESULTS_DIR, 'model_params.csv'), index=False)

# Самі вибірки (у вихідному вигляді, з індексом рядка з adac_ml_ready.csv)
df.iloc[train_idx].to_csv(os.path.join(RESULTS_DIR, 'train_set.csv'), index_label='row_id')
df.iloc[test_idx].to_csv(os.path.join(RESULTS_DIR, 'test_set.csv'), index_label='row_id')

# Розподіл цільової змінної та ознак по вибірках
df_split = df.assign(split=np.where(np.isin(np.arange(len(df)), test_idx), 'test', 'train'))
dist_rows = []
for col in [TARGET, 'age_years'] + CATEGORICAL:
    counts = df_split.groupby([col, 'split']).size().unstack(fill_value=0)
    for value, row in counts.iterrows():
        dist_rows.append({
            'variable': col,
            'value': value,
            'train_count': row['train'],
            'test_count': row['test'],
            'train_share': row['train'] / len(train_idx),
            'test_share': row['test'] / len(test_idx),
        })
dist_rows.append({
    'variable': 'total', 'value': 'all',
    'train_count': len(train_idx), 'test_count': len(test_idx),
    'train_share': len(train_idx) / len(df), 'test_share': len(test_idx) / len(df),
})
pd.DataFrame(dist_rows).to_csv(os.path.join(RESULTS_DIR, 'split_distribution.csv'), index=False)

# 6. Одразу перевіряємо модель на тестовій (і для порівняння на навчальній) вибірці
evaluate(RESULTS_DIR)
