import pandas as pd
from sklearn.model_selection import train_test_split
print("ПУНКТ 1: РАЗДЕЛЕНИЕ ВЫБОРКИ")
# Загрузка данных из файла
data = pd.read_csv('data_banknote_authentication.txt', header=None)
# Разделяем признаки (X) и целевую переменную (y)
X = data.iloc[:, :-1]  # Все столбцы кроме последнего - это признаки
y = data.iloc[:, -1]   # Последний столбец - целевая переменная
# Первое разделение: временная выборка (85%) и тестовая (15%)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y,
    test_size=0.15,  # 15% для тестовой выборки
    random_state=42,  # Для воспроизводимости результатов
    stratify=y       # Сохраняем распределение классов
)
# Второе разделение: обучающая (70%) и валидационная (15%) от исходных данных
val_relative_size = 0.15 / (1 - 0.15)  # Рассчитываем относительный размер валидационной выборки
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp,
    test_size=val_relative_size,  # 15% от исходных данных
    random_state=42,              # Для воспроизводимости
    stratify=y_temp              # Сохраняем распределение классов
)
# Вывод результатов разделения
print("Количество признаков:", X.shape[1])  # Количество столбцов в X - это количество признаков
print("Общий размер dataset:", len(data))   # Общее количество строк в данных
print(f"Обучающая выборка: {X_train.shape[0]} (70.0%)")
print(f"Валидационная выборка: {X_val.shape[0]} (15.0%)")
print(f"Тестовая выборка: {X_test.shape[0]} (15.0%)")

print("\nПУНКТ 2: МАСШТАБИРОВАНИЕ ПРИЗНАКОВ")
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
# Названия признаков
feature_names = [
    "1 признак Variance of Wavelet Transformed image",
    "2 признак Skewness of Wavelet Transformed image",
    "3 признак Kurtosis of Wavelet Transformed image",
    "4 признак Entropy of image"
]
# Выводим статистику для всех четырех признаков
print("Статистика по всем признакам до и после масштабирования:")
for i in range(4):
    print(f"{feature_names[i]}:")
    print(f"До масштабирования Среднее: {X_train.iloc[:, i].mean():.2f}, Стандартное отклонение: {X_train.iloc[:, i].std():.2f}")
    print(f"После масштабирования Среднее: {X_train_scaled[:, i].mean():.2f}, Стандартное отклонение: {X_train_scaled[:, i].std():.2f}")

print("\nПУНКТ 3 и 4: ОБУЧЕНИЕ МОДЕЛИ И РАСЧЕТ ТОЧНОСТИ")
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
# Обучение Perceptron
perceptron = Perceptron(random_state=42, max_iter=1000, eta0=0.1, tol=1e-3)
perceptron.fit(X_train_scaled, y_train)
# Обучение MLPClassifier
mlp = MLPClassifier(random_state=42, hidden_layer_sizes=(100,), activation='relu',
                   solver='adam', alpha=0.0001, learning_rate_init=0.001,
                   max_iter=1000, tol=1e-4)
mlp.fit(X_train_scaled, y_train)
# Предсказания на валидационной выборке
y_val_pred_perceptron = perceptron.predict(X_val_scaled)
y_val_pred_mlp = mlp.predict(X_val_scaled)
# Предсказания на тестовой выборке
y_test_pred_perceptron = perceptron.predict(X_test_scaled)
y_test_pred_mlp = mlp.predict(X_test_scaled)
# Вычисление точности
val_accuracy_perceptron = accuracy_score(y_val, y_val_pred_perceptron)
val_accuracy_mlp = accuracy_score(y_val, y_val_pred_mlp)
test_accuracy_perceptron = accuracy_score(y_test, y_test_pred_perceptron)
test_accuracy_mlp = accuracy_score(y_test, y_test_pred_mlp)
# Вывод только требуемой информации
print("Точность на валидационной выборке (для настройки параметров):")
print(f"Perceptron: {val_accuracy_perceptron:.4f} ({val_accuracy_perceptron*100:.2f}%)")
print(f"MLPClassifier: {val_accuracy_mlp:.4f} ({val_accuracy_mlp*100:.2f}%)")
print("Точность на тестовой выборке (финальная оценка):")
print(f"Perceptron: {test_accuracy_perceptron:.4f} ({test_accuracy_perceptron*100:.2f}%)")
print(f"MLPClassifier: {test_accuracy_mlp:.4f} ({test_accuracy_mlp*100:.2f}%)")

print("\nПУНКТ 5: ЭКСПЕРИМЕНТЫ")
import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron
# Инициализируем списки для сохранения результатов
perceptron_results = []
mlp_results = []
# Параметры для Perceptron
print("1. Подбираем лучшие параметры для Perceptron...")
# Определяем сетку параметров для Perceptron
perceptron_params = {
    'eta0': [0.001, 0.01, 0.1, 1.0],  # коэффициенты обучения
    'alpha': [0.0001, 0.001, 0.01, 0.1],  # параметры регуляризации
    'penalty': ['l1', 'l2', 'elasticnet']  # типы регуляризации
}
total_combinations = len(perceptron_params['eta0']) * len(perceptron_params['alpha']) * len(
    perceptron_params['penalty'])
start_time = time.time()
best_perceptron = None
best_perceptron_score = 0
best_perceptron_params = {}
# перебираем все комбинации параметров
for eta0 in perceptron_params['eta0']:
    for alpha in perceptron_params['alpha']:
        for penalty in perceptron_params['penalty']:
            # создаем и обучаем модель
            perceptron = Perceptron(
                random_state=42,
                eta0=eta0,
                alpha=alpha,
                penalty=penalty,
                max_iter=1000,
                tol=1e-3
            )
            perceptron.fit(X_train_scaled, y_train)
            # проверяем точность на валидационной выборке
            val_accuracy = perceptron.score(X_val_scaled, y_val)
            # сохраняем результат для анализа
            perceptron_results.append({
                'eta0': eta0,
                'alpha': alpha,
                'penalty': penalty,
                'accuracy': val_accuracy
            })
            # сохраняем лучший результат
            if val_accuracy > best_perceptron_score:
                best_perceptron_score = val_accuracy
                best_perceptron = perceptron
                best_perceptron_params = {
                    'eta0': eta0,
                    'alpha': alpha,
                    'penalty': penalty
                }
end_time = time.time()
search_time = end_time - start_time
print(f"Поиск занял: {search_time:.2f} секунд")
print(f"Всего комбинаций: {total_combinations}")
print(f"- Лучший коэффициент обучения (eta0): {best_perceptron_params['eta0']}")
print(f"- Лучший параметр регуляризации (alpha): {best_perceptron_params['alpha']}")
print(f"- Лучший тип регуляризации (penalty): {best_perceptron_params['penalty']}")
print(f"Лучшая точность на валидационной выборке: {best_perceptron_score:.4f}")
test_accuracy_perceptron = best_perceptron.score(X_test_scaled, y_test)
print(f"Лучшая точность на тестовой выборке: {test_accuracy_perceptron:.4f}")
# Параметры для MLPClassifier
print("2. Подбираем лучшие параметры для MLPClassifier...")
# Определяем сетку параметров для MLPClassifier
mlp_params = {
    'learning_rate_init': [0.001, 0.01, 0.1],  # коэффициенты обучения
    'alpha': [0.0001, 0.001, 0.01, 0.1],  # параметры регуляризации
    'solver': ['sgd', 'adam', 'lbfgs']  # функции оптимизации
}
total_combinations = len(mlp_params['learning_rate_init']) * len(mlp_params['alpha']) * len(mlp_params['solver'])
start_time = time.time()
best_mlp = None
best_mlp_score = 0
best_mlp_params = {}
# перебираем все комбинации параметров
for learning_rate in mlp_params['learning_rate_init']:
    for alpha in mlp_params['alpha']:
        for solver in mlp_params['solver']:
            # создаем и обучаем модель
            mlp = MLPClassifier(
                random_state=42,
                hidden_layer_sizes=(100,),
                learning_rate_init=learning_rate,
                alpha=alpha,
                solver=solver,
                max_iter=1000,
                tol=1e-4
            )
            mlp.fit(X_train_scaled, y_train)
            # проверяем точность на валидационной выборке
            val_accuracy = mlp.score(X_val_scaled, y_val)
            # сохраняем результат для анализа
            mlp_results.append({
                'learning_rate': learning_rate,
                'alpha': alpha,
                'solver': solver,
                'accuracy': val_accuracy
            })
            # сохраняем лучший результат
            if val_accuracy > best_mlp_score:
                best_mlp_score = val_accuracy
                best_mlp = mlp
                best_mlp_params = {
                    'learning_rate_init': learning_rate,
                    'alpha': alpha,
                    'solver': solver
                }
end_time = time.time()
search_time = end_time - start_time
print(f"Поиск занял: {search_time:.2f} секунд")
print(f"Всего комбинаций: {total_combinations}")
print(f"- Лучший коэффициент обучения: {best_mlp_params['learning_rate_init']}")
print(f"- Лучший параметр регуляризации: {best_mlp_params['alpha']}")
print(f"- Лучшая функция оптимизации: {best_mlp_params['solver']}")
print(f"Лучшая точность на валидационной выборке: {best_mlp_score:.4f}")
test_accuracy_mlp = best_mlp.score(X_test_scaled, y_test)
print(f"Лучшая точность на тестовой выборке: {test_accuracy_mlp:.4f}")

# Создаем графики для анализа результатов - 2 строки, 3 столбца
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Первая строка: графики для Perceptron
# График 1: Сравнение коэффициентов обучения для Perceptron
eta0_values = perceptron_params['eta0']
eta0_accuracies = []
for eta0 in eta0_values:
    eta0_results = [r for r in perceptron_results if r['eta0'] == eta0]
    avg_accuracy = np.mean([r['accuracy'] for r in eta0_results])
    eta0_accuracies.append(avg_accuracy)
bars1 = axes[0, 0].bar([str(eta0) for eta0 in eta0_values], eta0_accuracies,
                      color=['lightblue', 'lightgreen', 'salmon', 'gold'], alpha=0.8)
axes[0, 0].set_xlabel('Коэффициент обучения (eta0)')
axes[0, 0].set_ylabel('Средняя точность')
axes[0, 0].set_title('Perceptron: Влияние коэффициента обучения')
for bar, accuracy in zip(bars1, eta0_accuracies):
    axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                   f'{accuracy:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
axes[0, 0].grid(True, alpha=0.3)

# График 2: Сравнение параметров регуляризации для Perceptron
alpha_values_perceptron = perceptron_params['alpha']
alpha_accuracies_perceptron = []
for alpha_val in alpha_values_perceptron:
    alpha_results = [r for r in perceptron_results if r['alpha'] == alpha_val]
    avg_accuracy = np.mean([r['accuracy'] for r in alpha_results])
    alpha_accuracies_perceptron.append(avg_accuracy)
bars2 = axes[0, 1].bar([str(alpha) for alpha in alpha_values_perceptron], alpha_accuracies_perceptron,
                      color=['lightblue', 'lightgreen', 'salmon', 'gold'], alpha=0.8)
axes[0, 1].set_xlabel('Параметр регуляризации (Alpha)')
axes[0, 1].set_ylabel('Средняя точность')
axes[0, 1].set_title('Perceptron: Влияние параметра регуляризации')
for bar, accuracy in zip(bars2, alpha_accuracies_perceptron):
    axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                   f'{accuracy:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
axes[0, 1].grid(True, alpha=0.3)

# График 3: Сравнение типов регуляризации для Perceptron
penalties = perceptron_params['penalty']
penalty_accuracies = []
for penalty in penalties:
    penalty_results = [r for r in perceptron_results if r['penalty'] == penalty]
    avg_accuracy = np.mean([r['accuracy'] for r in penalty_results])
    penalty_accuracies.append(avg_accuracy)
bars3 = axes[0, 2].bar(penalties, penalty_accuracies,
                      color=['lightblue', 'lightgreen', 'salmon'], alpha=0.8)
axes[0, 2].set_xlabel('Тип регуляризации (Penalty)')
axes[0, 2].set_ylabel('Средняя точность')
axes[0, 2].set_title('Perceptron: Сравнение типов регуляризации')
for bar, accuracy in zip(bars3, penalty_accuracies):
    axes[0, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                   f'{accuracy:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
axes[0, 2].grid(True, alpha=0.3)

# Вторая строка: графики для mlpclassifier
# График 4: Сравнение коэффициентов обучения для MLPClassifier
learning_rate_values = mlp_params['learning_rate_init']
learning_rate_accuracies = []
for lr in learning_rate_values:
    lr_results = [r for r in mlp_results if r['learning_rate'] == lr]
    avg_accuracy = np.mean([r['accuracy'] for r in lr_results])
    learning_rate_accuracies.append(avg_accuracy)
bars4 = axes[1, 0].bar([str(lr) for lr in learning_rate_values], learning_rate_accuracies,
                      color=['lightblue', 'lightgreen', 'salmon'], alpha=0.8)
axes[1, 0].set_xlabel('Коэффициент обучения')
axes[1, 0].set_ylabel('Средняя точность')
axes[1, 0].set_title('MLPClassifier: Влияние коэффициента обучения')
for bar, accuracy in zip(bars4, learning_rate_accuracies):
    axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                   f'{accuracy:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
axes[1, 0].grid(True, alpha=0.3)

# График 5: Сравнение параметров регуляризации для MLPClassifier
alpha_values_mlp = mlp_params['alpha']
alpha_accuracies_mlp = []
for alpha_val in alpha_values_mlp:
    alpha_results = [r for r in mlp_results if r['alpha'] == alpha_val]
    avg_accuracy = np.mean([r['accuracy'] for r in alpha_results])
    alpha_accuracies_mlp.append(avg_accuracy)
bars5 = axes[1, 1].bar([str(alpha) for alpha in alpha_values_mlp], alpha_accuracies_mlp,
                      color=['lightblue', 'lightgreen', 'salmon', 'gold'], alpha=0.8)
axes[1, 1].set_xlabel('Параметр регуляризации (Alpha)')
axes[1, 1].set_ylabel('Средняя точность')
axes[1, 1].set_title('MLPClassifier: Влияние параметра регуляризации')
for bar, accuracy in zip(bars5, alpha_accuracies_mlp):
    axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                   f'{accuracy:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
axes[1, 1].grid(True, alpha=0.3)
# График 6: Сравнение функций оптимизации для MLPClassifier
solvers = mlp_params['solver']
solver_accuracies = []
for solver in solvers:
    solver_results = [r for r in mlp_results if r['solver'] == solver]
    avg_accuracy = np.mean([r['accuracy'] for r in solver_results])
    solver_accuracies.append(avg_accuracy)
bars6 = axes[1, 2].bar(solvers, solver_accuracies,
                      color=['lightblue', 'lightgreen', 'salmon'], alpha=0.8)
axes[1, 2].set_xlabel('Функция оптимизации')
axes[1, 2].set_ylabel('Средняя точность')
axes[1, 2].set_title('MLPClassifier: Сравнение функций оптимизации')
for bar, accuracy in zip(bars6, solver_accuracies):
    axes[1, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                   f'{accuracy:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
axes[1, 2].grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
print("Все графики успешно построены")