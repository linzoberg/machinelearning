import pandas as pd  # для работы с табличными данными
import numpy as np  # для математических операций
from sklearn.linear_model import LinearRegression  # модель линейной регрессии
from sklearn.metrics import mean_squared_error, r2_score  # метрики качества модели
import matplotlib.pyplot as plt  # для построения графиков

print("ПУНКТ 1: РАЗДЕЛЕНИЕ ВЫБОРКИ")
# Читаем данные из файла Excel, лист 'Sheet1'
data = pd.read_excel('Concrete_Data.xls', sheet_name='Sheet1')
# Функция для разделения данных на тренировочные и тестовые
def custom_split(x, y, test_size=0.2, random_state=42):
    # Устанавливаем "зерно" для случайных чисел, чтобы результаты были одинаковыми при каждом запуске
    np.random.seed(random_state)
    # n = общее количество строк в данных (сколько всего записей)
    n = len(x)
    # test_count = сколько записей отдать в тестовую выборку (20% от общего числа)
    test_count = int(n * test_size)
    # indices = перемешанные номера строк
    indices = np.random.permutation(n)
    # Делим данные:
    # x_train = признаки для обучения (все строки кроме первых test_count)
    x_train = x.iloc[indices[test_count:]]
    # x_test = признаки для теста (первые test_count строк)
    x_test = x.iloc[indices[:test_count]]
    # y_train = правильные ответы для обучения
    y_train = y.iloc[indices[test_count:]]
    # y_test = правильные ответы для теста
    y_test = y.iloc[indices[:test_count]]
    return x_train, x_test, y_train, y_test
# Подготавливаем данные:
# X = все столбцы кроме последнего (цемент, вода, добавки, возраст)
X = data.iloc[:, :-1]
# y = последний столбец (прочность бетона - то, что мы хотим предсказывать)
y = data.iloc[:, -1]
# Вызываем нашу функцию разделения
X_train, X_test, y_train, y_test = custom_split(X, y)
# Выводим информацию
print(f"Количество признаков (компонентов бетона): {X.shape[1]}")
print(f"Общий размер dataset: {len(data)}")
print(f"Обучающая выборка: {len(X_train)}")
print(f"Тестовая выборка: {len(X_test)}")
print(f"Соотношение: {len(X_train)/len(data):.1%} обучающей/ {len(X_test)/len(data):.1%} тестовой")

print("ПУНКТ 2 и 3: ОБУЧЕНИЕ ЛИНЕЙНОЙ РЕГРЕССИИ И ПРОВЕРКА ТОЧНОСТИ МОДЕЛИ")
# Создаем модель линейной регрессии
model = LinearRegression()
# Обучаем модель на тренировочных данных
model.fit(X_train, y_train)
# Получаем предсказания модели на обучающих и тестовых данных
y_pred_train = model.predict(X_train)  # предсказания на обучающей выборке
y_pred_test = model.predict(X_test)  # предсказания на тестовой выборке
# Вычисляем метрики качества модели на ОБУЧАЮЩЕЙ выборке
mse_train = mean_squared_error(y_train, y_pred_train)  # MSE на обучении
r2_train = r2_score(y_train, y_pred_train)  # R² на обучении
# Вычисляем метрики качества модели на ТЕСТОВОЙ выборке
mse_test = mean_squared_error(y_test, y_pred_test)  # MSE на тесте
r2_test = r2_score(y_test, y_pred_test)  # R² на тесте
# Выводим требуемые метрики
print(f"MSE на обучении: {mse_train:.4f}")
print(f"MSE на тесте:    {mse_test:.4f}")
print(f"• Средняя ошибка на обучении: {mse_train ** 0.5:.2f} МПа")
print(f"• Средняя ошибка на тесте: {mse_test ** 0.5:.2f} МПа")
print(f"R² на обучении:  {r2_train:.4f}")
print(f"R² на тесте:     {r2_test:.4f}")
print(f"• Модель объясняет {r2_train * 100:.1f}% изменчивости на обучающих данных")
print(f"• Модель объясняет {r2_test * 100:.1f}% изменчивости на тестовых данных")
# Анализ переобучения
if r2_train > r2_test + 0.1:
    print("ВНИМАНИЕ: Возможное переобучение (модель лучше работает на обучающих данных)")
else:
    print("Модель хорошо обобщает (результаты на обучении и тесте близки)")

print("ПУНКТ 4: ПОЛИНОМИАЛЬНАЯ ФУНКЦИЯ")
# Импортируем необходимые библиотеки для полиномиальной регрессии
from sklearn.preprocessing import PolynomialFeatures  # для создания полиномиальных признаков
from sklearn.pipeline import Pipeline  # для создания конвейера обработки данных
# Создаем списки для хранения метрик для разных степеней полинома
degrees = [1, 2, 3, 4, 5]  # степени полинома которые будем проверять
train_mse_scores = []  # MSE на обучении для каждой степени
test_mse_scores = []  # MSE на тесте для каждой степени
train_r2_scores = []  # R² на обучении для каждой степени
test_r2_scores = []  # R² на тесте для каждой степени
print("Исследуем полиномиальную регрессию для степеней:", degrees)
# Перебираем разные степени полинома
for degree in degrees:
    # Создаем конвейер: сначала создаем полиномиальные признаки, затем применяем линейную регрессию
    model = Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),  # этап создания полиномиальных признаков
        ('linear', LinearRegression())  # этап линейной регрессии
    ])
    # Обучаем модель на обучающих данных
    model.fit(X_train, y_train)
    # Делаем предсказания на обучающих и тестовых данных
    y_train_pred = model.predict(X_train)  # предсказания на данных обучения
    y_test_pred = model.predict(X_test)  # предсказания на данных тестирования
    # Вычисляем метрики качества для обучающей выборки
    train_mse = mean_squared_error(y_train, y_train_pred)  # MSE на обучении
    train_r2 = r2_score(y_train, y_train_pred)  # R² на обучении
    # Вычисляем метрики качества для тестовой выборки
    test_mse = mean_squared_error(y_test, y_test_pred)  # MSE на тесте
    test_r2 = r2_score(y_test, y_test_pred)  # R² на тесте
    # Сохраняем результаты в списки
    train_mse_scores.append(train_mse)
    test_mse_scores.append(test_mse)
    train_r2_scores.append(train_r2)
    test_r2_scores.append(test_r2)
    # Выводим результаты для текущей степени полинома
    print(
        f"Степень {degree}: MSE train={train_mse:.2f}, test={test_mse:.2f} | R² train={train_r2:.4f}, test={test_r2:.4f}")
# Находим лучшую степень полинома по максимальному R² на тестовой выборке
best_degree = degrees[np.argmax(test_r2_scores)]  # находим индекс максимального R² и получаем соответствующую степень
# Выводим информацию о лучшей степени полинома
print(f"Лучшая степень полинома: {best_degree}")
print(f"Строим финальную модель с лучшей степенью полинома ({best_degree})...")

print("ПУНКТ 5: РЕГУЛЯРИЗАЦИЯ")
# Импортируем модели регуляризации
from sklearn.linear_model import Ridge, Lasso  # Ridge - L2 регуляризация, Lasso - L1 регуляризация
from sklearn.preprocessing import StandardScaler  # для стандартизации данных
# Стандартизируем данные для регуляризации (важно для корректной работы)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # обучаем scaler на тренировочных данных и преобразуем их
X_test_scaled = scaler.transform(X_test)  # преобразуем тестовые данные используя параметры с тренировочных
# Создаем диапазон значений alpha для регуляризации (от очень маленьких до больших)
alphas = np.logspace(-6, 3, 50)  # 50 значений от 10^-6 до 10^3 в логарифмической шкале
# Списки для хранения метрик
ridge_train_mse = []  # MSE для Ridge на обучении
ridge_test_mse = []  # MSE для Ridge на тесте
ridge_train_r2 = []  # R² для Ridge на обучении
ridge_test_r2 = []  # R² для Ridge на тесте
lasso_train_mse = []  # MSE для Lasso на обучении
lasso_test_mse = []  # MSE для Lasso на тесте
lasso_train_r2 = []  # R² для Lasso на обучении
lasso_test_r2 = []  # R² для Lasso на тесте
# Перебираем разные значения alpha для Ridge и Lasso
for alpha in alphas:
    # Ridge регрессия (L2 регуляризация) - уменьшает веса всех признаков
    ridge = Ridge(alpha=alpha)  # создаем модель Ridge с текущим alpha
    ridge.fit(X_train_scaled, y_train)  # обучаем на стандартизированных данных
    # Lasso регрессия (L1 регуляризация) - может обнулять неважные признаки
    lasso = Lasso(alpha=alpha, max_iter=10000)  # создаем модель Lasso, увеличиваем max_iter для сходимости
    lasso.fit(X_train_scaled, y_train)  # обучаем на стандартизированных данных
    # Предсказания для Ridge
    ridge_train_pred = ridge.predict(X_train_scaled)  # предсказания на обучающих данных
    ridge_test_pred = ridge.predict(X_test_scaled)  # предсказания на тестовых данных
    # Предсказания для Lasso
    lasso_train_pred = lasso.predict(X_train_scaled)  # предсказания на обучающих данных
    lasso_test_pred = lasso.predict(X_test_scaled)  # предсказания на тестовых данных
    # Вычисляем метрики для Ridge и сохраняем
    ridge_train_mse.append(mean_squared_error(y_train, ridge_train_pred))  # MSE Ridge на обучении
    ridge_test_mse.append(mean_squared_error(y_test, ridge_test_pred))  # MSE Ridge на тесте
    ridge_train_r2.append(r2_score(y_train, ridge_train_pred))  # R² Ridge на обучении
    ridge_test_r2.append(r2_score(y_test, ridge_test_pred))  # R² Ridge на тесте
    # Вычисляем метрики для Lasso и сохраняем
    lasso_train_mse.append(mean_squared_error(y_train, lasso_train_pred))  # MSE Lasso на обучении
    lasso_test_mse.append(mean_squared_error(y_test, lasso_test_pred))  # MSE Lasso на тесте
    lasso_train_r2.append(r2_score(y_train, lasso_train_pred))  # R² Lasso на обучении
    lasso_test_r2.append(r2_score(y_test, lasso_test_pred))  # R² Lasso на тесте
# Находим лучшие alpha для Ridge и Lasso по минимальному MSE на тестовой выборке
best_ridge_alpha = alphas[np.argmin(ridge_test_mse)]  # лучшее alpha для Ridge
best_lasso_alpha = alphas[np.argmin(lasso_test_mse)]  # лучшее alpha для Lasso
# Находим соответствующие лучшие метрики
best_ridge_mse = min(ridge_test_mse)  # лучший MSE для Ridge
best_lasso_mse = min(lasso_test_mse)  # лучший MSE для Lasso
best_ridge_r2 = max(ridge_test_r2)  # лучший R² для Ridge
best_lasso_r2 = max(lasso_test_r2)  # лучший R² для Lasso
# Выводим результаты
print(f"Лучшие параметры регуляризации:")
print(f"Ridge - лучшее alpha: {best_ridge_alpha:.2e}")  # лучшее значение alpha для Ridge
print(f"Lasso - лучшее alpha: {best_lasso_alpha:.2e}")  # лучшее значение alpha для Lasso
print(f"Ridge (alpha={best_ridge_alpha:.2e}): MSE={best_ridge_mse:.2f}, R²={best_ridge_r2:.4f}")  # метрики для лучшей Ridge
print(f"Lasso (alpha={best_lasso_alpha:.2e}): MSE={best_lasso_mse:.2f}, R²={best_lasso_r2:.4f}")  # метрики для лучшей Lasso


# Создаем график для сравнения реальных и предсказанных значений
plt.figure(figsize=(15, 5))
# График 1: Результаты на ОБУЧАЮЩЕЙ выборке
plt.subplot(1, 3, 1)
# Идеальная линия (если бы предсказания были идеальными)
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--',
         linewidth=2, label='Идеальные предсказания')
# Наши предсказания vs Реальные значения
plt.scatter(y_train, y_pred_train, alpha=0.7, color='blue', s=50, label='Предсказания')
plt.xlabel('Реальная прочность (МПа)')
plt.ylabel('Предсказанная прочность (МПа)')
plt.title('ОБУЧАЮЩАЯ ВЫБОРКА')
plt.legend()
plt.grid(True, alpha=0.3)
# Добавляем текстовое поле с метриками для обучающей выборки
textstr_train = f'MSE = {mse_train:.2f}\nR² = {r2_train:.4f}'
props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
plt.text(0.05, 0.95, textstr_train, transform=plt.gca().transAxes, fontsize=10,
         verticalalignment='top', bbox=props)
# График 2: Результаты на тестовой выборке
plt.subplot(1, 3, 2)
# Идеальная линия
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--',
         linewidth=2, label='Идеальные предсказания')
# Наши предсказания vs Реальные значения
plt.scatter(y_test, y_pred_test, alpha=0.7, color='green', s=50, label='Предсказания')
plt.xlabel('Реальная прочность (МПа)')
plt.ylabel('Предсказанная прочность (МПа)')
plt.title('ТЕСТОВАЯ ВЫБОРКА')
plt.legend()
plt.grid(True, alpha=0.3)
# Добавляем текстовое поле с метриками для тестовой выборки
textstr_test = f'MSE = {mse_test:.2f}\nR² = {r2_test:.4f}'
plt.text(0.05, 0.95, textstr_test, transform=plt.gca().transAxes, fontsize=10,
         verticalalignment='top', bbox=props)
# График 3: Сравнение ошибок на обучающей и тестовой выборках
plt.subplot(1, 3, 3)
errors_train = y_pred_train - y_train  # ошибки на обучении
errors_test = y_pred_test - y_test  # ошибки на тесте
# Строим гистограммы распределения ошибок
plt.hist(errors_train, bins=30, alpha=0.7, color='blue', label='Ошибки (обучение)')
plt.hist(errors_test, bins=30, alpha=0.7, color='green', label='Ошибки (тест)')
plt.axvline(x=0, color='red', linestyle='--', linewidth=1)  # вертикальная линия на нуле
plt.xlabel('Величина ошибки (МПа)')
plt.ylabel('Количество случаев')
plt.title('РАСПРЕДЕЛЕНИЕ ОШИБОК')
plt.legend()
plt.grid(True, alpha=0.3)
# Добавляем текстовое поле со статистикой ошибок
error_stats = f'Ср. ошибка (train): {errors_train.mean():.2f} МПа\nСр. ошибка (test): {errors_test.mean():.2f} МПа'
plt.text(0.05, 0.95, error_stats, transform=plt.gca().transAxes, fontsize=9,
         verticalalignment='top', bbox=props)
plt.tight_layout()
plt.show()


# Строим графики зависимости точности от степени полинома
plt.figure(figsize=(15, 5))  # создаем фигуру размером 15x5 дюймов
# График 1: Зависимость MSE от степени полинома
plt.subplot(1, 2, 1)  # создаем первый подграфик (1 строка, 2 столбца, позиция 1)
plt.plot(degrees, train_mse_scores, 'o-', linewidth=2, markersize=8,
         label='MSE (обучение)')  # линия для обучающих данных
plt.plot(degrees, test_mse_scores, 'o-', linewidth=2, markersize=8, label='MSE (тест)')  # линия для тестовых данных
plt.xlabel('Степень полинома')  # подпись оси X
plt.ylabel('MSE')  # подпись оси Y
plt.title('ЗАВИСИМОСТЬ MSE ОТ СТЕПЕНИ ПОЛИНОМА')  # заголовок графика
plt.legend()  # отображаем легенду
plt.grid(True, alpha=0.3)  # включаем сетку с прозрачностью 0.3
# График 2: Зависимость R² от степени полинома
plt.subplot(1, 2, 2)  # создаем второй подграфик (1 строка, 2 столбца, позиция 2)
plt.plot(degrees, train_r2_scores, 'o-', linewidth=2, markersize=8, label='R² (обучение)')  # линия для обучающих данных
plt.plot(degrees, test_r2_scores, 'o-', linewidth=2, markersize=8, label='R² (тест)')  # линия для тестовых данных
plt.xlabel('Степень полинома')  # подпись оси X
plt.ylabel('R²')  # подпись оси Y
plt.title('ЗАВИСИМОСТЬ R² ОТ СТЕПЕНИ ПОЛИНОМА')  # заголовок графика
plt.legend()  # отображаем легенду
plt.grid(True, alpha=0.3)  # включаем сетку с прозрачностью 0.3
plt.tight_layout()  # автоматически настраиваем расположение подграфиков
plt.show()  # отображаем графики


# Строим графики зависимости точности от коэффициента регуляризации
plt.figure(figsize=(15, 6))  # создаем фигуру размером 15x6 дюймов
# График 1: MSE для Ridge и Lasso
plt.subplot(1, 2, 1)  # создаем первый подграфик (1 строка, 2 столбца, позиция 1)
plt.semilogx(alphas, ridge_train_mse, 'b-', label='Ridge train', linewidth=2)  # MSE Ridge на обучении (синяя сплошная)
plt.semilogx(alphas, ridge_test_mse, 'b--', label='Ridge test', linewidth=2)  # MSE Ridge на тесте (синяя пунктирная)
plt.semilogx(alphas, lasso_train_mse, 'r-', label='Lasso train',
             linewidth=2)  # MSE Lasso на обучении (красная сплошная)
plt.semilogx(alphas, lasso_test_mse, 'r--', label='Lasso test', linewidth=2)  # MSE Lasso на тесте (красная пунктирная)
plt.xlabel('Alpha (коэффициент регуляризации)')  # подпись оси X
plt.ylabel('MSE')  # подпись оси Y
plt.title('ЗАВИСИМОСТЬ MSE ОТ ALPHA\n(Ridge vs Lasso)')  # заголовок графика
plt.legend()  # отображаем легенду
plt.grid(True, alpha=0.3)  # включаем сетку с прозрачностью 0.3
# График 2: R² для Ridge и Lasso
plt.subplot(1, 2, 2)  # создаем второй подграфик (1 строка, 2 столбца, позиция 2)
plt.semilogx(alphas, ridge_train_r2, 'b-', label='Ridge train', linewidth=2)  # R² Ridge на обучении (синяя сплошная)
plt.semilogx(alphas, ridge_test_r2, 'b--', label='Ridge test', linewidth=2)  # R² Ridge на тесте (синяя пунктирная)
plt.semilogx(alphas, lasso_train_r2, 'r-', label='Lasso train', linewidth=2)  # R² Lasso на обучении (красная сплошная)
plt.semilogx(alphas, lasso_test_r2, 'r--', label='Lasso test', linewidth=2)  # R² Lasso на тесте (красная пунктирная)
plt.xlabel('Alpha (коэффициент регуляризации)')  # подпись оси X
plt.ylabel('R²')  # подпись оси Y
plt.title('ЗАВИСИМОСТЬ R² ОТ ALPHA\n(Ridge vs Lasso)')  # заголовок графика
plt.legend()  # отображаем легенду
plt.grid(True, alpha=0.3)  # включаем сетку с прозрачностью 0.3
plt.tight_layout()  # автоматически настраиваем расположение подграфиков
plt.show()  # отображаем графики
