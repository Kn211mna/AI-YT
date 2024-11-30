import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import ExtraTreesClassifier
from matplotlib.colors import ListedColormap

def visualize_classifier(classifier, X, y):
    # Задаємо мінімум і максимум для графіка
    x_min, x_max = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
    y_min, y_max = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0

    # Кроки сітки
    step_size = 0.01

    # Створюємо сітку точок
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step_size),
                         np.arange(y_min, y_max, step_size))

    # Передбачаємо класи для кожної точки сітки
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Створюємо кольорову карту для відображення класифікаційних областей
    cmap_background = ListedColormap(['lightgray', 'lightblue', 'lightgreen'])
    cmap_points = ListedColormap(['black', 'blue', 'green'])

    # Відображаємо області класифікації
    plt.contourf(xx, yy, Z, cmap=cmap_background, alpha=0.3)

    # Відображаємо точки навчального набору
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_points, s=50)

    # Настройки графіка
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision boundaries and data points')
    plt.show()

# Завантаження даних
input_file = 'data_random_forests.txt'
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]

# Розподіл даних на тренувальний та тестовий набори
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=5)

# Налаштування параметрів для GridSearchCV
parameter_grid = [
    {'n_estimators': [100], 'max_depth': [2, 4, 7, 12, 16]},
    {'max_depth': [4], 'n_estimators': [25, 50, 100, 250]}
]

metrics = ['precision_weighted', 'recall_weighted']

# Пошук оптимальних параметрів
for metric in metrics:
    print("\n#### Пошук оптимальних параметрів для", metric)
    classifier = GridSearchCV(ExtraTreesClassifier(random_state=0), parameter_grid, cv=5, scoring=metric)
    classifier.fit(X_train, y_train)

    # Виведення результатів
    print("\nРезультати оцінки параметрів:")
    for params, avg_score in zip(classifier.cv_results_['params'], classifier.cv_results_['mean_test_score']):
        print(params, '-->', round(avg_score, 3))

    print("\nНайкращі параметри:", classifier.best_params_)

# Оцінка моделі
y_pred = classifier.predict(X_test)
print("\nЗвіт про продуктивність:\n")
print(classification_report(y_test, y_pred))

# Візуалізація класифікатора для тренувальних даних
visualize_classifier(classifier.best_estimator_, X_train, y_train)
