import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report

def visualize_classifier(classifier, X, y, title='Classifier boundaries'):
    # Визначення меж для сітки
    min_x, max_x = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
    min_y, max_y = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0
    mesh_step_size = 0.01  # Крок сітки для відображення області рішень

    # Визначення сітки точок для області рішень
    x_vals, y_vals = np.meshgrid(np.arange(min_x, max_x, mesh_step_size),
                                 np.arange(min_y, max_y, mesh_step_size))

    # Прогнозування для кожної точки на сітці
    output = classifier.predict(np.c_[x_vals.ravel(), y_vals.ravel()])
    output = output.reshape(x_vals.shape)

    # Створення графіку
    plt.figure()
    plt.title(title)

    # Відображення областей рішень
    plt.contourf(x_vals, y_vals, output, cmap=plt.cm.coolwarm, alpha=0.3)

    # Відображення вхідних точок
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], c='black', marker='x', label='Class 0')
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], c='white', edgecolors='black', marker='o', label='Class 1')

    # Додавання легенди та показ графіку
    plt.legend()
    plt.show()

# Зчитування даних
input_file = 'data_imbalance.txt'
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]

class_0 = np.array(X[y == 0])
class_1 = np.array(X[y == 1])

# Візуалізація вхідних даних
plt.figure()
plt.scatter(class_0[:, 0], class_0[:, 1], s=75, facecolors='black', linewidth=1, marker='x')
plt.scatter(class_1[:, 0], class_1[:, 1], s=75, facecolors='white', edgecolors='black', linewidth=1, marker='o')
plt.title('Вхідні данні')

# Розділення даних на тренувальні та тестові
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=5)

# Налаштування параметрів для класифікатора
params = {'n_estimators': 100, 'max_depth': 4, 'random_state': 0}

if len(sys.argv) > 1:
    if sys.argv[1] == 'balance':
        params['class_weight'] = 'balanced'
    else:
        raise TypeError("Invalid input argument; should be 'balance'")

# Ініціалізація та тренування класифікатора
classifier = ExtraTreesClassifier(**params)
classifier.fit(X_train, y_train)
visualize_classifier(classifier, X_test, y_test, 'Trained dataset')

# Прогнозування та візуалізація результатів
y_test_pred = classifier.predict(X_test)
visualize_classifier(classifier, X_test, y_test_pred, 'Тестовий набор даних')

class_names = ['Class-0', 'Class-1']
print("\n" + "#"*40)
print("\nClassifier performance on training dataset\n")
print(classification_report(y_train, classifier.predict(X_train), target_names=class_names))
print("#"*40 + "\n")
print("#"*40)
print("\nClassifier performance on test dataset\n")
print(classification_report(y_test, y_test_pred, target_names=class_names))
print("#"*40 + "\n")
plt.show()