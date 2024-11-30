import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Вхідний файл, який містить дані
input_file = 'income_data.txt'

# Читання даних
X = []
y = []
count_class1 = 0
count_class2 = 0
max_datapoints = 25000

with open(input_file, 'r') as f:
    for line in f.readlines():
        if count_class1 >= max_datapoints and count_class2 >= max_datapoints:
            break
        if '?' in line:
            continue
        data = line[:-1].split(', ')

        if data[-1] == '<=50K' and count_class1 < max_datapoints:
            X.append(data)
            count_class1 += 1
        elif data[-1] == '>50K' and count_class2 < max_datapoints:
            X.append(data)
            count_class2 += 1

# Перетворення на масив numpy
X = np.array(X)

# Перетворення рядкових даних на числові
label_encoder = []
X_encoded = np.empty(X.shape)
for i, item in enumerate(X[0]):
    if item.isdigit():
        X_encoded[:, i] = X[:, i]
    else:
        le = preprocessing.LabelEncoder()
        X_encoded[:, i] = le.fit_transform(X[:, i])
        label_encoder.append(le)

X = X_encoded[:, :-1].astype(int)
y = X_encoded[:, -1].astype(int)

# Розділення даних на навчальні та тестові вибірки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# Функція для навчання та оцінки SVM
def train_and_evaluate_svm(kernel_type):
    print(f"\nSVM з {kernel_type} ядром:")
    classifier = SVC(kernel=kernel_type)
    classifier.fit(X_train, y_train)
    # Передбачення та оцінка
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    # Виведення результатів
    print(f"Accuracy (Акуратність): {accuracy * 100:.2f}%")
    print(f"Precision (Точність): {precision * 100:.2f}%")
    print(f"Recall (Повнота): {recall * 100:.2f}%")
    print(f"F1 Score (F-міра): {f1 * 100:.2f}%")

# Гаусове (радіальне) ядро
train_and_evaluate_svm('rbf')

# Сигмоїдальне ядро
train_and_evaluate_svm('sigmoid')

# Поліноміальне ядро
train_and_evaluate_svm('poly')