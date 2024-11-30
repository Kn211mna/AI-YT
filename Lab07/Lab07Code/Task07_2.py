import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.metrics import pairwise_distances_argmin

iris = load_iris()
X = iris['data']   # Дані про атрибути квіток Iris
y = iris['target'] # Класи квіток (Setosa, Versicolour, Virginica)

# Ініціалізація моделі
kmeans = KMeans(
    n_clusters=3,      # Кількість кластерів дорівнює 3 (оскільки є 3 класи квітів)
    init='k-means++',   # Використання методу ініціалізації k-means++ для швидшої збіжності
    n_init=10,          # Кількість запусків алгоритму з різними початковими центроїдами
    max_iter=300,       # Максимальна кількість ітерацій для одного запуску алгоритму
    tol=0.0001,         # Допустима похибка для зупинки алгоритму
    random_state=42     # Встановлюємо random_state для відтворюваності результатів
)

# Навчання моделі
kmeans.fit(X)

# Отримання передбачених міток для кожної точки даних
y_kmeans = kmeans.predict(X)

# Візуалізація кластерів
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')  # Відображення точок з кольорами

# Отримання центрів кластерів
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5, marker='X')  # Відображення центрів кластерів

plt.title('Кластеризація K-середніх для набору даних Iris')
plt.xlabel('Довжина чашолистка')
plt.ylabel('Ширина чашолистка')
plt.show()

# Функція для ручної кластеризації, що знаходить центри кластерів
def find_cluster(X, n_clusters, rseed=2):
    rng = np.random.RandomState(rseed)                  # Ініціалізація випадкових чисел для відтворюваності
    i = rng.permutation(X.shape[0])[:n_clusters]        # Вибір початкових центрів випадковим чином
    centers = X[i]                                      # Обрані центри

    while True:
        # Призначення кожної точки до найближчого центру кластера
        labels = pairwise_distances_argmin(X, centers)

        # Обчислення нових центрів кластерів як середнє значення точок у кожному кластері
        new_centers = np.array([X[labels == i].mean(0) for i in range(n_clusters)])

        # Якщо центри не змінилися, виходимо з циклу
        if np.all(centers == new_centers):
            break

        centers = new_centers

    return centers, labels

# Використання функції find_cluster з 3 кластерами
centers, labels = find_cluster(X, 3)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
plt.title("Кластеризація з find_cluster")
plt.xlabel("Довжина чашолистка")
plt.ylabel("Ширина чашолистка")
plt.show()
