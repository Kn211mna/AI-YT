import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth
from itertools import cycle

# Завантаження вхідних даних з файлу
X = np.loadtxt('data_clustering.txt', delimiter=',')

# Оцінка ширини вікна (bandwidth) для набору даних X
# Ширина вікна визначає, наскільки великими будуть кластери
# Параметр quantile впливає на ширину вікна: при більшому значенні кількість кластерів зменшується
bandwidth_X = estimate_bandwidth(X, quantile=0.1, n_samples=len(X))

# Ініціалізація та навчання моделі MeanShift з використанням обчисленої ширини вікна
meanshift_model = MeanShift(bandwidth=bandwidth_X, bin_seeding=True)
meanshift_model.fit(X)

# Отримання центрів кластерів після навчання
cluster_centers = meanshift_model.cluster_centers_
print('\nЦентри кластерів:\n', cluster_centers)

# Отримання міток кластерів для кожної точки в наборі даних
labels = meanshift_model.labels_

# Оцінка кількості кластерів (кількість унікальних міток)
num_clusters = len(np.unique(labels))
print("\nКількість кластерів у вхідних даних =", num_clusters)

# Візуалізація точок даних з кольорами для кожного кластера
plt.figure()
markers = cycle('o*xvs')  # Цикл для маркерів, щоб кожен кластер мав свій стиль точки

for i, marker in zip(range(num_clusters), markers):
    # Відображення точок, що належать до поточного кластера
    plt.scatter(X[labels == i, 0], X[labels == i, 1], marker=marker, color='black', label=f'Кластер {i+1}')

    # Відображення центру поточного кластера
    center = cluster_centers[i]
    plt.plot(center[0], center[1], marker='o', markerfacecolor='black',
             markeredgecolor='black', markersize=15)

plt.title('Кластери, отримані методом зсуву середнього')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()
