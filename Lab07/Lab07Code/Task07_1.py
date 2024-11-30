import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

X = np.loadtxt('data_clustering.txt', delimiter=',')

# Кількість кластерів
num_clusters = 5

# Візуалізація вхідних даних
plt.figure()
plt.scatter(X[:, 0], X[:, 1], marker='o', facecolors='none', edgecolors='black', s=80)
plt.title('Вхідні дані')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# Ініціалізація та навчання моделі KMeans
kmeans = KMeans(init='k-means++', n_clusters=num_clusters, n_init=10, random_state=42)
kmeans.fit(X)

# Виведення оцінки силуета для оцінки якості кластеризації
silhouette_avg = silhouette_score(X, kmeans.labels_)
print(f"Середній коефіцієнт силуета: {silhouette_avg:.2f}")

# Встановлення параметрів сітки
step_size = 0.01
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

# Побудова сітки координат для відображення кордонів кластерів
x_vals, y_vals = np.meshgrid(np.arange(x_min, x_max, step_size), np.arange(y_min, y_max, step_size))

# Передбачення кластерних міток для кожної точки сітки
output = kmeans.predict(np.c_[x_vals.ravel(), y_vals.ravel()])
output = output.reshape(x_vals.shape)

# Візуалізація меж кластерів
plt.figure()
plt.imshow(output, interpolation='nearest', extent=(x_vals.min(), x_vals.max(), y_vals.min(), y_vals.max()),
           cmap=plt.cm.Paired, aspect='auto', origin='lower')

# Відображення вхідних даних
plt.scatter(X[:, 0], X[:, 1], marker='o', facecolors='none', edgecolors='black', s=80)

# Відображення центрів кластерів
cluster_centers = kmeans.cluster_centers_
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], marker='x', s=150, linewidths=3, color='red', zorder=10)

# Настроювання меж графіку
plt.title('Кластеризація методом K-середніх з межами кластерів')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.show()
