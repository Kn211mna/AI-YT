import json
import numpy as np
from sklearn import covariance, cluster
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

# Завантаження даних із JSON
json_path = "updated_stock_data.json"
with open(json_path, "r") as file:
    data = json.load(file)

# Отримання символів компаній та їхніх назв
symbols = [item["Symbol"] for item in data]
names = [item["Name"] for item in data]

# Замість завантаження даних із yfinance
quotes = []
valid_symbols = []
valid_names = []

# Отримання історичних даних з JSON
for item in data:
    historical_data = item.get("HistoricalData", [])
    if historical_data:
        # Збереження котирувань (тільки цін закриття та відкриття)
        quotes.append(historical_data)
        valid_symbols.append(item["Symbol"])
        valid_names.append(item["Name"])
    else:
        print(f"Немає історичних даних для {item['Symbol']}")

# Перевірка, чи отримано дані
if not quotes:
    print("Не вдалося отримати дані для жодної компанії.")
    exit()

# Обчислення нормалізованих змін цін
closing_prices = []
opening_prices = []

# Отримання даних цін закриття та відкриття
for stock_data in quotes:
    closing_prices.append([day['Close'] for day in stock_data])
    opening_prices.append([day['Open'] for day in stock_data])

closing_prices = np.array(closing_prices)
opening_prices = np.array(opening_prices)
quotes_diff = closing_prices - opening_prices

# Нормалізація даних (кожен рядок відповідає окремій компанії)
X = quotes_diff / quotes_diff.std(axis=1, keepdims=True)

# Перевірка розмірів масивів
print(f"Кількість компаній: {len(valid_names)}")
print(f"Розмір X: {X.shape}")

# Побудова графової моделі залежностей з меншою кількістю сплітів
edge_model = covariance.GraphicalLassoCV(cv=KFold(n_splits=3))

# Навчання графової моделі
with np.errstate(invalid='ignore'):
    edge_model.fit(X)

# Виконання кластеризації
affinity_model = cluster.AffinityPropagation(affinity="euclidean", damping=0.9)
affinity_model.fit(X)  # Використання X для кластеризації, а не подібностей

# Отримання міток кластерів
labels = affinity_model.labels_
num_clusters = len(np.unique(labels))
print(f"\nКількість кластерів: {num_clusters}\n")

# Додаткове виведення для перевірки розмірів
print(f"Розмір labels: {len(labels)}")

# Перевірка відповідності міток компаніям
if len(labels) != len(valid_names):
    print("Невідповідність між кількістю міток та компаній.")
else:
    # Виведення компаній у кожному кластері
    for i in range(num_clusters):
        cluster_members = np.array(valid_names)[labels == i]
        if cluster_members.size > 0:
            print(f"Кластер {i + 1} =>", ', '.join(cluster_members))
        else:
            print(f"Кластер {i + 1} порожній.")

# Візуалізація кластерів
plt.figure(figsize=(10, 8))

# Перевірка, чи розмір міток відповідає розміру даних
if X.shape[0] == len(labels):
    for i in range(num_clusters):
        members = X[labels == i]
        if members.size > 0:
            plt.scatter(members.mean(axis=1), np.zeros_like(members[:, 0]) + i, label=f"Кластер {i + 1}")
else:
    print("Невідповідність між даними та мітками кластерів. Неможливо побудувати графік.")

plt.title("Кластери фондового ринку на основі методу поширення подібності")
plt.xlabel("Нормалізована різниця котирувань")
plt.yticks(range(num_clusters), [f"Кластер {i + 1}" for i in range(num_clusters)])
plt.legend(loc="best")
plt.show()
