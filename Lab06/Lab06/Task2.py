import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

url = "https://raw.githubusercontent.com/susanli2016/Machine-Learning-with-Python/master/data/renfe_small.csv"
data = pd.read_csv(url)

print("Перші 5 рядків даних:")
print(data.head())
print("\nКолонки в наборі даних:", data.columns,"\n")

# Видалення рядків з пропущеними значеннями
data = data.dropna()

# Видаляємо текстові/дата-часові колонки, які не потрібні для моделі
data = data.drop(columns=['insert_date', 'origin', 'destination', 'start_date', 'end_date', 'train_class'])

# Список колонок, які хочемо закодувати
columns_to_encode = ['train_type', 'fare']

# Фільтруємо, залишаючи тільки ті, що наявні в даних
columns_to_encode = [col for col in columns_to_encode if col in data.columns]

# Перетворення категоріальних змінних на числові за допомогою One-Hot Encoding
data = pd.get_dummies(data, columns=columns_to_encode)

# Створимо категорії цін: низька (< 25), середня (25-50), висока (> 50)
data['price_category'] = pd.cut(data['price'], bins=[0, 25, 50, np.inf], labels=['Low', 'Medium', 'High'])

# Визначимо вхідні та вихідні змінні
X = data.drop(['price', 'price_category'], axis=1)
y = data['price_category']

# Розділимо дані на тренувальну та тестову вибірки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Ініціалізація та навчання наївного байєсівського класифікатора
model = GaussianNB()
model.fit(X_train, y_train)

# Прогнозуємо на тестовій вибірці
y_pred = model.predict(X_test)

# Оцінюємо точності моделі
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
