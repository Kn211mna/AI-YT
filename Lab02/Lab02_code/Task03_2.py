from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
import numpy as np

# Завантаження датасету
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)

# shape
print("Розмірність датасету: {}".format(dataset.shape))

# Зріз даних head
print("\nПерші 20 рядків даних:")
print(dataset.head(20))

# Стастичні зведення методом describe
print("\nСтатистичні зведення даних:")
print(dataset.describe())

# Розподіл за атрибутом class
print("\nРозподіл класів:")
print(dataset.groupby('class').size())

# Діаграма розмаху
dataset.plot(kind='box', subplots=True, layout=(2, 2), sharex=False, sharey=False)
pyplot.title('Діаграма розмаху для атрибутів')
pyplot.show()

# Гістограма розподілу атрибутів датасета
dataset.hist()
pyplot.title('Гістограма розподілу атрибутів')
pyplot.show()

# Матриця діаграм розсіювання
scatter_matrix(dataset)
pyplot.title('Матриця діаграм розсіювання')
pyplot.show()

# Розділення датасету на навчальну та контрольну вибірки
array = dataset.values
X = array[:, 0:4]  # Вибір перших 4-х стовпців
Y = array[:, 4]  # Вибір 5-го стовпця
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=0.20, random_state=1)

# Завантажуємо алгоритми моделі
models = []
models.append(('LR', OneVsRestClassifier(LogisticRegression(solver='liblinear'))))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

# Оцінюємо модель на кожній ітерації
results = []
names = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('{}: {:.2f} ± {:.2f}'.format(name, cv_results.mean(), cv_results.std()))

# Порівняння алгоритмів
pyplot.boxplot(results, tick_labels=names)  # Зміна labels на tick_labels
pyplot.title('Порівняння алгоритмів')
pyplot.show()

# Створюємо прогноз на контрольній вибірці
model = SVC(gamma='auto')
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)

# Оцінюємо прогноз
print("\nОцінка моделі на контрольній вибірці:")
print("Точність: {:.2f}".format(accuracy_score(Y_validation, predictions)))
print("\nМатриця плутанини:")
print(confusion_matrix(Y_validation, predictions))
print("\nЗвіт про класифікацію:")
print(classification_report(Y_validation, predictions))

# Прогноз для нових даних
X_new = np.array([[5, 2.9, 1, 0.2]])
print("\nФорма масиву X_new: {}".format(X_new.shape))
prediction = model.predict(X_new)

# Виводимо результати прогнозу
print("Прогноз для нових даних: {}".format(prediction))
print("Спрогнозована мітка класу: {}".format(prediction[0]))
