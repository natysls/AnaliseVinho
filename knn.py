import numpy as np
import pandas as pd
import filter
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

data = pd.read_csv('wine.csv')
data.replace("?", pd.NA, inplace=True)
data = filter.toda_filtragem(data)

alcohol = 'Alcohol'
malic_acid = 'Malic Acid'
wine = 'Wine'

df = data[[alcohol, malic_acid, wine]]

X = df[[alcohol, malic_acid]]
y = df[wine]

# Dividindo o conjunto de dados em treinamento e em teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
""""
fig, ax = plt.subplots()
ax.scatter(x=X_train[alcohol], y=X_train[malic_acid], 
           c=y_train, alpha=0.7, cmap='viridis')
plt.show()
"""
k_values = [3, 5, 7] 
distance_metrics = ['euclidean', 'minkowski', 'manhattan'] 

for k in k_values:
    for metric in distance_metrics:
        knn = KNeighborsClassifier(n_neighbors=k, metric=metric) 
        
        knn.fit(X_train, y_train) # Treinando o modelo
        
        y_pred = knn.predict(X_test) # Previsão no conjunto de teste

        accuracy = accuracy_score(y_test, y_pred)
        print(f"K = {k}, Métrica = {metric}, Acurácia = {accuracy:.2f}")

        print(X_test[y_test != y_pred])

        fig, ax = plt.subplots()
        ax.scatter(x=X_train[alcohol], y=X_train[malic_acid], c=y_train, alpha=0.7, cmap='viridis')
        ax.scatter(x=X_test[alcohol], y=X_test[malic_acid], c=y_pred, alpha=0.2, cmap='RdYlGn')
        ax.scatter(x=X_test[alcohol], y=X_test[malic_acid], c=y_test, alpha=0.2, cmap='RdYlGn')
        plt.show()


        
