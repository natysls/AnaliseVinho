import pandas as pd
import numpy as np
import random
import filter
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

random.seed(42)
data = pd.read_csv('wine.csv')
data.replace("?", pd.NA, inplace=True)
data = filter.toda_filtragem(data)

classes = np.array(pd.unique(data[data.columns[-1]]),dtype=str)
print("Número de linhas e colunas de uma matriz de atributos: ", data.shape)
atributos = list(data.columns)

data = data.to_numpy()
nrow, ncol = data.shape
X = data[:,0:ncol-1]
y = data[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

def probabilidade_gaussiana(y, z):
    prob = 1
    for j in np.arange(0, z.shape[1]):
        media = np.mean(z[:, j])
        desvio = np.std(z[:, j])
        prob = prob * np.exp(-np.power(y[j] - media, 2.) / (2 * np.power(desvio, 2.)))
    return prob

def probabilidade_multinomial(y, z):
    prob = 1
    for j in np.arange(0, z.shape[1]):
        count_y = (z[:, j] == y[j]).sum()
        prob = prob * count_y / len(z)
    return prob

def probabilidade_bernoulli(y, z):
    prob = 1
    for j in np.arange(0, z.shape[1]):
        prob = prob * (z[:, j] == y[j]).mean()
    return prob

P_gaussiano = pd.DataFrame(data=np.zeros((X_test.shape[0], len(classes))), columns=classes)
P_multinomial = pd.DataFrame(data=np.zeros((X_test.shape[0], len(classes))), columns=classes)
P_bernoulli = pd.DataFrame(data=np.zeros((X_test.shape[0], len(classes))), columns=classes)

for i in np.arange(0, len(classes)):
    elementos = tuple(np.nonzero(y_train==float(classes[i])))
    z = X_train[elementos,:][0]
    for j in np.arange(0, X_test.shape[0]):
        x = X_test[j,:]
        pj_gaussiana = probabilidade_gaussiana(x, z)
        P_gaussiano[classes[i]][j] = (pj_gaussiana * len(elementos[0])) / X_train.shape[0]

        pj_multinomial = probabilidade_multinomial(x, z)
        P_multinomial[classes[i]][j] = (pj_multinomial * len(elementos[0])) / X_train.shape[0]

        pj_bernoulli = probabilidade_bernoulli(x, z)
        P_bernoulli[classes[i]][j] = (pj_bernoulli * len(elementos[0])) / X_train.shape[0]


distance_metrics = ['gaussiana', 'multinomial', 'bernoulli'] 
index = 0
for matriz in [P_gaussiano, P_multinomial, P_bernoulli]:
    y_pred = []
    for i in np.arange(0, matriz.shape[0]):
        c = np.argmax(np.array(matriz.iloc[[i]]))
        y_pred.append(matriz.columns[c])
    y_pred = np.array(y_pred, dtype=str)

    y_test_string = []
    for i, valor in enumerate(y_test):
        valor = str(valor)
        y_test_string.append(valor.rstrip('0').rstrip('.'))
    y_test_string = np.array(y_test_string, dtype=str) 

    score = accuracy_score(y_pred, y_test_string)
    print(f"Métrica = {distance_metrics[index]}, Acurácia = {score:.2f}")
    index += 1