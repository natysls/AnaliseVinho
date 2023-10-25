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

def gaussiano(valor, media, desvio):
    return np.exp(-np.power(valor - media, 2.) / (2 * np.power(desvio, 2.)))

def probabilidade(y, z):
    prob = 1
    for j in np.arange(0, z.shape[1]):
        media = np.mean(z[:, j])
        desvio = np.std(z[:, j])
        prob = prob * gaussiano(y[j], media, desvio)
    return prob

P = pd.DataFrame(data=np.zeros((X_test.shape[0], len(classes))), columns=classes)
for i in np.arange(0, len(classes)):
    elementos = tuple(np.where(y_train==float(classes[i])))
    z = X_train[elementos,:][0]
    for j in np.arange(0, X_test.shape[0]):
        x = X_test[j,:]
        pj = probabilidade(x, z)
        P[classes[i]][j] = pj*len(elementos)/X_train.shape[0]

y_pred = []
for i in np.arange(0, P.shape[0]):
    c = np.argmax(np.array(P.iloc[[i]]))
    y_pred.append(P.columns[c])
y_pred = np.array(y_pred, dtype=str)

y_test_string = []
for i, valor in enumerate(y_test):
    valor = str(valor)
    y_test_string.append(valor.rstrip('0').rstrip('.'))
y_test_string = np.array(y_test_string, dtype=str) 

score = accuracy_score(y_pred, y_test_string)
print("Acurácia: ", score)