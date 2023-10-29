import pandas as pd
import numpy as np
import random
import filter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

random.seed(42)
data = pd.read_csv('wine.csv')
data.replace("?", pd.NA, inplace=True)
data = filter.toda_filtragem(data)
data = data[data['Wine'] != 3]

X_classes = data['Wine'].values
X_classes = X_classes.reshape(-1, 1)

""""
sns.displot(X_classes, kde=True)
plt.show()
"""

y_prev = data['Hue'].values
y_prev = filter.formatar_em_numero(y_prev)

bernoulli = BernoulliNB()
bernoulli = bernoulli.fit(X_classes, y_prev)

previsoes = bernoulli.predict(X_classes)

print("Previsões: ", previsoes)
print("Previstos: ", y_prev)

score = accuracy_score(y_prev, previsoes)
print(f"Métrica = bernoulli, Acurácia = {score:.2f}")
