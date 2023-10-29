import pandas as pd
import numpy as np
import filter
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree

data = pd.read_csv('wine.csv')
data.replace("?", pd.NA, inplace=True)
data = filter.toda_filtragem(data)

alcohol = 'Alcohol'
malic_acid = 'Malic Acid'
hue = 'Hue'
wine = 'Wine'

df = data[[alcohol, malic_acid, hue, wine]]
X = df[[alcohol, malic_acid, hue]]
y = df[wine]
y = filter.formatar_em_string(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

arvore_decisao = DecisionTreeClassifier(criterion='entropy')
arvore_decisao.fit(X_test, y_test)

print(arvore_decisao.feature_importances_)
previsores = [alcohol, malic_acid, hue, wine]
figura, eixos = plt.subplots(nrows=1, ncols=1, figsize=(10,10))
tree.plot_tree(arvore_decisao, feature_names=previsores, class_names=arvore_decisao.classes_, filled=True)
plt.show()