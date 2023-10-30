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
previsores = [alcohol, malic_acid, hue, wine]

df = data[[alcohol, malic_acid, hue, wine]]
X = df[[alcohol, malic_acid, hue]]
y = df[wine]
y = filter.formatar_em_string(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

arvore_decisao_entropia = DecisionTreeClassifier(criterion='entropy')
arvore_decisao_entropia.fit(X_test, y_test)
print("Entropia: ", arvore_decisao_entropia.feature_importances_)
previsoes_entropia = arvore_decisao_entropia.predict([[13.39, 1.77, 0.92], [13.62, 4.95, 0.91]])
print("Previsões Entropia: ", previsoes_entropia)
figura, eixos = plt.subplots(nrows=1, ncols=1, figsize=(10,10))
tree.plot_tree(arvore_decisao_entropia, feature_names=previsores, class_names=arvore_decisao_entropia.classes_, filled=True)
plt.show()


arvore_decisao_gini = DecisionTreeClassifier(criterion='gini')
arvore_decisao_gini.fit(X_test, y_test)
print("Gini: ", arvore_decisao_gini.feature_importances_)
previsoes_gini = arvore_decisao_entropia.predict([[13.39, 1.77, 0.92], [13.62, 4.95, 0.91]])
print("Previsões Gini: ", previsoes_gini)

figura, eixos = plt.subplots(nrows=1, ncols=1, figsize=(10,10))
tree.plot_tree(arvore_decisao_gini, feature_names=previsores, class_names=arvore_decisao_gini.classes_, filled=True)
plt.show()



