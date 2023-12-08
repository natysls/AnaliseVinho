# Importar bibliotecas
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import filter

data = pd.read_csv('wine.csv')
data.replace("?", pd.NA, inplace=True)
data = filter.toda_filtragem(data)

X = data.drop('Wine', axis=1) 
y = data['Wine']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

n_estimators_values = [10, 50, 100]
criteria_values = ['gini', 'entropy']

for n_estimators in n_estimators_values:
    for criterion in criteria_values:
        model = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(f"Número de Estimadores: {n_estimators}, Critério: {criterion}")
        filter.validacao(y_test, y_pred)
        filter.validacao_cruzada(model, X_train, y_train)
        filter.curva_roc(model, X_test, y_test)