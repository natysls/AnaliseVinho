# Importar bibliotecas
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import filter

data = pd.read_csv('wine.csv')
data.replace("?", pd.NA, inplace=True)
data = filter.toda_filtragem(data)

X = data.drop('Wine', axis=1) 
y = data['Wine']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def validacao(df_y_test, df_y_pred):
    acuracia = accuracy_score(df_y_test, df_y_pred)
    precisao = precision_score(df_y_test, df_y_pred, average='weighted', zero_division=0)
    recall = recall_score(df_y_test, df_y_pred, average='weighted')
    f1 = f1_score(df_y_test, df_y_pred, average='weighted')

    # Imprimir as métricas
    print(f'Acurácia: {acuracia:.4f}')
    print(f'Precisão: {precisao:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F-Score: {f1:.4f}\n')

valores_C = [0.1, 1, 10]
valores_gamma = [0.01, 0.1, 1]

kernels = ['linear', 'rbf', 'poly']
for kernel in kernels:
    print(f"\nKernel: {kernel}\n")
    
    for C in valores_C:
        
        if kernel in ['rbf', 'poly']:
            for gamma in valores_gamma:
                modelo = SVC(kernel=kernel, C=C, gamma=gamma)
                modelo.fit(X_train, y_train)
                y_pred = modelo.predict(X_test)

                print(f"C={C}, gamma={gamma}:")
                validacao(y_test, y_pred)
        else:
            modelo = SVC(kernel=kernel, C=C)
            modelo.fit(X_train, y_train)
            y_pred = modelo.predict(X_test)

            print(f"C={C}:")
            validacao(y_test, y_pred)


