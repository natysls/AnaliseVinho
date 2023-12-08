# Importar bibliotecas
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import filter

data = pd.read_csv('wine.csv')
data.replace("?", pd.NA, inplace=True)
data = filter.toda_filtragem(data)

X = data.drop('Wine', axis=1) 
y = data['Wine']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

valores_C = [0.1, 1, 10]
valores_gamma = [0.01, 0.1, 1]

def svm():
    kernels = ['linear', 'rbf', 'poly']
    for kernel in kernels:
        print(f"\nKernel: {kernel}\n")
        
        for C in valores_C:
            
            if kernel in ['rbf', 'poly']:
                for gamma in valores_gamma:
                    modelo = SVC(kernel=kernel, C=C, gamma=gamma, probability=True)
                    modelo.fit(X_train, y_train)
                    y_pred = modelo.predict(X_test)

                    print(f"C={C}, gamma={gamma}:")
                    filter.validacao(y_test, y_pred)
                    filter.validacao_cruzada(modelo, X, y)
                    filter.curva_roc(modelo, X_test, y_test)
                    
            else:
                modelo = SVC(kernel=kernel, C=C, probability=True)
                modelo.fit(X_train, y_train)
                y_pred = modelo.predict(X_test)

                print(f"C={C}:")
                filter.validacao(y_test, y_pred)
                filter.validacao_cruzada(modelo, X, y)
                filter.curva_roc(modelo, X_test, y_test)
    
svm()
