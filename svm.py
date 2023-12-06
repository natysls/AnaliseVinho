# Importar bibliotecas
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
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

def curva_roc(model, df_X, df_y, df_X_test, df_y_test):
    probabilidade_positivo_classe = model.predict_proba(df_X_test)
    y_test_binarized = preprocessing.label_binarize(df_y_test, classes=np.unique(df_y_test))
    taxa_fpr, taxa_tpr, thresholds = roc_curve(y_test_binarized[:, 1], probabilidade_positivo_classe[:,1])
    auc_score = auc(taxa_fpr, taxa_tpr)
    print("Acurácia da área sob a curva (ROC):", auc_score)
    print("\n")

    plt.plot(taxa_fpr, taxa_tpr, color='darkorange', lw=2, label=f'AUC = {auc_score:.2f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('Taxa de Falsos Positivos')
    plt.ylabel('Taxa de Verdadeiros Positivos')
    plt.title('Curva ROC')
    plt.legend()
    plt.show()

def validacao_cruzada(model, df_X, df_y):
    cross_val_results = cross_val_score(model, df_X, df_y, cv=5, scoring='accuracy')
    print("Resultados da Validacao Cruzada:", cross_val_results)
    print("Precisão Média: {:.2f}%".format(cross_val_results.mean() * 100))

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
                    #validacao(y_test, y_pred)
                    curva_roc(modelo, X, y, X_test, y_test)
            else:
                modelo = SVC(kernel=kernel, C=C, probability=True)
                modelo.fit(X_train, y_train)
                y_pred = modelo.predict(X_test)

                print(f"C={C}:")
                #validacao(y_test, y_pred)
                curva_roc(modelo, X, y, X_test, y_test)

#svm()

def novo_svm():
    modelo = SVC(gamma='auto')
    modelo.fit(X_train, y_train)
    ac = modelo.score(X_test, y_test)
    print(ac)
    validacao_cruzada(modelo, X, y)

novo_svm()

