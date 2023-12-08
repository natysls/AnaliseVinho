import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn import preprocessing

def coluna_filtrada(df, nome_coluna):
    return df[nome_coluna].drop([0,1])

# Tratando o outlier de Color intensity
def remover_segundo_ponto(palavra): 
    primeira_ocorrencia = palavra.find(".")
    
    if primeira_ocorrencia != -1:
        segunda_ocorrencia = palavra.find(".", primeira_ocorrencia + 1)
        
        if segunda_ocorrencia != -1:
            palavra_sem_segundo_ponto = palavra[:segunda_ocorrencia] + palavra[segunda_ocorrencia+1:]
            return palavra_sem_segundo_ponto
    
    return palavra

def filtragem(df, nome_coluna):
    dt_coluna = coluna_filtrada(df, nome_coluna)

    index = 0
    for valor in dt_coluna.values:
        if pd.notna(valor):
            numero_pontos = valor.count(".")
            if pd.to_numeric(numero_pontos) > 1:
                # mais dois porque cortei duas linhas na primeira filtragem
                dt_coluna[index + 2] = remover_segundo_ponto(valor)
        index += 1

    return pd.to_numeric(dt_coluna)

def toda_coluna_filtrada(df):
    return df.drop([0,1])

def toda_filtragem(df):
    dt_coluna = toda_coluna_filtrada(df)

    index = 0
    for valor in dt_coluna.values:
        index_valor = 0
        for elemento in valor:
            if pd.isna(elemento):
                valor[index_valor] = 0
                dt_coluna.loc[index + 2] = valor
            else:
                numero_pontos = elemento.count(".")
                if pd.to_numeric(numero_pontos) > 1:
                    valor[index_valor] = remover_segundo_ponto(elemento)
                    dt_coluna.loc[index + 2] = valor
            index_valor += 1       
        index += 1
    
    dt_coluna = dt_coluna.apply(pd.to_numeric, errors='coerce')
    return dt_coluna

def formatar_em_numero(x):
    x_numerico = []
    for valor in x:
        numero_inteiro = int(valor)
        x_numerico.append(numero_inteiro)
    x_numerico = np.array(x_numerico) 
    return x_numerico

def formatar_em_string(x):
    x_string = []
    for valor in x:
        palavra = str(valor)
        x_string.append(palavra)
    x_string = np.array(x_string) 
    return x_string

def validacao(df_y_test, df_y_pred):
    acuracia = accuracy_score(df_y_test, df_y_pred)
    precisao = precision_score(df_y_test, df_y_pred, average='weighted', zero_division=0)
    recall = recall_score(df_y_test, df_y_pred, average='weighted')
    f1 = f1_score(df_y_test, df_y_pred, average='weighted')

    # Imprimir as métricas
    print(f'Acurácia: {acuracia:.4f}')
    print(f'Precisão: {precisao:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F-Score: {f1:.4f}')


def curva_roc(model, df_X_test, df_y_test):
    probabilidade_positivo_classe = model.predict_proba(df_X_test)
    y_test_binarized = preprocessing.label_binarize(df_y_test, classes=np.unique(df_y_test))
    taxa_fpr, taxa_tpr, thresholds = roc_curve(y_test_binarized[:, 1], probabilidade_positivo_classe[:,1])
    auc_score = auc(taxa_fpr, taxa_tpr)
    print("Área sob a curva (ROC):", auc_score)
    print("\n")

    #plot_curva_roc(taxa_fpr, taxa_tpr, auc_score)


def plot_curva_roc(taxa_fpr, taxa_tpr, auc_score):
    plt.plot(taxa_fpr, taxa_tpr, color='darkorange', lw=2, label=f'AUC = {auc_score:.2f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('Taxa de Falsos Positivos')
    plt.ylabel('Taxa de Verdadeiros Positivos')
    plt.title('Curva ROC')
    plt.legend()
    plt.show()

def validacao_cruzada(model, df_train_X, df_train_y):
    cross_val_results = cross_val_score(model, df_train_X, df_train_y, cv=5, scoring='accuracy')
    print("Validação Cruzada:", cross_val_results)
    print("Precisão Média da Validação Cruzada: {:.2f}%".format(cross_val_results.mean() * 100))