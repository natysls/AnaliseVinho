import pandas as pd
import numpy as np

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