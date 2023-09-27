import pandas as pd

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