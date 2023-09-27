import pandas as pd
import filter

data = pd.read_csv('wine.csv')
data.replace("?", pd.NA, inplace=True)
   
def media(df, nome_coluna):
    df = filter.filtragem(df, nome_coluna)
    media = df.mean()
    print("A média é: ", media)   

def moda(df, nome_coluna):
    df = filter.filtragem(df, nome_coluna)
    moda = df.mode()
    valor_moda = moda.iloc[0]
    print("A moda é: ", valor_moda)  

def mediana(df, nome_coluna):
    df = filter.filtragem(df, nome_coluna)
    mediana = df.median()
    print("A mediana é: ", mediana) 

def percentil(df, nome_coluna):
    df = filter.filtragem(df, nome_coluna)
    print("O percentil 20 é: ", df.quantile(20 / 100))
    print("O percentil 50 é: ", df.quantile(50 / 100)) 
    print("O percentil 75 é: ", df.quantile(75 / 100))  

def quartis(df, nome_coluna):
    df = filter.filtragem(df, nome_coluna)
    quartis = df.quantile([0.20, 0.5, 0.75])
    print("O primeiro Quartil (Q1):", quartis[0.20])
    print("A mediana (Q2):", quartis[0.5])
    print("O terceiro Quartil (Q3):", quartis[0.75])

def variancia(df, nome_coluna):
    df = filter.filtragem(df, nome_coluna)
    variancia = df.var()
    print("A variância é: ", variancia)

def desvio_padrao(df, nome_coluna):
    df = filter.filtragem(df, nome_coluna)
    desvio_padrao = df.std()
    print("O desvio Padrão é: ", desvio_padrao)

colunas_selecionadas = ['Alcohol', 'Magnesium', 'Color intensity', 'Wine']

for nome_coluna in colunas_selecionadas:
    print("Coluna: ", nome_coluna)
    media(data, nome_coluna)
    moda(data, nome_coluna)
    mediana(data, nome_coluna)
    percentil(data, nome_coluna)
    quartis(data, nome_coluna)
    variancia(data, nome_coluna)
    desvio_padrao(data, nome_coluna)
    print("\n")
