import pandas as pd
import matplotlib.pyplot as plt
import filter

data = pd.read_csv('wine.csv')
data.replace("?", pd.NA, inplace=True)

def boxplot(df, nome_coluna):
    df = filter.filtragem(df, nome_coluna)
    plt.boxplot(df, vert=False)
    plt.xlabel(nome_coluna)
    plt.title('Boxplot da coluna ' + nome_coluna)
    plt.show()

def histograma(df, nome_coluna):
    df = filter.filtragem(df, nome_coluna)
    plt.hist(df, bins=10, edgecolor='black')  # bins define o número de caixas no histograma
    plt.xlabel('Valores')
    plt.ylabel('Frequência')
    plt.title('Histograma da coluna ' + nome_coluna)
    plt.show()

def scatter(df, nome_coluna_1, nome_coluna_2):
    df_1 = filter.filtragem(df, nome_coluna_1)
    df_2 = filter.filtragem(df, nome_coluna_2)
    plt.scatter(df_1, df_2)
    plt.xlabel(nome_coluna_1)
    plt.ylabel(nome_coluna_2)
    plt.title('Scatter Plot entre ' + nome_coluna_1 + ' e ' + nome_coluna_2)
    plt.show()


colunas_selecionadas = ['Alcohol', 'Magnesium', 'Color intensity', 'Wine']

for nome_coluna in colunas_selecionadas:
    #boxplot(data, nome_coluna)
    histograma(data, nome_coluna)

#scatter(data, 'Alcohol', 'Magnesium')