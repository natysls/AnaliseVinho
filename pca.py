import pandas as pd
import numpy as np
import filter
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

data = pd.read_csv('wine.csv')
data.replace("?", pd.NA, inplace=True)

def pca(df):
    df = filter.toda_filtragem(df)

    # Matriz de covariancia
    cov_matrix = df.cov()

    # Autovalores e Autovetores da matriz de covariancia
    autovalores, autovetores = np.linalg.eig(cov_matrix)

    # Variancia explicada de acordo com a quantidade de autovetores
    variancia_total = autovalores.sum()
    variancia_explicada = autovalores / variancia_total

    # Matriz de autovetores a matriz dos dados originais, escolhi 4 componentes principais
    pca = PCA(n_components=4)
    x_pca = pca.fit_transform(df) 
    """""
    print("Matriz de dados projetada no novo espaço: ") 
    print(x_pca)
    print("Variância explicada por cada componente principal: ")
    print(variancia_explicada)
    """
    column_names = df.columns
    components = pca.components_ # Autovetores (componentes principais)

    # Crie gráficos de barras para visualizar a contribuição de cada característica em cada componente principal
    for i in range(len(components)):
        plt.bar(column_names, components[i])
        plt.title(f'Contribuição das Características na Componente Principal {i + 1}')
        plt.xlabel('Características')
        plt.ylabel('Contribuição')
        plt.xticks(rotation=45)
        plt.show()

    # Crie gráficos de dispersão para visualizar as amostras no novo espaço de componentes principais
    plt.figure(figsize=(8, 6))
    plt.scatter(x_pca[:, 0], x_pca[:, 1], label='Componente Principal 1 vs. Componente Principal 2')
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.legend()
    plt.show()



pca(data)
