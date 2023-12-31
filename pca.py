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

    x_centered = cov_matrix - cov_matrix.mean(axis=0)
    u, s, vt = np.linalg.svd(x_centered)
    c1 = vt.T[:, 0]
    c2 = vt.T[:, 1]

    w2 = vt.T[:, :2]
    x2d = x_centered.dot(w2)
    print("Redução da dimensão em 2: ")
    print(x2d)

    # Autovalores e Autovetores da matriz de covariancia
    autovalores, autovetores = np.linalg.eig(cov_matrix)
    print("Autovalores: ")
    print(autovalores)
    print("Autovetores: ")
    print(autovetores)

    # Matriz de autovetores a matriz dos dados originais, escolhi 2 componentes principais
    pca = PCA(n_components=2)
    x_pca = pca.fit_transform(cov_matrix) 
    print("Matriz de dados projetada no novo espaço: ") 
    print(x_pca)

    # Variancia explicada de acordo com a quantidade de autovetores
    variance_explained = []
    for i in autovalores:
        variance_explained.append((i/sum(autovalores))*100)
    print("Variância explicada de acordo com a quantidade de autovetores: ")
    print(variance_explained)

    var_exp = pca.explained_variance_ratio_
    print("Variância explicada por cada componente principal: ")
    print(var_exp)

    pca2 = PCA()
    pca2.fit(cov_matrix)
    cumsum = np.cumsum(var_exp)
    d = np.argmax(cumsum >= 0.95) + 1
    pca2 = PCA(n_components=d)
    x_pca2 = pca2.fit_transform(cov_matrix) 
    print("Matriz de dados projetada no novo espaço de dimensão: ", d) 
    print(x_pca2)

    # Gráficos de dispersão para visualizar as amostras no novo espaço de componentes principais
    plt.figure(figsize=(8, 6))
    plt.scatter(x_pca[:, 0], x_pca[:, 1], label='Componente Principal 1 vs. Componente Principal 2')
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.legend()
    plt.show()

pca(data)
