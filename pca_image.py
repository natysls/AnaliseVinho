import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from skimage import io

imagem = io.imread('digital.jpeg')

def recontruir_imagem():
    imagem_achatada = imagem.reshape(-1, imagem.shape[-1]) # Para calcular matriz de covariancia
    cov_matrix = np.cov(imagem_achatada, rowvar=False)
    autovalores, autovetores = np.linalg.eig(cov_matrix)
    variancia_explicada = np.cumsum(autovalores) / np.sum(autovalores)
    print("Autovalores:")
    print(autovalores)
    print("Autovetores:")
    print(autovetores)
    print("Variância Explicada: ")
    print(variancia_explicada)

    plt.figure(figsize=(10, 5))
    plt.subplot(2, 5, 1)
    plt.imshow(imagem, cmap='gray')
    plt.title('Imagem Original')

    num_componentes_plot = [1, 2, 3]

    for i, n in enumerate(num_componentes_plot):
        pca = PCA(n_components=n)
        imagem_reduzida = pca.fit_transform(imagem_achatada)
        imagem_reconstruida = pca.inverse_transform(imagem_reduzida)
        imagem_reconstruida = imagem_reconstruida.reshape(imagem.shape)
        
        plt.subplot(2, 5, i + 6)
        plt.imshow(imagem_reconstruida, cmap='gray')
        plt.title(f'{n} Componentes')

    plt.show()

recontruir_imagem()