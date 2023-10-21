import pandas as pd
import numpy as np
import filter
import matplotlib.pyplot as plt
from skimage import io

data = pd.read_csv('wine.csv')
data.replace("?", pd.NA, inplace=True)
df = filter.toda_filtragem(data)
"""" # Para se usar na imagem
imagem = io.imread('digital.jpeg')
df = imagem.reshape(-1, imagem.shape[-1])
"""

autovalores_experimentos = []
autovetores_experimentos = []

cov_matrix = np.cov(df, rowvar=False)
autovalores, autovetores = np.linalg.eig(cov_matrix)
autovalores_experimentos.append(autovalores)
autovetores_experimentos.append(autovetores)

autovalores_experimentos = [np.sort(autovalores)[::-1] for autovalores in autovalores_experimentos]

plt.figure(figsize=(12, 6))

for i, autovalores in enumerate(autovalores_experimentos):
    variância_explicada_cumulativa = np.cumsum(autovalores) / np.sum(autovalores)
    
    plt.subplot(1, len(autovalores_experimentos), i + 1)
    plt.plot(range(1, len(autovalores) + 1), variância_explicada_cumulativa, marker='o')
    plt.xlabel('Número de Componentes Principais')
    plt.ylabel('Variância Explicada Cumulativa')
    plt.title(f'Experimento {i+1}')

plt.tight_layout()
plt.show()

