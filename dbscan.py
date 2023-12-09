import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score, adjusted_rand_score, adjusted_mutual_info_score, silhouette_score
import filter
import warnings

data = pd.read_csv('wine.csv')
data.replace("?", pd.NA, inplace=True)
data = filter.toda_filtragem(data)
y_true = data['Wine']
data = data.drop('Wine', axis=1)

X = StandardScaler().fit_transform(data)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# eps (distância relativa à vizinhança)
eps_values = [0.3, 0.4]

# min_samples (quantidade de amostras em uma vizinhança)
min_samples_values = [5, 6]

warnings.filterwarnings("ignore", category=UserWarning)

fig, axes = plt.subplots(len(eps_values), len(min_samples_values), figsize=(12, 8), sharex=True, sharey=True)

for i, eps in enumerate(eps_values):
    for j, min_samples in enumerate(min_samples_values):

        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        y_pred = dbscan.fit_predict(X_pca)

        print("EPS:", eps, "e Min Samples:", min_samples)

        axes[i, j].scatter(X_pca[:, 0], X_pca[:, 1], c=y_pred, cmap='viridis', s=50, edgecolor='k')
        axes[i, j].scatter(dbscan.components_[:, 0], dbscan.components_[:, 1], marker='*', s=20, color='red')

        axes[i, j].set_title(f'DBSCAN (eps={eps}, min_samples={min_samples})')

        silhouette_avg = silhouette_score(X_pca, y_pred)
        print(f'Silhouette Score: {silhouette_avg}')

        filter.validacao(y_true, y_pred)

plt.tight_layout()
plt.show()
