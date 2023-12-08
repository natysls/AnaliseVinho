import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import filter

data = pd.read_csv('wine.csv')
data.replace("?", pd.NA, inplace=True)
data = filter.toda_filtragem(data)
data = data.drop('Wine', axis=1)

# Padronizar os dados antes de aplicar o KMeans
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Reduzir a dimensionalidade para facilitar a visualização 
pca = PCA(n_components=2)
X_pca = pca.fit_transform(data_scaled)

inertia_values = []

def kmeans(n_cluster_a, n_cluster_b):
    for k in [n_cluster_a, n_cluster_b]:
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        kmeans.fit(X_pca)
        inertia = kmeans.inertia_
        print("Variância inter-cluster do k=", k, ":", inertia)
        inertia_values.append(inertia)
        
        plt.figure(figsize=(8, 6))
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans.labels_, cmap='viridis', edgecolor='k', s=50)
        plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='*', s=200, color='red')
        plt.title(f'KMeans Clustering (k={k})')
        plt.xlabel('PCA Componente 1')
        plt.ylabel('PCA Componente 2')
        plt.show()
    
    plt.plot([n_cluster_a, n_cluster_b], inertia_values, marker='o')
    plt.title('Elbow Method para Determinar o Número Ótimo de Clusters')
    plt.xlabel('Número de Clusters (k)')
    plt.ylabel('Inertia')
    plt.show()  
        
kmeans(2, 3)
