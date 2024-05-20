import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics import silhouette_score
import scipy.cluster.hierarchy as shc

# Reading the CSV file
X = pd.read_csv('glass.csv')

# Handling the missing values
X.fillna(method='ffill', inplace=True)

# Scaling the data so that all the features become comparable
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Normalizing the data so that the data approximately follows a Gaussian distribution
X_normalized = normalize(X_scaled)

# Converting the numpy array into a pandas DataFrame
X_normalized = pd.DataFrame(X_normalized)

pca = PCA(n_components=2)
X_principal = pca.fit_transform(X_normalized)
X_principal = pd.DataFrame(X_principal, columns=['P1', 'P2'])

plt.figure(figsize=(15, 10))

# Plotting dendrogram
plt.subplot(2, 2, 1)
plt.title('Dendrogram')
Z = shc.linkage(X_principal, method='ward')
Dendrogram = shc.dendrogram(Z)

# Calculating silhouette scores after clustering
k = range(2, 6)
silhouette_scores = []

for i in k:
    ac = AgglomerativeClustering(n_clusters=i)
    cluster_labels = ac.fit_predict(X_principal)
    silhouette_scores.append(silhouette_score(X_principal, cluster_labels))

# Plotting silhouette scores
plt.subplot(2, 2, 4)
plt.bar(k, silhouette_scores)
plt.xlabel('Number of clusters', fontsize=12)
plt.ylabel('Silhouette Score', fontsize=12)
plt.title('Silhouette Score for Different Clusters')

plt.tight_layout()
plt.show()