import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering, KMeans

def my_kmeans(X, K, max_iters=100):
    centroids = X[:K]

    for _ in range(max_iters):
        # Assign each data point to the nearest centroid
        expanded_x = X[:, np.newaxis]
        euc_dist = np.linalg.norm(expanded_x - centroids, axis=2)
        labels = np.argmin(euc_dist, axis=1)

        # Update the centroids based on the assigned point
        new_centroids = np.array([X[labels == k].mean(axis=0) for k in range(K)])

        # If the centroids did not change, stop iterating
        if np.all(centroids == new_centroids):
            break

        centroids = new_centroids

    return labels, centroids

# Load the Iris dataset
X = load_iris().data
K = 3
labels, centroids = my_kmeans(X, K)
print("Labels without using sklearn(K-means):", labels)
print("Centroids (K-means) without using sklearn:", centroids)

# Apply hierarchical clustering using Agglomerative clustering
agglomerative = AgglomerativeClustering(n_clusters=3)
labels_agg = agglomerative.fit_predict(X)
print("Labels (Agglomerative):", labels_agg)

# Plotting K-means results
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', color='red', s=200)
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('K-means Clustering of Iris Dataset')

# Plotting Agglomerative clustering results
plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=labels_agg)
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('Agglomerative Clustering of Iris Dataset')

plt.tight_layout()
plt.show()

# K Means using SK Learn 
K = 3
kmeans_sklearn = KMeans(n_clusters=K, random_state=0)
labels_sklearn = kmeans_sklearn.fit_predict(X)
centroids_sklearn = kmeans_sklearn.cluster_centers_
print("Scikit-learn K-means Labels:", labels_sklearn)
print("Scikit-learn K-means Centroids:", centroids_sklearn)

# Plotting scikit-learn K-means results
plt.figure(figsize=(12, 5))
plt.scatter(X[:, 0], X[:, 1], c=labels_sklearn)
plt.scatter(centroids_sklearn[:, 0], centroids_sklearn[:, 1], marker='x', color='red', s=200)
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('Scikit-learn K-means Clustering of Iris Dataset')
plt.show()
