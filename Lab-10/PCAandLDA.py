import numpy as np
from sklearn.datasets import load_iris

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.mean = None
        self.components = None

    def fit(self, X):
        
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        covariance_matrix = np.cov(X_centered, rowvar=False)

        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        self.components = eigenvectors[:, :self.n_components]

    def transform(self, X):
        X_centered = X - self.mean
        return np.dot(X_centered, self.components)

class LDA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.means = None
        self.scalings = None

    def fit(self, X, y):
        n_features = X.shape[1]
        class_labels = np.unique(y)

        mean_vectors = []
        for label in class_labels:
            mean_vectors.append(np.mean(X[y == label], axis=0))
        self.means = mean_vectors

        S_W = np.zeros((n_features, n_features))
        for label, mean_vec in zip(class_labels, mean_vectors):
            class_scatter = np.cov(X[y == label].T)
            S_W += class_scatter * (X[y == label].shape[0] - 1)

        overall_mean = np.mean(X, axis=0)
        S_B = np.zeros((n_features, n_features))
        for i, mean_vec in enumerate(mean_vectors):
            n = X[y == class_labels[i]].shape[0]
            mean_vec = mean_vec.reshape(n_features, 1)
            overall_mean = overall_mean.reshape(n_features, 1)
            S_B += n * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)

        eigenvalues, eigenvectors = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))

        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        self.scalings = eigenvectors[:, :self.n_components]

    def transform(self, X):
        return np.dot(X, self.scalings)

X, y = load_iris(return_X_y=True)

pca = PCA(n_components=2)
pca.fit(X)
X_pca = pca.transform(X)

lda = LDA(n_components=2)
lda.fit(X, y)
X_lda = lda.transform(X)

print("PCA Result:\n", X_pca[:5])
print("LDA Result:\n", X_lda[:5])


"""
Output:
PCA Result:
 [[-2.68412563 -0.31939725]
 [-2.71414169  0.17700123] 
 [-2.88899057  0.14494943] 
 [-2.74534286  0.31829898] 
 [-2.72871654 -0.32675451]]
LDA Result:
 [[-1.49920971 -1.88675441]
 [-1.2643595  -1.59214275]
 [-1.35525305 -1.73341462]
 [-1.18495616 -1.62358806]
 [-1.5169559  -1.94476227]]
"""
