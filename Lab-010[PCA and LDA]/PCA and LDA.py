import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

data = load_iris()
X = data.data
y = data.target

pca = SklearnPCA(n_components=2)
X_pca = pca.fit_transform(X)

print("Shape of Data:", X.shape)
print("Shape of PCA transformed Data:", X_pca.shape)

pc1 = X_pca[:, 0]
pc2 = X_pca[:, 1]

lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X, y)

print("Shape of Data:", X.shape)
print("Shape of LDA transformed Data:", X_lda.shape)

ld1 = X_lda[:, 0]
ld2 = X_lda[:, 1]

plt.figure()
plt.scatter(pc1, pc2, c=y, cmap="jet")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA of Iris Dataset (sklearn Implementation)")
plt.colorbar()
plt.show()

plt.figure()
plt.scatter(ld1, ld2, c=y, cmap="jet")
plt.xlabel("Linear Discriminant 1")
plt.ylabel("Linear Discriminant 2")
plt.title("LDA of Iris Dataset (sklearn Implementation)")
plt.colorbar()
plt.show()

"""
Output:
Shape of Data: (150, 4)
Shape of PCA transformed Data: (150, 2)
Shape of Data: (150, 4)
Shape of LDA transformed Data: (150, 2)
"""
