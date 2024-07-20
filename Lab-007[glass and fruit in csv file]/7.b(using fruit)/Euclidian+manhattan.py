
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

def manhattan_distance(x1, x2):
    return np.sum(np.abs(x1 - x2))

class KNN:
    def __init__(self, k, distance_metric):
        self.k = k
        self.distance_metric = distance_metric

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return predictions

    def _predict(self, x):
        distances = [self.distance_metric(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

df = pd.read_csv('fruits.csv')
y = df['fruit_label'].values
X = df[['mass', 'width', 'height', 'color_score']].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=45)

# Without using scikit-learn with Euclidean distance
clf_euclidean_custom = KNN(k=5, distance_metric=euclidean_distance)
clf_euclidean_custom.fit(X_train, y_train)
predictions_euclidean_custom = clf_euclidean_custom.predict(X_test)

accuracy_euclidean_custom = np.sum(predictions_euclidean_custom == y_test) / len(y_test)
print("Accuracy with Euclidean distance (without sklearn):", accuracy_euclidean_custom)

# Without using scikit-learn with Manhattan distance
clf_manhattan_custom = KNN(k=5, distance_metric=manhattan_distance)
clf_manhattan_custom.fit(X_train, y_train)
predictions_manhattan_custom = clf_manhattan_custom.predict(X_test)

accuracy_manhattan_custom = np.sum(predictions_manhattan_custom == y_test) / len(y_test)
print("Accuracy with Manhattan distance (without sklearn):", accuracy_manhattan_custom)