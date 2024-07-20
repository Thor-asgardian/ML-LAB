import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

iris = pd.read_csv('Iris.csv')
species = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
iris['Species'] = iris['Species'].map(species)

def calculate_mean_std_by_class(dataset):
    classes = np.unique(dataset[:, -1])
    mean_std_by_class = {}
    for label in classes:
        rows = dataset[dataset[:, -1] == label][:, :-1]
        mean_std_by_class[label] = [(np.mean(col), np.std(col)) for col in rows.T]
    return mean_std_by_class

def calculate_prob_density(x, mean, std):
    if std == 0:
        return 0
    exponent = np.exp(-((x - mean) ** 2) / (2 * std ** 2))
    return (1 / (np.sqrt(2 * np.pi) * std)) * exponent

def predict_label(mean_std_by_class, test_case):
    probabilities = {}
    for label, mean_std in mean_std_by_class.items():
        probabilities[label] = np.prod([calculate_prob_density(test_case[i], mean, std) 
                                        for i, (mean, std) in enumerate(mean_std)])
    return max(probabilities, key=probabilities.get)

def naive_bayes_classifier(training_set, test_set):
    mean_std_by_class = calculate_mean_std_by_class(training_set)
    return [predict_label(mean_std_by_class, test_case) for test_case in test_set]

X = iris[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].values
y = iris['Species'].values
data = np.column_stack((X, y))
train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)

predicted_labels = naive_bayes_classifier(train_data, test_data[:, :-1])

accuracy = accuracy_score(test_data[:, -1], predicted_labels)
print("Accuracy:", accuracy)
cm = confusion_matrix(test_data[:, -1], predicted_labels)
print("Confusion Matrix:\n", cm)