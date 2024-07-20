import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("weather_forecast.csv")
print(df.head())

df = pd.get_dummies(df, drop_first=True)

X = df.drop('Play_Yes', axis=1)  # Modify the column name here if needed
y = df['Play_Yes']  # Modify the column name here if needed

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def train_and_evaluate(X_train, X_test, y_train, y_test, criterion, algo_name):
    clf = DecisionTreeClassifier(criterion=criterion, random_state=42)
    clf.fit(X_train, y_train)
    
    plt.figure(figsize=(12, 8))
    plot_tree(clf, filled=True, feature_names=X.columns, class_names=['No', 'Yes'])
    plt.title(f"{algo_name} Decision Tree")
    plt.show()

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"{algo_name} Algorithm Results:")
    print(f"Accuracy: {accuracy}")
    print(f"Classification Report:\n{report}")

train_and_evaluate(X_train, X_test, y_train, y_test, criterion='entropy', algo_name='ID3')

train_and_evaluate(X_train, X_test, y_train, y_test, criterion='gini', algo_name='CART')