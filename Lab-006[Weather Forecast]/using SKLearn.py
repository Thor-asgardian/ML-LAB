import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
df = pd.read_csv("weather_forecast.csv")
print(df.head())

# One-hot encode categorical features
df = pd.get_dummies(df, drop_first=True)

# Define features and target variable
X = df.drop('Play_Yes', axis=1)  # Adjust this column name if it differs
y = df['Play_Yes']  # Adjust this column name if it differs

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Function to train and evaluate the decision tree models
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
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Classification Report:\n{report}")

# Train and evaluate ID3 algorithm
train_and_evaluate(X_train, X_test, y_train, y_test, criterion='entropy', algo_name='ID3')

# Train and evaluate CART algorithm
train_and_evaluate(X_train, X_test, y_train, y_test, criterion='gini', algo_name='CART')
