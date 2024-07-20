import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def create_andnot_model():
    model = {'weights': [np.random.rand(2, 1)], 'biases': [np.random.rand(1)]}
    return model

def create_xor_model():
    model = {'weights': [np.random.rand(2, 2), np.random.rand(2, 1)], 'biases': [np.random.rand(2), np.random.rand(1)]}
    return model

def forward_propagation(model, X):
    layer_input = X
    for i in range(len(model['weights'])):
        layer_output = sigmoid(np.dot(layer_input, model['weights'][i]) + model['biases'][i])
        layer_input = layer_output
    return layer_output

def train_model(model, X, y, epochs=2000, learning_rate=0.1):
    for epoch in range(epochs):
        output = forward_propagation(model, X)

        error = y - output
        d_output = error * sigmoid_derivative(output)

        for i in range(len(model['weights']) - 1, -1, -1):
            model['weights'][i] += learning_rate * np.dot(X if i == 0 else forward_output[i-1].T, d_output)
            model['biases'][i] += learning_rate * np.sum(d_output, axis=0)
            d_output = np.dot(d_output, model['weights'][i].T) * sigmoid_derivative(forward_output[i - 1])

X_andnot = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_andnot = np.array([[0], [0], [1], [0]])

X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([[0], [1], [1], [0]])

andnot_model = create_andnot_model()
train_model(andnot_model, X_andnot, y_andnot)

xor_model = create_xor_model()
train_model(xor_model, X_xor, y_xor)

andnot_predictions = np.round(forward_propagation(andnot_model, X_andnot))
xor_predictions = np.round(forward_propagation(xor_model, X_xor))

andnot_acc = np.mean(andnot_predictions == y_andnot) * 100
xor_acc = np.mean(xor_predictions == y_xor) * 100

print(f"AND-NOT Model Accuracy: {andnot_acc:.2f}%")
print(f"XOR Model Accuracy: {xor_acc:.2f}%")

print("AND-NOT Predictions:")
print(andnot_predictions)

print("XOR Predictions:")
print(xor_predictions)