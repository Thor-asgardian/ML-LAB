import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD

# Function to create a model for the AND-NOT boolean function
def create_andnot_model():
    model = Sequential()
    model.add(Dense(2, input_dim=2, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=SGD(lr=0.1), metrics=['accuracy'])
    return model

# Function to create a model for the XOR boolean function
def create_xor_model():
    model = Sequential()
    model.add(Dense(2, input_dim=2, activation='relu'))
    model.add(Dense(2, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=SGD(lr=0.1), metrics=['accuracy'])
    return model

# Data for AND-NOT
X_andnot = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_andnot = np.array([0, 0, 1, 0])

# Data for XOR
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([0, 1, 1, 0])

# Train AND-NOT model
andnot_model = create_andnot_model()
andnot_model.fit(X_andnot, y_andnot, epochs=2000, verbose=0)

# Train XOR model
xor_model = create_xor_model()
xor_model.fit(X_xor, y_xor, epochs=2000, verbose=0)

# Evaluate models
andnot_loss, andnot_acc = andnot_model.evaluate(X_andnot, y_andnot, verbose=0)
xor_loss, xor_acc = xor_model.evaluate(X_xor, y_xor, verbose=0)

print(f"AND-NOT Model Accuracy: {andnot_acc * 100:.2f}%")
print(f"XOR Model Accuracy: {xor_acc * 100:.2f}%")

# Predict AND-NOT
print("AND-NOT Predictions:")
print(andnot_model.predict(X_andnot).round())

# Predict XOR
print("XOR Predictions:")
print(xor_model.predict(X_xor).round())
