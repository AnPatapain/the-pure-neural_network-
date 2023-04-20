import csv

import numpy as np

class Data:
    def __init__(self):
        with open("iris.csv", 'r') as iris_data:
            irises = list(csv.reader(iris_data))
        np.random.shuffle(irises)
        print(irises)

        self.data = irises
        for row in self.data:
            if row[-1] == 'Iris-setosa':
                row[-1] = 0
            elif row[-1] == 'Iris-versicolor':
                row[-1] = 1
            elif row[-1] == 'Iris-virginica':
                row[-1] = 2
            for i in range(len(row) - 1):
                row[i] = float(row[i])

        # print(self.data)

        self.train_size = int(0.7 * len(self.data))
        self.val_size = int(0.15 * len(self.data))
        self.test_size = len(self.data) - self.train_size - self.val_size

        self.trainData = self.get_train_data()

    def get_X(self, data):
        return [row[:len(row)-1] for row in data]

    def get_Y(self, data):
        Y = []
        for row in data:
            y = [0, 0, 0]
            y[row[-1]] = 1
            Y.append(y)
        return Y
    def get_train_data(self):
        return self.data[:self.train_size]

    def get_val_data(self):
        return self.data[self.train_size:self.train_size + self.val_size]

    def get_test_data(self):
        return self.data[self.train_size + self.val_size:]

# Define the sigmoid and softmax functions and their derivatives
# Define the sigmoid and softmax functions and their derivatives
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def softmax(x):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def softmax_derivative(x):
    s = x.reshape(-1,1)
    return np.diagflat(s) - np.dot(s, s.T)

# Define the train function
def train(X, y, num_epochs=15000, learning_rate=0.1):
    # Initialize weights randomly
    input_size = X.shape[1]
    hidden_size = 2
    output_size = 3
    W1 = np.random.randn(input_size, hidden_size)
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, output_size)
    b2 = np.zeros((1, output_size))

    # Training loop
    for epoch in range(num_epochs):
        # Feedforward
        hidden_layer_input = np.dot(X, W1) + b1
        hidden_layer_output = sigmoid(hidden_layer_input)
        output_layer_input = np.dot(hidden_layer_output, W2) + b2
        output = softmax(output_layer_input)

        # Backpropagation
        d_output = output - y
        d_hidden = np.dot(d_output, W2.T) * sigmoid_derivative(hidden_layer_output)

        # Update weights
        W2 -= learning_rate * np.dot(hidden_layer_output.T, d_output)
        b2 -= learning_rate * np.sum(d_output, axis=0, keepdims=True)
        W1 -= learning_rate * np.dot(X.T, d_hidden)
        b1 -= learning_rate * np.sum(d_hidden, axis=0, keepdims=True)

    # Return the learned weights
    return (W1, b1, W2, b2)

# Define the data and labels
data = Data()
data_train = data.get_train_data()
data_test = data.get_test_data()
print(np.array(data.get_X(data_train)))
print(np.array(data.get_Y(data_train)))

# print(np.array(data.get_X(data_test)))
# print(np.array(data.get_Y(data_test)))

# X_train = np.array([[6.7, 3.1, 5.6, 2.4], [5.0, 3.0, 1.6, 0.2], [4.4, 3.2, 1.3, 0.2], [5.9, 3.0, 4.2, 1.5], [6.6, 3.0, 4.4, 1.4], [4.9, 2.4, 3.3, 1.0], [5.6, 2.8, 4.9, 2.0], [5.8, 2.8, 5.1, 2.4]])
# y_train = np.array([[0, 0, 1], [1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]])

X_train = np.array(data.get_X(data_train))
Y_train = np.array(data.get_Y(data_train))

# Train the network
W1, b1, W2, b2 = train(X_train, Y_train)

# Define a test input
for x_test in X_train:
    # Feedforward to get the output
    hidden_layer_input = np.dot(x_test, W1) + b1
    hidden_layer_output = sigmoid(hidden_layer_input)
    output_layer_input = np.dot(hidden_layer_output, W2) + b2
    output = softmax(output_layer_input)
    print("Predicted class:", np.argmax(output))  # Output: 1



