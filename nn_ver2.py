from math import exp
from random import seed
from random import random
import numpy as np


class Data:
    def __init__(self):
        data = np.loadtxt("data.csv", delimiter=",")
        np.random.shuffle(data)
        self.data = [[e for e in row] for row in data]
        for row in self.data:
            if row[-1] == -1.0:
                row[-1] = 0
            elif row[-1] == 1.0:
                row[-1] = 1

        # print(self.data)

        self.train_size = int(0.7 * len(self.data))
        self.val_size = int(0.15 * len(self.data))
        self.test_size = len(self.data) - self.train_size - self.val_size

        self.trainData = self.get_train_data()

    def get_train_data(self):
        return self.data[:self.train_size]

    def get_val_data(self):
        return self.data[self.train_size:self.train_size + self.val_size]

    def get_test_data(self):
        return self.data[self.train_size + self.val_size:]


# Initialize a network
def initialize_network(n_inputs, n_hidden, n_outputs):
    network = list()
    hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
    network.append(hidden_layer)
    output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
    network.append(output_layer)
    print_network(network)
    return network


# Calculate neuron activation for an input
def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights)-1):
        activation += weights[i] * inputs[i]
    return activation


# Transfer neuron activation
def transfer(activation):
    return 1.0 / (1.0 + exp(-activation))


# Forward propagate input to a network output
def forward_propagate(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = transfer(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs


# Calculate the derivative of an neuron output
def transfer_derivative(output):
    return output * (1.0 - output)


# Backpropagation error and store in neurons
def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network)-1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(neuron['output'] - expected[j])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])


# Update network weights with error
def update_weights(network, row, learning_rate):
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] -= learning_rate * neuron['delta'] * inputs[j]
            neuron['weights'][-1] -= learning_rate * neuron['delta']


# Train a network for a fixed number of epochs
def train_network(network, dataset, learning_rate, iterations, n_outputs):
    print(n_outputs)
    for epoch in range(iterations):
        sum_error = 0
        for row in dataset:
            outputs = forward_propagate(network, row)
            expected = [0 for i in range(n_outputs)]
            expected[row[-1]] = 1
            sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
            backward_propagate_error(network, expected)
            update_weights(network, row, learning_rate)
        print('>iteration=%d, lrate=%.3f, error=%.3f' % (epoch, learning_rate, sum_error))


def predict(network, row):
    outputs = forward_propagate(network, row)
    return outputs.index(max(outputs))


def print_network(network):
    for layer in network:
        print(layer)


# Test training backprop algorithm
seed(1)

data = Data()
dataset = data.get_train_data()
datatest = data.get_test_data()

# Train
n_inputs = len(dataset[0]) - 1
n_outputs = len(set([row[-1] for row in dataset]))
network = initialize_network(n_inputs, 2, n_outputs)
train_network(network, dataset, 0.5, 20, n_outputs)

network_after_train = []
for layer in network:
    network_after_train.append(layer)


print("START TESTING")
passed = 0
for row in datatest:
    prediction = predict(network_after_train, row)
    if row[-1] == prediction:
        passed += 1
    print('Expected=%d, Got=%d' % (row[-1], prediction))

print("Passed", (passed/len(datatest))*100, "%", "test")
