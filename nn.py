from math import exp
from random import seed
from random import random


# Initialize a network
def initialize_network(n_inputs, n_hidden, n_outputs):
    network = list()
    hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
    network.append(hidden_layer)
    output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
    network.append(output_layer)
    for layer in network:
        print(layer)
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


# Test training backprop algorithm
seed(1)

dataset = [[7.2, 3.2, 6., 1.8, 0],
           [5., 3.5, 1.3, 0.3, 1],
           [7.7, 2.8, 6.7, 2., 0],
           [6.8, 3., 5.5, 2.1, 0],
           [5.4, 3.9, 1.3, 0.4, 1],
           [5.5, 4.2, 1.4, 0.2, 1],
           [4.4, 2.9, 1.4, 0.2, 1],
           [5.3, 3.7, 1.5, 0.2, 1],
           [5., 3.4, 1.5, 0.2, 1],
           [5.1, 3.8, 1.9, 0.4, 1],
           [6.7, 3.3, 5.7, 2.1, 0],
           [5.7, 4.4, 1.5, 0.4, 1],
           [4.9, 2.5, 4.5, 1.7, 0],
           [7.1, 3., 5.9, 2.1, 0],
           [5.2, 3.5, 1.5, 0.2, 1],
           [6.3, 3.3, 6., 2.5, 0],
           [4.7, 3.2, 1.3, 0.2, 1],
           [4.8, 3.1, 1.6, 0.2, 1],
           [4.3, 3., 1.1, 0.1, 1],
           [6.5, 3., 5.2, 2., 0],
           [6., 3., 4.8, 1.8, 0],
           [4.8, 3.4, 1.9, 0.2, 1],
           [6.3, 2.8, 5.1, 1.5, 0],
           [7.6, 3., 6.6, 2.1, 0],
           [4.7, 3.2, 1.6, 0.2, 1],
           [7.7, 3., 6.1, 2.3, 0],
           [5., 3.6, 1.4, 0.2, 1],
           [6.4, 2.8, 5.6, 2.2, 0],
           [7.7, 2.6, 6.9, 2.3, 0],
           [6.3, 2.5, 5., 1.9, 0],
           [4.6, 3.4, 1.4, 0.3, 1],
           [6.3, 2.9, 5.6, 1.8, 0],
           [6.4, 3.2, 5.3, 2.3, 0],
           [6.5, 3.2, 5.1, 2., 0],
           [5.1, 3.4, 1.5, 0.2, 1],
           [7.9, 3.8, 6.4, 2., 0],
           [6.5, 3., 5.5, 1.8, 0],
           [4.6, 3.1, 1.5, 0.2, 1],
           [4.6, 3.2, 1.4, 0.2, 1],
           [5.8, 2.7, 5.1, 1.9, 0],
           [5.9, 3., 5.1, 1.8, 0],
           [4.9, 3.1, 1.5, 0.1, 1],
           [7.4, 2.8, 6.1, 1.9, 0],
           [5.4, 3.4, 1.7, 0.2, 1],
           [5., 3., 1.6, 0.2, 1],
           [5.8, 2.8, 5.1, 2.4, 0],
           [7.2, 3., 5.8, 1.6, 0],
           [6.7, 3.1, 5.6, 2.4, 0],
           [5., 3.5, 1.6, 0.6, 1],
           [5.6, 2.8, 4.9, 2., 0],
           [5.8, 4., 1.2, 0.2, 1],
           [6.3, 3.4, 5.6, 2.4, 0],
           [5.5, 3.5, 1.3, 0.2, 1],
           [5.1, 3.5, 1.4, 0.2, 1],
           [4.4, 3., 1.3, 0.2, 1],
           [5.4, 3.4, 1.5, 0.4, 1],
           [6.9, 3.2, 5.7, 2.3, 0],
           [6.9, 3.1, 5.4, 2.1, 0],
           [4.9, 3., 1.4, 0.2, 1],
           [5., 3.3, 1.4, 0.2, 1],
           [4.9, 3.1, 1.5, 0.1, 1],
           [4.6, 3.6, 1., 0.2, 1],
           [5.7, 2.5, 5., 2., 0],
           [6.1, 2.6, 5.6, 1.4, 0],
           [5.8, 2.7, 5.1, 1.9, 0],
           [6.1, 3., 4.9, 1.8, 0],
           [5.2, 3.4, 1.4, 0.2, 1],
           [6.7, 3.3, 5.7, 2.5, 0],
           [7.2, 3.6, 6.1, 2.5, 0],
           [6.3, 2.7, 4.9, 1.8, 0]]

#Train

n_inputs = len(dataset[0]) - 1
n_outputs = len(set([row[-1] for row in dataset]))
network = initialize_network(n_inputs, 2, n_outputs)
train_network(network, dataset, 0.5, 20, n_outputs)
for layer in network:
    print(layer)

#Test
# network = [[{'weights': [-1.2019650100733577, -0.9525445319277358, 2.4988110277351034, 1.2284598024946591, 0.027391828688717243], 'output': 0.9868363313455423, 'delta': -0.0005715849603749642}, {'weights': [0.5526745430621051, 0.7225054883958114, 0.8165707317078637, 0.0972162937811014, 0.0498504987971637], 'output': 0.9999361350613765, 'delta': 6.73098375892811e-07}],
#            [{'weights': [5.1963692002275526, -1.3919252390821717, -1.1049292296831363], 'output': 0.9324453254615329, 'delta': -0.0042553392389028085}, {'weights': [-5.209817197328115, 1.0937807660366863, 1.409748528783971], 'output': 0.06713684087656284, 'delta': 0.004204745800427319}]]
#
# for row in dataset:
#     prediction = predict(network, row)
#     print('Expected=%d, Got=%d' % (row[-1], prediction))