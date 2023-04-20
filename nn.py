import csv

import numpy as np
import math
from random import random


def sigmoid(x): return 1 / (1 + math.exp(-x))


def sigmoid_derivative(sigmoid_output): return sigmoid_output * (1 - sigmoid_output)


def softmax(z):
    """
    :param z: input vector
    :return: probability vector of the input vector
    """
    exp_z = np.exp(z)
    sum_ = exp_z.sum()
    softmax_z = np.round(exp_z/sum_,3)
    return softmax_z


def transfer(weight, row):
    """
    Calculate the transfer value for one neuron.
    :param weight: Weights that associate the neuron to the neurons at previous layer
    :param row: the inputs that feed into the neuron
    :return: the transfer value for the neuron
    """
    bias = weight[-1]
    for i in range(len(weight) - 1):
        bias += weight[i] * row[i]
    return bias


def loss(expected, output):
    """
    :param expected: the expect label value
    :param output: the output that returned by the network
    :return: the loss value
    """
    return (expected - output) ** 2


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

        self.train_size = int(0.7 * len(self.data))
        self.val_size = int(0.15 * len(self.data))
        self.test_size = len(self.data) - self.train_size - self.val_size

        self.trainData = self.get_train_data()

    def get_train_data(self):
        """
        :return: Data for model training
        """
        return self.data[:self.train_size]

    def get_val_data(self):
        """
        :return: Data for model validation while training the model
        """
        return self.data[self.train_size:self.train_size + self.val_size]

    def get_test_data(self):
        """
        :return: Data to test the model after the model training is finish
        """
        return self.data[self.train_size + self.val_size:]


class Neurone:
    """
    The Neurone is modeled as the node with the weights that associates with the nodes in the previous layer
    The Neurone has input, output and delta (gradient) which represents the change that the weight need to be modified
    """
    def __init__(self, weights):
        self.weights = weights
        self.input = None
        self.output = None
        self.delta = None


class Layer:
    """
    The Layer is modeled as the list of neurones. For example in output layer the number of nodes is 3.
    """
    def __init__(self, neurones):
        self.neurones = neurones


class NeuralNetwork:
    """
    The Neural Network is modeled as the list of the layers.
    """
    def __init__(self, data, data_validation, n_input, n_hidden, n_output):
        # Data that feed network. n_input, n_hidden, n_output is the number of nodes in input, hidden and output layer
        self.data = data
        self.data_validation = data_validation
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output

        # Network is array of layer. We use two layers. Hidden layer and output layer
        self.hidden_layer = Layer([Neurone([random() for i in range(n_input + 1)]) for i in range(n_hidden)])
        self.output_layer = Layer([Neurone([random() for i in range(n_hidden + 1)]) for i in range(n_output)])
        self.layers = [self.hidden_layer, self.output_layer]

        # Activation function and its derivative for the hidden layer. Easy to change between sigmoid and tanh
        self.activation = sigmoid
        self.activation_derivative = sigmoid_derivative

    def feed_forward(self, row):
        """
        :param row: one row of inputs that feed into the network
        :return: the output array at output layer. The shape of this array is 1 x 3 [o1, o2, o3]
        """
        hidden_layer_outputs = []
        for neurone in self.hidden_layer.neurones:
            output = self.activation(transfer(neurone.weights, row))
            neurone.output = output
            hidden_layer_outputs.append(output)

        transfer_outs = []
        for neurone in self.output_layer.neurones:
            transfer_value = transfer(neurone.weights, hidden_layer_outputs)
            transfer_outs.append(transfer_value)
        softmax_outs = softmax(transfer_outs)
        for i in range(len(softmax(transfer_outs))):
            self.output_layer.neurones[i].output = softmax_outs[i]
        return softmax_outs

    def back_propagation(self, expected):
        """
        Calculate the error at output layer and send backwardly to the hidden layer
        Calculate the delta for each neurone for each layer in the network
        :param expected: the expected output for one row of inputs that feed into the network
        """
        # We go backwardly, from output layer back to hidden layer to fine-tune w for both layers
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            errors = []

            # layer is output layer
            if i == len(self.layers) - 1:
                for j in range(len(self.output_layer.neurones)):
                    error = self.output_layer.neurones[j].output - expected[j]
                    self.output_layer.neurones[j].delta = error


            # layer is hidden layer
            else:
                for j in range(len(self.hidden_layer.neurones)):
                    error = 0.0
                    for neurone in self.output_layer.neurones:
                        error += neurone.weights[j] * neurone.delta
                        errors.append(error)

                # Calculate delta for each neurone in layer
                for j in range(len(self.hidden_layer.neurones)):
                    self.hidden_layer.neurones[j].delta = errors[j] * \
                                                          self.activation_derivative(self.hidden_layer.neurones[j].output)

    def update_weight(self, row, learning_rate):
        """
        :param row: the row that caused error
        :param learning_rate: How much to change the weight
        :return: None
        """
        for layer in self.layers:
            inputs = row[:-1]
            if layer is self.output_layer:
                inputs = [neurone.output for neurone in self.hidden_layer.neurones]

            for neurone in layer.neurones:
                for j in range(len(inputs)):
                    neurone.weights[j] -= learning_rate * neurone.delta * inputs[j]
                # update the bias
                neurone.weights[-1] -= learning_rate * neurone.delta

    def train(self, learning_rate, iterations):
        """
        :param learning_rate: How much change the weight if we had already the delta for this weight
        :param iterations: How many times we iterate the process feed data and let the network modify the weight.
        :return: The network with the weight is modified
        """
        print("START TRAINING")
        for epoch in range(iterations):
            # Train
            sum_error = 0
            for row in self.data:
                outputs = self.feed_forward(row)
                # print(outputs)
                # Hot encode the expected to the form [1 0 0] [0 1 0] [0 0 1]
                expected = [0 for i in range(self.n_output)]
                expected[row[-1]] = 1
                sum_error += sum([loss(expected[i], outputs[i]) for i in range(len(expected))])
                self.back_propagation(expected)
                self.update_weight(row, learning_rate)
            if epoch % 500 == 0:
                print('>iteration=%d, learning_rate=%.3f, error=%.3f' % (epoch, learning_rate, sum_error))
                # Validation to monitor the performance of model
                if self.data_validation is not None:
                    passed = 0
                    for row in self.data_validation:
                        prediction = self.predict(row)
                        if row[-1] == prediction:
                            passed += 1
                    print("Passed", (passed/len(self.data_validation))*100, "%", "test")

    def predict(self, row):
        """
        :param row: the input row
        :return: the predicted class for the row
        """
        outputs = self.feed_forward(row)
        return np.argmax(outputs)

    def print_network(self, is_trained=False):
        """
        Print the network
        :param is_trained: to know whether the network is trained or not
        """
        if not is_trained:
            print("--------------------------------- LE NEURAL NETWORK MODEL ----------------------------")
        else:
            print("------------------------- LE NEURAL NETWORK MODEL AFTER TRAINED ------------------------")
        print("hidden layer")
        for neuron in self.hidden_layer.neurones:
            print('Neuron {', 'weight=', neuron.weights, 'output=', neuron.output, 'delta=', neuron.delta, '}')

        print("output layer")
        for neuron in self.output_layer.neurones:
            print('Neuron {', 'weight=', neuron.weights, 'output=', neuron.output, 'delta=', neuron.delta, '}')

        print("-------------------------------------------------------------------------------------------------------")


def main():
    data = Data()
    data_train = data.get_train_data()
    data_validation = data.get_val_data()
    data_test = data.get_test_data()
    print('data train', data_train)
    print('data validation', data_validation)
    print('data test', data_test)

    # Train
    nn = NeuralNetwork(data_train, data_validation, n_input=len(data_train[0])-1, n_hidden=10, n_output=3)
    nn.print_network(is_trained=False)
    nn.train(learning_rate=0.1, iterations=10000)

    print("FINISH TRAINING MODEL")
    nn.print_network(is_trained=True)

    # Test
    print("START TESTING")
    passed = 0
    for row in data_test:
        prediction = nn.predict(row)
        if row[-1] == prediction:
            passed += 1
        print('Expected=%d, Got=%d' % (row[-1], prediction))

    print("Passed", (passed/len(data_test))*100, "%", "test")


main()
