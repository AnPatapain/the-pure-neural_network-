import numpy as np
import math
from random import seed, random


def sigmoid(x): return 1 / (1 + math.exp(-x))


def tanh(x): return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))


def tanh_derivative(tanh_output): return 1 - tanh_output ** 2


def sigmoid_derivative(sigmoid_output): return sigmoid_output * (1 - sigmoid_output)


def transfer(weight, row):
    bias = weight[-1]
    for i in range(len(weight) - 1):
        bias += weight[i] * row[i]
    return bias


def loss(expected, output):
    return (expected - output)**2


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


class Neurone:
    def __init__(self, weight):
        self.weight = weight
        self.output = None
        self.delta = None

    def set_output(self, output): self.output = output

    def set_delta(self, delta): self.delta = delta


class Layer:
    def __init__(self, neurones):
        self.neurones = neurones


class NeuralNetwork:
    def __init__(self, data, data_validation, n_input, n_hidden, n_output):
        # Data that feed network. n_input, n_hidden, n_output is the number of nodes in input, hidden and output layer
        self.data = data
        self.data_validation = data_validation
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output

        # Network is array of layer. We use two layers. Hidden layer and output layer
        self.hidden_layer = Layer([Neurone([random() for i in range(n_input+1)]) for i in range(n_hidden)])
        self.output_layer = Layer([Neurone([random() for i in range(n_hidden+1)]) for i in range(n_output)])
        self.layers = [self.hidden_layer, self.output_layer]

        # Activation function and its derivative. Easy to change between sigmoid and tanh
        self.activation = sigmoid
        self.activation_derivative = sigmoid_derivative

    def feed_forward(self, row):
        # self.layer1 contains 2 neurones of hidden layer and their output
        hidden_layer_outputs = []
        for neurone in self.hidden_layer.neurones:
            output = self.activation(transfer(neurone.weight, row))
            neurone.set_output(output)
            hidden_layer_outputs.append(output)

        # self.output contains 2 neurones of output layer and their output
        for neurone in self.output_layer.neurones:
            neurone.set_output(self.activation(transfer(neurone.weight, hidden_layer_outputs)))

        return [neurone.output for neurone in self.output_layer.neurones]

    def back_propagation(self, expected):
        # We go backwardly, from output layer back to hidden layer to fine-tune w for both layers
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            errors = []

            # layer is output layer
            if i == len(self.layers)-1:
                for j in range(len(self.output_layer.neurones)):
                    errors.append(self.output_layer.neurones[j].output - expected[j])

            # layer is hidden layer
            else:
                for j in range(len(self.hidden_layer.neurones)):
                    error = 0.0
                    for neurone in self.output_layer.neurones:
                        error += neurone.weight[j] * neurone.delta
                        errors.append(error)

            # Calculate delta for each neurone in layer
            for j in range(len(layer.neurones)):
                layer.neurones[j].delta = errors[j] * self.activation_derivative(layer.neurones[j].output)

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
                    neurone.weight[j] -= learning_rate * neurone.delta * inputs[j]
                # update the bias
                neurone.weight[-1] -= learning_rate * neurone.delta

    def train(self, learning_rate, iterations):
        print("START TRAINING")
        for epoch in range(iterations):
            # Train
            sum_error = 0
            for row in self.data:
                outputs = self.feed_forward(row)
                expected = [0 for i in range(self.n_output)]
                expected[row[-1]] = 1
                sum_error += sum([loss(expected[i], outputs[i]) for i in range(len(expected))])
                self.back_propagation(expected)
                self.update_weight(row, learning_rate)
            print('>iteration=%d, learning_rate=%.3f, error=%.3f' % (epoch, learning_rate, sum_error))

            # Validation to monitor the performance of model
            if self.data_validation is not None:
                validation_error = 0
                for row in self.data_validation:
                    output_validation = self.feed_forward(row)
                    expected = [0 for i in range(self.n_output)]
                    expected[row[-1]] = 1
                    validation_error += sum([loss(expected[i], output_validation[i]) for i in range(len(expected))])
                print('>validation_error=%.3f' % validation_error)

    def predict(self, row):
        outputs = self.feed_forward(row)
        return outputs.index(max(outputs))

    def print_network(self, is_trained=False):
        if not is_trained:
            print("--------------------------------- LE NEURAL NETWORK MODEL ----------------------------")
        else:
            print("------------------------- LE NEURAL NETWORK MODEL AFTER TRAINED ------------------------")
        print("hidden layer")
        for neuron in self.hidden_layer.neurones:
            print('Neuron {', 'weight=', neuron.weight, 'output=', neuron.output, 'delta=', neuron.delta, '}')

        print("output layer")
        for neuron in self.output_layer.neurones:
            print('Neuron {', 'weight=', neuron.weight, 'output=', neuron.output, 'delta=', neuron.delta, '}')

        print("-------------------------------------------------------------------------------------------------------")

            
def main():
    data = Data()
    data_train = data.get_train_data()
    data_validation = data.get_val_data()
    data_test = data.get_test_data()

    nn = NeuralNetwork(data_train, data_validation, n_input=len(data_train[0])-1, n_hidden=10, n_output=2)
    nn.print_network(is_trained=False)
    nn.train(learning_rate=0.5, iterations=100)

    print("FINISH TRAINING MODEL")
    nn.print_network(is_trained=True)

    print("START TESTING")
    passed = 0
    for row in data_test:
        prediction = nn.predict(row)
        if row[-1] == prediction:
            passed += 1
        print('Expected=%d, Got=%d' % (row[-1], prediction))

    print("Passed", (passed/len(data_test))*100, "%", "test")


main()
