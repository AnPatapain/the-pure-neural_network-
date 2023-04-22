import csv
import numpy as np
import math
from random import random


def sigmoid(x):
    """
    :return: sigmoid of x that passed in argument
    """
    return 1 / (1 + math.exp(-x))


def sigmoid_derivative(sigmoid_output):
    """
    :param sigmoid_output: the sigmoid output of particular x variable
    :return: the sigmoid derivative by x
    """
    return sigmoid_output * (1 - sigmoid_output)


def softmax(z):
    """
    :param z: input vector
    :return: probability vector of the input vector
    """
    exp_z = np.exp(z)
    sum_ = exp_z.sum()
    softmax_z = np.round(exp_z / sum_, 3)
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
    """
    Read data from iris.csv to get data_train, data_validation, data_test and store them in attribute
    """
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
        self.gradient = None


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
                    self.output_layer.neurones[j].gradient = error


            # layer is hidden layer
            else:
                for j in range(len(self.hidden_layer.neurones)):
                    error = 0.0
                    for neurone in self.output_layer.neurones:
                        error += neurone.weights[j] * neurone.gradient
                        errors.append(error)

                # Calculate delta for each neurone in layer
                for j in range(len(self.hidden_layer.neurones)):
                    self.hidden_layer.neurones[j].gradient = errors[j] * \
                                                             self.activation_derivative(
                                                              self.hidden_layer.neurones[j].output)

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
                    neurone.weights[j] -= learning_rate * neurone.gradient * inputs[j]
                # update the bias
                neurone.weights[-1] -= learning_rate * neurone.gradient

    def train(self, learning_rate, iterations):
        """
        :param learning_rate: How much change the weight if we had already the delta for this weight
        :param iterations: How many times we iterate the process feed data and let the network modify the weight.
        :return: The percentage of successful prediction on data validation
        """
        percent_passed = 0
        epoch = 0
        # for epoch in range(iterations):
        while epoch < iterations and percent_passed < 0.9:
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
            if epoch % 20 == 0:
                print('>iteration=%d, learning_rate=%.3f, error=%.3f' % (epoch, learning_rate, sum_error))
                # Validation to monitor the performance of model
                if self.data_validation is not None:
                    passed = 0
                    for row in self.data_validation:
                        prediction = self.predict(row)
                        if row[-1] == prediction:
                            passed += 1
                    percent_passed = passed / len(self.data_validation)
                    print("Passed", percent_passed * 100, "%", "test")

            epoch += 1

        return percent_passed

    def predict(self, row):
        """
        :param row: the input row
        :return: the predicted class for the row
        """
        outputs = self.feed_forward(row)
        return np.argmax(outputs)

    def load_neural_network(self, network):
        """
        Load the weights from the network that has precision more than 90%
        :param network: the network that has accuracy more than 90%
        """
        for i in range(len(self.hidden_layer.neurones)):
            self.hidden_layer.neurones[i].weights = network['hidden_layer'][i]

        for j in range(len(self.output_layer.neurones)):
            self.output_layer.neurones[j].weights = network['output_layer'][j]

    def get_dict(self):
        """
        :return: the dictionary structure of network to be able to load it from the function load_neural_network
        """
        network = {
            'hidden_layer': [],
            'output_layer': []
        }
        for neurone in self.hidden_layer.neurones:
            network['hidden_layer'].append(neurone.weights)
        for neurone in self.output_layer.neurones:
            network['output_layer'].append(neurone.weights)

        return network

    def print_network(self):
        """
        print network for human-readable way
        """
        print("--------------------------------- LE NEURAL NETWORK MODEL ----------------------------")
        print("hidden layer")
        for neuron in self.hidden_layer.neurones:
            print('Neuron {', 'weight=', neuron.weights, 'output=', neuron.output, 'delta=', neuron.gradient, '}')

        print("output layer")
        for neuron in self.output_layer.neurones:
            print('Neuron {', 'weight=', neuron.weights, 'output=', neuron.output, 'delta=', neuron.gradient, '}')

        print("-------------------------------------------------------------------------------------------------------")


def main():
    data = Data()
    data_train = data.get_train_data()
    data_validation = data.get_val_data()
    data_test = data.get_test_data()
    print('data train', data_train)
    print('data validation', data_validation)
    print('data test', data_test)

    """
    This is the good network that has accuracy 95.8%. If you want to use, replace none value of good_network with this.
    {'hidden_layer': [[0.5597013020316779, 0.505212335621036, 0.40534619579459297, 0.1879523544737978, 0.5816204840266999], [0.8922371040395833, 0.7350803039030537, 0.6467528820090848, 0.10829380896175081, 0.9887165983647901], [-1.0743190863105871, -0.40821756050189006, 0.26879266008281333, 0.34795563879886227, 0.48552909158925034], [0.9048796295705168, 0.8544282850996451, 0.6943455563207859, 0.7528697920691767, 0.9189652008791742], [0.5596640405686716, 0.3177569000006153, 0.9991880468059201, 0.023005954785245347, 0.5913271277195034], [0.754198337183547, 0.3022133285094067, 0.25073770787358063, 0.021503131398798205, 0.3229271573960877], [-0.6167426102415126, -0.6980420816540865, 1.5143716942956704, 1.2533863204525169, 0.3904730632501247], [-0.7732076735857647, -1.0139516353327342, 2.3244739626655235, 1.4828880306945629, 0.15605164083960318], [-2.544754538864572, -2.0671510640535495, 3.772523351035693, 2.895162643039866, -0.6395603771736472], [0.9919138698587168, 0.5597425327200244, 0.10957597923433345, 0.1804492970635585, 0.4951962607552142]], 'output_layer': [[0.7403492589379729, 1.0256262494185933, -1.9024747078995343, 1.2766172046504918, 0.9830965181788786, 1.0075902520748163, -0.29199251063872006, -2.9024191488226148, -3.6185307281896057, 0.827454416694655, 1.4050633955420064], [0.519414111318999, 0.21446569517968847, 0.514409993932081, 0.663203079399354, 0.1094442081012084, 0.5513938987315323, 0.896804837502692, 3.438088532564276, -0.422547930070896, 0.4122498708425246, 0.6399824126093314], [-0.022799447191977393, -0.13828485923817235, 2.8021328189431913, 0.1897475456930727, 0.2100423752260654, -0.49250220352866897, 1.4703854971479486, 1.991061135780741, 5.719035762767129, 0.30226146570299445, 0.09708085707915981]]}
    """
    # good_network = {'hidden_layer': [[0.5597013020316779, 0.505212335621036, 0.40534619579459297, 0.1879523544737978, 0.5816204840266999], [0.8922371040395833, 0.7350803039030537, 0.6467528820090848, 0.10829380896175081, 0.9887165983647901], [-1.0743190863105871, -0.40821756050189006, 0.26879266008281333, 0.34795563879886227, 0.48552909158925034], [0.9048796295705168, 0.8544282850996451, 0.6943455563207859, 0.7528697920691767, 0.9189652008791742], [0.5596640405686716, 0.3177569000006153, 0.9991880468059201, 0.023005954785245347, 0.5913271277195034], [0.754198337183547, 0.3022133285094067, 0.25073770787358063, 0.021503131398798205, 0.3229271573960877], [-0.6167426102415126, -0.6980420816540865, 1.5143716942956704, 1.2533863204525169, 0.3904730632501247], [-0.7732076735857647, -1.0139516353327342, 2.3244739626655235, 1.4828880306945629, 0.15605164083960318], [-2.544754538864572, -2.0671510640535495, 3.772523351035693, 2.895162643039866, -0.6395603771736472], [0.9919138698587168, 0.5597425327200244, 0.10957597923433345, 0.1804492970635585, 0.4951962607552142]], 'output_layer': [[0.7403492589379729, 1.0256262494185933, -1.9024747078995343, 1.2766172046504918, 0.9830965181788786, 1.0075902520748163, -0.29199251063872006, -2.9024191488226148, -3.6185307281896057, 0.827454416694655, 1.4050633955420064], [0.519414111318999, 0.21446569517968847, 0.514409993932081, 0.663203079399354, 0.1094442081012084, 0.5513938987315323, 0.896804837502692, 3.438088532564276, -0.422547930070896, 0.4122498708425246, 0.6399824126093314], [-0.022799447191977393, -0.13828485923817235, 2.8021328189431913, 0.1897475456930727, 0.2100423752260654, -0.49250220352866897, 1.4703854971479486, 1.991061135780741, 5.719035762767129, 0.30226146570299445, 0.09708085707915981]]}
    good_network = None

    if good_network:
        nn = NeuralNetwork(data_train, data_validation, n_input=len(data_train[0]) - 1, n_hidden=10, n_output=3)
        nn.load_neural_network(good_network)
    else:
        while True:
            nn = NeuralNetwork(data_train, data_validation, n_input=len(data_train[0]) - 1, n_hidden=10, n_output=3)
            percent_passed = nn.train(learning_rate=0.1, iterations=100)
            if percent_passed > 0.9:
                break

    # Test
    print("START TESTING")
    passed = 0
    for row in data_test:
        prediction = nn.predict(row)
        if row[-1] == prediction:
            passed += 1
        print('Expected=%d, Got=%d' % (row[-1], prediction))

    print("Passed", (passed / len(data_test)) * 100, "%", "test")

    if passed / len(data_test) >= 0.8:
        good_network = nn.get_dict()
        print("------------------------- NETWORK THAT HAS PRECISION >= 90% ------------------------------------")
        print(good_network)


if __name__ == '__main__':
    main()
