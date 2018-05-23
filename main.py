from itertools import tee

import numpy as np
from numpy import random, dot

from input_processor import InputProcessor


def pairwise(*iterable):
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def sigmoid(x):
    return np.tanh(x)


def sigmoid_derivative(x):
    return 1.0 - np.tanh(x) ** 2


class NeuralNetwork:
    def __init__(self, input_nodes, hidden_layers, output_nodes):
        # random.seed(1)
        self.weights = [random.random((a, b)) for a, b in pairwise(input_nodes, *hidden_layers, output_nodes)]

    def predict(self, input_values):
        outputs = [input_values]

        for layer in self.weights:
            # print(values)
            input_values = sigmoid(dot(input_values, layer))
            outputs.append(input_values)

        return outputs

    def train(self, train_input, desired_output):
        for _ in range(5):
            outputs = self.predict(train_input)
            size = len(self.weights)
            adjustments = [0] * size

            error = desired_output - outputs[-1]
            for i in range(size, 0, -1):
                delta = error * sigmoid_derivative(outputs[i])
                adjustments[i - 1] = dot(outputs[i - 1].T, delta)
                error = dot(delta, self.weights[i - 1].T)

            for i in range(size - 1, 0, -1):
                self.weights[i] += adjustments[i] * .1


if __name__ == "__main__":
    random.seed(1)
    neural_network = NeuralNetwork(7, [8, 9], 3)

    input_processor = InputProcessor('data.in')

    values = input_processor.values
    training_inputs = np.array(list(map(lambda l: l[:7], values)))
    training_outputs = np.array(list(map(lambda l: l[7:], values)))
    neural_network.train(training_inputs, training_outputs)

    errors = []
    for value in values:
        out = neural_network.predict(value[:7])[-1]
        predicted = input_processor.revert(out)
        result = input_processor.revert(value[7:])
        print(result, predicted)
        errors.append(np.sum(np.abs(predicted - result) / predicted))

    print(np.average(errors))

# https://rolisz.ro/2013/04/18/neural-networks-in-python/
