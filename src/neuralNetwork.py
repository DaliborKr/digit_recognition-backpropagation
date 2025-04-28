from src import reluLayer as rl
from src import softMaxLayer as sml
import pickle
import numpy as np

class NeuralNetwork:
    def __init__(self, describ, dropout):
        self.dropout = dropout
        self.describ = describ
        self.layers = []

        for i in range(len(describ)):
            if describ[i][1] == "relu":
                self.layers.append(rl.ReluLayer(describ[i][0], describ[i - 1][0], dropout["hidden_prob"]))
            elif describ[i][1] == "softmax":
                self.layers.append(sml.SoftMaxLayer(describ[i][0], describ[i - 1][0]))

    def epoch_init(self):
        for layer in self.layers:
            layer.epoch_init()

    def forward(self, x, is_train):
        if self.dropout["enabled"] and is_train:
            dropout_mask = np.random.binomial(1, self.dropout["input_prob"], size = x.shape)
            x = x * dropout_mask

        for layer in self.layers:
            layer.y  = layer.forward(x, self.dropout["enabled"], is_train)
            x = layer.y

    def backward(self, label):
        self.layers[-1].backward(label)
        i = len(self.layers) - 2
        while (i >= 0):
            self.layers[i].backward(self.layers[i + 1])
            i -= 1

    def update_grads(self, x):
        self.layers[0].weights_growth_minibatch += np.dot(x.T, self.layers[0].delta_e)
        self.layers[0].biases_growth_minibatch += np.sum(self.layers[0].delta_e)

        for i in range(1, len(self.layers)):
            self.layers[i].weights_growth_minibatch += np.dot(self.layers[i - 1].y.T, self.layers[i].delta_e)
            self.layers[i].biases_growth_minibatch += np.sum(self.layers[i].delta_e)

    def update_parameters(self, minibatch_size, lr, opt):
        for layer in self.layers:
            layer.update_parameters(minibatch_size, lr, opt)

    def save_to_file(self, name):
        for layer in self.layers:
            layer.clear_to_save()

        with open(f"trained_models/{name}.pkl", "wb") as file:
            pickle.dump(self, file)