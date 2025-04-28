import numpy as np

class Layer:
    def __init__(self, number_of_neurons, prev_number_of_neurons):
        self.weights = np.random.randn(prev_number_of_neurons, number_of_neurons) * 0.01

        self.biases = np.zeros((1, number_of_neurons))

        self.y = np.zeros((1, number_of_neurons))
        self.delta_e = np.zeros((1, number_of_neurons))
        self.delta_e_minibatches = np.zeros((1, number_of_neurons))
        self.weights_growth_minibatch = np.zeros((prev_number_of_neurons, number_of_neurons))
        self.biases_growth_minibatch = np.zeros((1, number_of_neurons))

        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 0.0000001

        self.m_w = np.zeros_like(self.weights)
        self.m_w_correct = np.zeros_like(self.weights)
        self.v_w = np.zeros_like(self.weights)
        self.v_w_correct = np.zeros_like(self.weights)
        self.m_b = np.zeros_like(self.biases)
        self.m_b_correct = np.zeros_like(self.biases)
        self.v_b = np.zeros_like(self.biases)
        self.v_b_correct = np.zeros_like(self.biases)

    def epoch_init(self):
        self.weights_growth_minibatch = np.zeros((self.weights.shape[0], self.weights.shape[1]))
        self.biases_growth_minibatch = np.zeros((1, self.biases.shape[1]))
        self.delta_e_minibatches = np.zeros((1, self.biases.shape[1]))

    def forward(self, x):
        return np.dot(x, self.weights) + self.biases

    def backward_l(self):
        return np.dot(self.delta_e, self.weights.T)

    def update_parameters(self, minibatch_size, lr, opt):
        weights_grad = self.weights_growth_minibatch / minibatch_size
        biases_grad = self.biases_growth_minibatch / minibatch_size

        if opt["Adam"] or opt["AmsGrad"]:

            self.v_w = self.beta2 * self.v_w + (1 - self.beta2) * (weights_grad**2)
            self.v_b = self.beta2 * self.v_b + (1 - self.beta2) * (biases_grad**2)

            self.m_w = self.beta1 * self.m_w + (1 - self.beta1) * weights_grad
            self.m_b = self.beta1 * self.m_b + (1 - self.beta1) * biases_grad

            if opt["Adam"]:
                self.v_w_correct = self.v_w / (1 - self.beta2)
                self.v_b_correct = self.v_b / (1 - self.beta2)
            else:
                self.v_w_correct = np.maximum(self.v_w_correct, self.v_w)
                self.v_b_correct = np.maximum(self.v_b_correct, self.v_b)


            self.weights +=  (lr * self.m_w) / (np.sqrt(self.v_w_correct) + self.eps)
            self.biases += lr * biases_grad
        else:
            self.weights += lr * weights_grad
            self.biases += lr * biases_grad

    def clear_to_save(self):
        self.y = None
        self.delta_e = None
        self.delta_e_minibatches = None
        self.weights_growth_minibatch = None
        self.biases_growth_minibatch = None

        self.m_w = None
        self.m_w_correct = None
        self.v_w = None
        self.v_w_correct = None
        self.m_b = None
        self.m_b_correct = None
        self.v_b = None
        self.v_b_correct = None