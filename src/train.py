import numpy as np
from src import dataLoader as dl
from src import neuralNetwork as nn

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

class NNTrainer():
    def __init__(self, epochs, minibatch_size, lr, opt, name, dropout, imgs, labels, test_imgs, test_labels, accEvalEnabled):

        layers_describ = [(28*28, "in"), (128, "relu"), (64, "relu"), (10, "softmax")]
        self.model = nn.NeuralNetwork(layers_describ, dropout)
        self.epochs = epochs
        self.minibatch_size = minibatch_size
        self.lr = lr
        self.opt = opt
        self.name = name
        self.imgs = imgs
        self.labels = labels
        self.test_imgs = test_imgs
        self.test_labels = test_labels
        self.accEvalEnabled = accEvalEnabled

        self.losses = []
        self.accuracies = []

    def evaluate_single_epoch(self, epoch):
        e_total = 0
        p = np.random.randint(0, len(self.imgs), size=self.minibatch_size)

        self.model.epoch_init()

        for i in p:
            e_p = 0
            img = self.imgs[i]
            label = self.labels[i]

            self.model.forward(img, True)

            # Compute Loss
            e_p = -np.log(self.model.layers[-1].y[0][label])
            e_total += e_p

            self.model.backward(label)

            self.model.update_grads(img.reshape(1, 784))

        self.model.update_parameters(self.minibatch_size, self.lr, self.opt)

        e_total /= self.minibatch_size

        self.losses.append(e_total)
        print (epoch, ":", e_total)

        if self.accEvalEnabled:
            self.meassure_model_accuracy()

    def save_to_file(self):
        self.model.save_to_file(self.name)

    def meassure_model_accuracy(self):
        right_predicted = 0
        for i in range(len(self.test_imgs)):
            self.model.forward(self.test_imgs[i], False)
            prediction = np.argmax(self.model.layers[-1].y)
            if (prediction == self.test_labels[i]):
                right_predicted += 1

        accuracy = (right_predicted / len(self.test_imgs)) * 100
        self.accuracies.append(accuracy)


def plot_losses_accuraries(trainers, epoch):

    fig1, ax1 = plt.subplots(figsize=(9, 6))
    fig2, ax2 = plt.subplots(figsize=(9, 6))

    epochs1 = []
    epochs2 = []
    epochs3 = []

    isAccEvalEnabled = False

    if trainers["model1"] is not None:
        epochs1 = list(range(1, len(trainers["model1"].losses)+1))
        ax1.plot(epochs1, trainers["model1"].losses, label='Model 1', marker='o', linestyle='-', color="tab:blue")
        if trainers["model1"].accEvalEnabled:
            ax2.plot(epochs1, trainers["model1"].accuracies, label='Model 1', marker='o', linestyle='-', color="tab:blue")
            isAccEvalEnabled = True

    if trainers["model2"] is not None:
        epochs2 = list(range(1, len(trainers["model2"].losses)+1))
        ax1.plot(epochs2, trainers["model2"].losses, label='Model 2', marker='s', linestyle='--', color="tab:orange")
        if trainers["model2"].accEvalEnabled:
            ax2.plot(epochs2, trainers["model2"].accuracies, label='Model 2', marker='s', linestyle='--', color="tab:orange")
            isAccEvalEnabled = True

    if trainers["model3"] is not None:
        epochs3 = list(range(1, len(trainers["model3"].losses)+1))
        ax1.plot(epochs3, trainers["model3"].losses, label='Model 3', marker='^', linestyle='-.', color="tab:green")
        if trainers["model3"].accEvalEnabled:
            ax2.plot(epochs3, trainers["model3"].accuracies, label='Model 3', marker='^', linestyle='-.', color="tab:green")
            isAccEvalEnabled = True

    max_ticks = 10
    n = 1 + max([len(epochs1), len(epochs2), len(epochs3)])
    if n <= max_ticks:
        xTicks = np.arange(n)
    else:
        step = max(1, n // max_ticks)
        xTicks = np.arange(1, n, step)


    ax1.set_xlabel('Epochs', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Model Loss Across Epochs', fontsize=14)
    ax1.set_xticks(xTicks)
    ax1.legend(fontsize=10)
    ax1.grid(True)

    fig1.savefig("plot.png", format="png", dpi=300)
    plt.close(fig1)

    if isAccEvalEnabled:
        ax2.set_xlabel('Epochs', fontsize=12)
        ax2.set_ylabel('Accuracy [%]', fontsize=12)
        ax2.set_title('Model Accuracies Across Epochs', fontsize=14)
        ax2.set_xticks(xTicks)
        ax2.legend(fontsize=10)
        ax2.grid(True)

        fig2.savefig("plot2.png", format="png", dpi=300)
    plt.close(fig2)

def start_train_model(epochs, minibatch_sizes, lrs, opts, dropouts, names, active_models, accEvalEnabled):
    imgs, labels = dl.loadData('data/train-images-idx3-ubyte.gz', 'data/train-labels-idx1-ubyte.gz')
    test_imgs, test_labels = dl.loadData('data/t10k-images-idx3-ubyte.gz', 'data/t10k-labels-idx1-ubyte.gz')

    trainers = {"model1": None, "model2" : None, "model3" : None}


    for i, (key, value) in enumerate(trainers.items()):
        if active_models[i] == True:
            trainers[key] = NNTrainer(epochs[i], minibatch_sizes[i], lrs[i], opts[i], names[i], dropouts[i], imgs, labels, test_imgs, test_labels, accEvalEnabled)

    return trainers


def save_models_to_files(trainers):
    for trainer in trainers.values():
        if trainer is not None:
            trainer.save_to_file()

def meassure_trained_models_accuracy(trainers):
    for trainer in trainers.values():
        if trainer is not None:
            trainer.meassure_model_accuracy()
