import sys
from src import train as tr
import numpy as np
import os
import pickle

from PyQt5.QtGui import QPixmap, QColor, QPainter
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QPushButton,
    QTabWidget,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLineEdit,
    QCheckBox,
    QComboBox,
    QTextEdit,
    QSizePolicy
)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

        self.setWindowTitle("Project SFC - XKRICK01")
        self.imagePath = "unknown"
        self.image2Path = "unknown"
        self.inputs = {}
        self.trainers = {}
        self.epoch = 1
        self.maxEpoch = 0

        self.imageInputByteArray = np.zeros((784,), dtype=np.uint8)

        self.tabs = self.createTabs()

        self.setCentralWidget(self.tabs)
        self.setFixedWidth(1300)
        self.setFixedHeight(850)


    def createTabs(self):
        tabs = QTabWidget()
        tabs.setTabPosition(QTabWidget.North)

        tab1 = self.createTab1()

        tab2 = self.createTab2()

        tabs.addTab(tab1, "Train")
        tabs.addTab(tab2, "Classify")
        return tabs

    def createTab1(self):
        tab1 = QWidget()

        tab1Layout = QHBoxLayout()

        settingWidget = QWidget(self)
        settingWidget.setFixedWidth(450)
        settingWidget.setFixedHeight(750)

        self.settingLayout = QVBoxLayout()

        # Model 1


        model1GridLayout = QGridLayout()

        self.model1Checkbox = QCheckBox("Model 1")
        self.model1Checkbox.setChecked(True)
        self.model1Checkbox.stateChanged.connect(lambda state: self.disableModelInput(model1GridLayout, self.model1Checkbox.isChecked(), state))

        self.model1Checkbox.setStyleSheet("font-size: 20px; font-weight: bold;")
        self.settingLayout.addWidget(self.model1Checkbox, alignment=Qt.AlignBottom)

        lr1Layout = self.createLineInput("LR:", "0.001", "lr1")
        model1GridLayout.addLayout(lr1Layout, 0, 0)

        bs1Layout = self.createLineInput("Batch size:", "256", "bs1")
        model1GridLayout.addLayout(bs1Layout, 0, 1)

        epochs1Layout = self.createLineInput("Epochs:", "50", "e1")
        model1GridLayout.addLayout(epochs1Layout, 1, 0)

        modelName1Layout = self.createLineInput("Model name:", "model_basic1", "nm1")
        model1GridLayout.addLayout(modelName1Layout, 1, 1)

        dropout1Layout = self.createModelCheckBox("Dropout:", "do1")
        model1GridLayout.addLayout(dropout1Layout, 2, 0)

        opt1Layout = self.createModelComboBox("Optimalization:", ["None", "Adam", "AmsGrad"], "opt1")
        self.inputs["opt1"].setCurrentIndex(0)
        model1GridLayout.addLayout(opt1Layout, 2, 1)

        drop1PIn = self.createLineInput("Drop probability input:", "0.3", "di1")
        model1GridLayout.addLayout(drop1PIn, 3, 0)

        drop1PHid = self.createLineInput("Drop probability hidden:", "0.5", "dh1")
        model1GridLayout.addLayout(drop1PHid, 3, 1)

        model1GridLayout.setAlignment(Qt.AlignTop)

        self.settingLayout.addLayout(model1GridLayout)

        # Model 2

        model2GridLayout = QGridLayout()

        self.model2Checkbox = QCheckBox("Model 2")
        self.model2Checkbox.setChecked(True)
        self.model2Checkbox.stateChanged.connect(lambda state: self.disableModelInput(model2GridLayout, self.model2Checkbox.isChecked(), state))

        self.model2Checkbox.setStyleSheet("font-size: 20px; font-weight: bold;")
        self.settingLayout.addWidget(self.model2Checkbox, alignment=Qt.AlignBottom)

        lr2Layout = self.createLineInput("LR:", "0.001", "lr2")
        model2GridLayout.addLayout(lr2Layout, 0, 0)

        bs2Layout = self.createLineInput("Batch size:", "256", "bs2")
        model2GridLayout.addLayout(bs2Layout, 0, 1)

        epochs2Layout = self.createLineInput("Epochs:", "50", "e2")
        model2GridLayout.addLayout(epochs2Layout, 1, 0)

        modelName2Layout = self.createLineInput("Model name:", "model_basic2", "nm2")
        model2GridLayout.addLayout(modelName2Layout, 1, 1)

        dropout2Layout = self.createModelCheckBox("Dropout:", "do2")
        model2GridLayout.addLayout(dropout2Layout, 2, 0)

        opt2Layout = self.createModelComboBox("Optimalization:", ["None", "Adam", "AmsGrad"], "opt2")
        self.inputs["opt2"].setCurrentIndex(1)
        model2GridLayout.addLayout(opt2Layout, 2, 1)

        drop2PIn = self.createLineInput("Drop probability input:", "0.3", "di2")
        model2GridLayout.addLayout(drop2PIn, 3, 0)

        drop2PHid = self.createLineInput("Drop probability hidden:", "0.5", "dh2")
        model2GridLayout.addLayout(drop2PHid, 3, 1)

        model2GridLayout.setAlignment(Qt.AlignTop)


        self.settingLayout.addLayout(model2GridLayout)

        # Model 3

        model3GridLayout = QGridLayout()

        self.model3Checkbox = QCheckBox("Model 3")
        self.model3Checkbox.setChecked(True)
        self.model3Checkbox.stateChanged.connect(lambda state: self.disableModelInput(model3GridLayout, self.model3Checkbox.isChecked(), state))

        self.model3Checkbox.setStyleSheet("font-size: 20px; font-weight: bold;")
        self.settingLayout.addWidget(self.model3Checkbox, alignment=Qt.AlignBottom)

        lr3Layout = self.createLineInput("LR:", "0.001", "lr3")
        model3GridLayout.addLayout(lr3Layout, 0, 0)

        bs3Layout = self.createLineInput("Batch size:", "256", "bs3")
        model3GridLayout.addLayout(bs3Layout, 0, 1)

        epochs3Layout = self.createLineInput("Epochs:", "50", "e3")
        model3GridLayout.addLayout(epochs3Layout, 1, 0)

        modelName3Layout = self.createLineInput("Model name:", "model_basic3", "nm3")
        model3GridLayout.addLayout(modelName3Layout, 1, 1)

        dropout3Layout = self.createModelCheckBox("Dropout:", "do3")
        model3GridLayout.addLayout(dropout3Layout, 2, 0)

        opt3Layout = self.createModelComboBox("Optimalization:", ["None", "Adam", "AmsGrad"], "opt3")
        self.inputs["opt3"].setCurrentIndex(2)
        model3GridLayout.addLayout(opt3Layout, 2, 1)

        drop3PIn = self.createLineInput("Drop probability input:", "0.3", "di3")
        model3GridLayout.addLayout(drop3PIn, 3, 0)

        drop3PHid = self.createLineInput("Drop probability hidden:", "0.5", "dh3")
        model3GridLayout.addLayout(drop3PHid, 3, 1)

        model3GridLayout.setAlignment(Qt.AlignTop)

        self.settingLayout.addLayout(model3GridLayout)

        # Evaluate Accuracy
        self.accEnCheckbox = QCheckBox("Enable accuracy evaluation\n(turn off for better training performance)")
        self.accEnCheckbox.setChecked(False)
        self.accEnCheckbox.setContentsMargins(0, 80, 0, 0)
        self.settingLayout.addWidget(self.accEnCheckbox, alignment= Qt.AlignCenter | Qt.AlignBottom)

        # Buttons

        buttonLayout = QHBoxLayout()

        self.buttonStart = QPushButton("Start/Reset")
        self.buttonStart.clicked.connect(self.startButtonClicked)

        self.buttonStep = QPushButton("Train step")
        self.buttonStep.clicked.connect(self.stepButtonClicked)
        self.buttonStep.setDisabled(True)
        self.buttonStep.setStyleSheet("color: gray;")

        self.buttonTrain = QPushButton("Train full")
        self.buttonTrain.clicked.connect(self.fullTrainButtonClicked)
        self.buttonTrain.setDisabled(True)
        self.buttonTrain.setStyleSheet("color: gray;")

        self.buttonStop = QPushButton("Stop")
        self.buttonStop.clicked.connect(self.stopButtonClicked)
        self.buttonStop.setDisabled(True)
        self.buttonStop.setStyleSheet("color: gray;")

        buttonLayout.addWidget(self.buttonStart, alignment= Qt.AlignCenter | Qt.AlignBottom)
        buttonLayout.addWidget(self.buttonStep, alignment= Qt.AlignCenter | Qt.AlignBottom)
        buttonLayout.addWidget(self.buttonTrain, alignment= Qt.AlignCenter | Qt.AlignBottom)
        buttonLayout.addWidget(self.buttonStop, alignment= Qt.AlignCenter | Qt.AlignBottom)

        buttonLayout.setContentsMargins(0, 40, 0, 0)

        self.settingLayout.addLayout(buttonLayout)

        # Image

        resultWidget = QWidget(self)
        resultWidget.setFixedWidth(600)
        resultWidget.setFixedHeight(400)

        resultLayout = QVBoxLayout()

        self.graphLabel = QLabel("Image Area")
        self.graphLabel.setStyleSheet("border: 1px solid black;")
        self.graphLabel.setAlignment(Qt.AlignCenter)
        resultLayout.addWidget(self.graphLabel)

        result2Widget = QWidget(self)
        result2Widget.setFixedWidth(600)
        result2Widget.setFixedHeight(400)

        result2Layout = QVBoxLayout()

        self.graph2Label = QLabel("Image Area")
        self.graph2Label.setStyleSheet("border: 1px solid black;")
        self.graph2Label.setAlignment(Qt.AlignCenter)
        result2Layout.addWidget(self.graph2Label)

        self.loadImage()

        # Output

        outputLayout = QVBoxLayout()
        outputLabel = QLabel("Output:", self)
        self.consoleOutput = QTextEdit(self)
        self.consoleOutput.setReadOnly(True)
        self.consoleOutput.setStyleSheet("font-family: 'Courier New'; font-size: 10pt;")
        self.consoleOutput.setFixedWidth(220)

        outputLayout.addWidget(outputLabel)
        outputLayout.addWidget(self.consoleOutput)


        result2Widget.setLayout(result2Layout)
        resultWidget.setLayout(resultLayout)
        settingWidget.setLayout(self.settingLayout)

        resultsLayout = QVBoxLayout()
        resultsLayout.addWidget(resultWidget, alignment=Qt.AlignTop)
        resultsLayout.addWidget(result2Widget, alignment=Qt.AlignTop)

        tab1Layout.addWidget(settingWidget, alignment=Qt.AlignTop)
        tab1Layout.addLayout(resultsLayout)
        tab1Layout.addLayout(outputLayout)

        tab1.setLayout(tab1Layout)
        return tab1

    def createLineInput(self, labelTxt, defaultValue, inputKey):
        inputLayout = QHBoxLayout()
        label = QLabel(labelTxt, self)
        input = QLineEdit(self)

        self.inputs[inputKey] = input

        input.setText(defaultValue)
        inputLayout.addWidget(label)
        inputLayout.addWidget(input)
        return inputLayout

    def createModelCheckBox(self, labelTxt, inputKey):
        checkBoxLayout = QHBoxLayout()
        label = QLabel(labelTxt, self)

        checkbox = QCheckBox()
        checkbox.setCheckState(Qt.Unchecked)

        self.inputs[inputKey] = checkbox

        checkBoxLayout.addWidget(label)
        checkBoxLayout.addWidget(checkbox)
        return checkBoxLayout

    def createModelComboBox(self, labelTxt, options, inputKey):
        cheackComboLayout = QHBoxLayout()
        label = QLabel(labelTxt, self)

        combobox = QComboBox()
        combobox.addItems(options)

        self.inputs[inputKey] = combobox

        cheackComboLayout.addWidget(label)
        cheackComboLayout.addWidget(combobox)
        return cheackComboLayout


    def disableModelInput(self, layout, isChecked, state):
        for i in range(layout.count()):
            item = layout.itemAt(i)
            if item.widget() is not None:
                item.widget().setDisabled(not isChecked)
                item.widget().setStyleSheet("" if isChecked else "color: gray;")

                if item.widget() is self.model1Checkbox or item.widget() is self.model2Checkbox or item.widget() is self.model3Checkbox:
                    item.widget().setStyleSheet("font-size: 20px; font-weight: bold;" if isChecked else "font-size: 20px; font-weight: bold; color: gray;")
            elif item.layout() is not None:
                self.disableModelInput(item.layout(), isChecked, state)

    def loadImage(self):
        if self.imagePath:
            pixmap = QPixmap(self.imagePath)
            pixmap2 = QPixmap(self.image2Path)
            if not pixmap.isNull():
                self.graphLabel.setPixmap(pixmap.scaled(self.graphLabel.size() * 1.05, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                self.graphLabel.repaint()
            else:
                self.graphLabel.setText("No results to be shown")

            if not pixmap2.isNull():
                self.graph2Label.setPixmap(pixmap2.scaled(self.graph2Label.size() * 1.05, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                self.graph2Label.repaint()
            else:
                self.graph2Label.setText("Accuracy evaluation is turned off")


    def startButtonClicked(self):
        self.consoleOutput.setText("")
        self.imagePath = "plot.png"
        self.graphLabel.setText("No results to be shown")
        if not self.accEnCheckbox.isChecked():
            self.image2Path = "unknown"
            self.graph2Label.setText("Accuracy evaluation is turned off")
        else:
            self.image2Path = "plot2.png"
            self.graph2Label.setText("No results to be shown")

        self.disableModelInput(self.settingLayout, False, None)

        self.buttonStep.setDisabled(False)
        self.buttonStep.setStyleSheet("")

        self.buttonTrain.setDisabled(False)
        self.buttonTrain.setStyleSheet("")

        self.buttonStop.setDisabled(False)
        self.buttonStop.setStyleSheet("")

        epochs = [
            int(self.inputs["e1"].text()),
            int(self.inputs["e2"].text()),
            int(self.inputs["e3"].text())
            ]
        minibatch_sizes = [
            int(self.inputs["bs1"].text()),
            int(self.inputs["bs2"].text()),
            int(self.inputs["bs3"].text())
            ]
        lrs = [
            float(self.inputs["lr1"].text()),
            float(self.inputs["lr2"].text()),
            float(self.inputs["lr3"].text())
            ]
        opts = self.parseOptimalizationInputs()
        dropouts = self.parseDropOutInputs()
        names = [
            self.inputs["nm1"].text(),
            self.inputs["nm2"].text(),
            self.inputs["nm3"].text()
            ]
        active_models = [
            self.model1Checkbox.isChecked(),
            self.model2Checkbox.isChecked(),
            self.model3Checkbox.isChecked()
            ]

        self.epoch = 1
        self.maxEpoch = np.max(np.array(epochs) * np.array(active_models))

        self.trainers = tr.start_train_model(epochs, minibatch_sizes, lrs, opts, dropouts, names, active_models, self.accEnCheckbox.isChecked())

    def displayConsoleOutput(self):
        text = f"Epoch {self.epoch:}:\n"
        if self.trainers["model1"] is not None:
            text += f" Model 1: \n   Loss = {self.trainers["model1"].losses[-1]:.4f}"
            if self.accEnCheckbox.isChecked():
                text += f", \n   Accuracy = {self.trainers["model1"].accuracies[-1]:.4f}\n"
            else:
                text += "\n"

        if self.trainers["model2"] is not None:
            text += f" Model 2: \n   Loss = {self.trainers["model2"].losses[-1]:.4f}"
            if self.accEnCheckbox.isChecked():
                text += f", \n   Accuracy = {self.trainers["model2"].accuracies[-1]:.4f}\n"
            else:
                text += "\n"

        if self.trainers["model3"] is not None:
            text += f" Model 3: \n   Loss = {self.trainers["model3"].losses[-1]:.4f}"
            if self.accEnCheckbox.isChecked():
                text += f", \n   Accuracy = {self.trainers["model3"].accuracies[-1]:.4f}\n"
            else:
                text += "\n"

        text += "\n_____________________\n"

        self.consoleOutput.append(text)
        self.consoleOutput.moveCursor(self.consoleOutput.textCursor().End)

    def stepButtonClicked(self):
        for trainer in self.trainers.values():
            if trainer is not None and self.epoch <= trainer.epochs:
                trainer.evaluate_single_epoch(self.epoch)

        tr.plot_losses_accuraries(self.trainers, self.epoch)

        self.loadImage()

        self.displayConsoleOutput()
        QApplication.processEvents()

        self.epoch += 1

        if self.epoch > self.maxEpoch:
            self.training_end()


    def fullTrainButtonClicked(self):
        self.buttonStep.setDisabled(True)
        self.buttonStep.setStyleSheet("color: gray;")

        self.buttonTrain.setDisabled(True)
        self.buttonTrain.setStyleSheet("color: gray;")

        while self.epoch <= self.maxEpoch:
            self.stepButtonClicked()

    def stopButtonClicked(self):
        self.maxEpoch = 0

        self.disableModelInput(self.settingLayout, True, None)

        self.buttonStep.setDisabled(True)
        self.buttonStep.setStyleSheet("color: gray;")

        self.buttonTrain.setDisabled(True)
        self.buttonTrain.setStyleSheet("color: gray;")

        self.buttonStop.setDisabled(True)
        self.buttonStop.setStyleSheet("color: gray;")

    def training_end(self):
        self.disableModelInput(self.settingLayout, True, None)

        self.buttonStep.setDisabled(True)
        self.buttonStep.setStyleSheet("color: gray;")

        self.buttonTrain.setDisabled(True)
        self.buttonTrain.setStyleSheet("color: gray;")

        self.buttonStop.setDisabled(True)
        self.buttonStop.setStyleSheet("color: gray;")

        self.consoleOutput.append(f"\nTraining finished after {self.epoch - 1} epochs.\n")

        tr.meassure_trained_models_accuracy(self.trainers)

        self.consoleOutput.append(f"\n\nTrained models accuracies after last trainnig epoch:\n")
        if self.trainers["model1"] is not None:
           self.consoleOutput.append(f"Model 1 accuracy = {self.trainers["model1"].accuracies[-1]:.4f}\n")

        if self.trainers["model2"] is not None:
            self.consoleOutput.append(f"Model 2 accuracy = {self.trainers["model2"].accuracies[-1]:.4f}\n")


        if self.trainers["model3"] is not None:
            self.consoleOutput.append(f"Model 3 accuracy = {self.trainers["model3"].accuracies[-1]:.4f}\n")

        tr.save_models_to_files(self.trainers)

        fileCombobox = self.inputs["cbFile"]
        fileCombobox.clear()

        fileCombobox.addItems(self.getFileComboboxOptions())

        self.consoleOutput.append("\nModels saved to files.")

        self.trainers = {}


    def parseOptimalizationInputs(self):

        inputKeys = ["opt1", "opt2", "opt3"]
        opts = [
            {"Adam" : False, "AmsGrad" : False},
            {"Adam" : False, "AmsGrad" : False},
            {"Adam" : False, "AmsGrad" : False}
            ]

        for i in range(len(opts)):
            selectedValue = self.inputs[inputKeys[i]].currentText()
            if selectedValue == "Adam":
                opts[i][selectedValue] = True
            elif selectedValue == "AmsGrad":
                opts[i][selectedValue] = True

        return opts


    def parseDropOutInputs(self):

        inputKeysEn = ["do1", "do2", "do3"]
        inputKeysDi = ["di1", "di2", "di3"]
        inputKeysDh = ["dh1", "dh2", "dh3"]
        dropouts = [
            {"enabled": False, "input_prob" : -1, "hidden_prob" : -1},
            {"enabled": False, "input_prob" : -1, "hidden_prob" : -1},
            {"enabled": False, "input_prob" : -1, "hidden_prob" : -1}
            ]

        for i in range(len(dropouts)):
            dropouts[i]["enabled"] = self.inputs[inputKeysEn[i]].isChecked()
            dropouts[i]["input_prob"] = 1 - float(self.inputs[inputKeysDi[i]].text())
            dropouts[i]["hidden_prob"] = 1 - float(self.inputs[inputKeysDh[i]].text())

        return dropouts



    def createTab2(self):
        tab2 = QWidget()


        tab2Layout = QHBoxLayout()


        imagePaintMainWidget = QWidget()
        imagePaintMainWidget.setFixedWidth(500)


        imagePaintLayout = QVBoxLayout()

        canvasLabel = QLabel("Canvas")
        canvasLabel.setStyleSheet("font-size: 40px; font-weight: bold;")

        canvasHintLabel = QLabel("Hint: Press right mouse button and drag to paint.")


        self.imagePaintWidget = QWidget(self)
        self.imagePaintWidget.setStyleSheet("border: 1px solid black;")

        self.gridImageLayout = QGridLayout()
        self.gridImageLayout.setSpacing(0)


        self.squares = []
        for i in range(28):
            row = []
            for j in range(28):
                square = ClickableSquare(i, j)
                self.gridImageLayout.addWidget(square, i, j)
                row.append(square)
            self.squares.append(row)

        self.mouse_down = False
        self.right_button = False

        self.imagePaintWidget.setLayout(self.gridImageLayout)


        buttonImageLayout = QHBoxLayout()

        self.buttonClear = QPushButton("Clear")
        self.buttonClear.clicked.connect(self.clearImage)

        buttonImageLayout.addWidget(self.buttonClear, alignment=Qt.AlignTop | Qt.AlignLeft)

        imagePaintLayout.addWidget(canvasLabel, alignment=Qt.AlignBottom | Qt.AlignLeft)
        imagePaintLayout.addWidget(canvasHintLabel, alignment=Qt.AlignBottom | Qt.AlignLeft)

        imagePaintLayout.addWidget(self.imagePaintWidget, alignment=Qt.AlignTop | Qt.AlignLeft)
        imagePaintLayout.addLayout(buttonImageLayout)


        classifyImageLayout = QVBoxLayout()
        self.filesLayout = self.createComboBoxFiles("Trained model:", "cbFile")
        imagePaintLayout.addLayout(self.filesLayout)

        self.buttonClassify = QPushButton("Classify image")
        self.buttonClassify.clicked.connect(self.classifyImage)
        imagePaintLayout.addWidget(self.buttonClassify, alignment=Qt.AlignTop | Qt.AlignCenter)

        self.labelClassify = QLabel("", self)
        classifyImageLayout.addWidget(self.labelClassify, alignment=Qt.AlignBottom | Qt.AlignCenter)

        self.labelClassifiedDigit = QLabel("")
        self.labelClassifiedDigit.setStyleSheet("font-size: 180px; font-weight: bold;")
        classifyImageLayout.addWidget(self.labelClassifiedDigit, alignment=Qt.AlignTop | Qt.AlignCenter)

        self.labelDigitsProb = QLabel("")
        classifyImageLayout.addWidget(self.labelDigitsProb, alignment=Qt.AlignTop | Qt.AlignCenter)

        imagePaintMainWidget.setLayout(imagePaintLayout)

        tab2Layout.addWidget(imagePaintMainWidget)
        tab2Layout.addLayout(classifyImageLayout)




        tab2.setLayout(tab2Layout)
        return tab2

    def classifyImage(self):
        modelPath = "trained_models/" + self.inputs["cbFile"].currentText() + ".pkl"
        with open(modelPath, "rb") as file:
            model = pickle.load(file)

        model.forward(self.imageInputByteArray, False)
        y = model.layers[-1].y.reshape(10)
        result = np.argmax(y)

        self.labelClassifiedDigit.setText(str(result))
        self.labelClassify.setText(f"Number on the picture was classified by model '{self.inputs["cbFile"].currentText()}' as digit:")

        digitProbText = "Classification probability for each digit:\n\n"

        for i in range(int(len(y) / 2)):
            digitProbText += f"{i}:  {y[i]:<10.3f}\t\t{i + 5}:  {y[i + 5]:<10.3f}\n"

        self.labelDigitsProb.setText(digitProbText)


    def createComboBoxFiles(self, labelTxt, inputKey):
        fileComboLayout = QGridLayout()

        label = QLabel(labelTxt, self)
        label.setFixedWidth(150)

        combobox = QComboBox()
        combobox.setFixedWidth(150)
        combobox.addItems(self.getFileComboboxOptions())

        self.inputs[inputKey] = combobox

        fileComboLayout.addWidget(label, 0, 0)
        fileComboLayout.addWidget(combobox, 0, 1)


        return fileComboLayout

    def getFileComboboxOptions(self):
        options = [f for f in os.listdir("trained_models") if os.path.isfile(os.path.join("trained_models", f))]

        for i in range(len(options) - 1, -1, -1):
            if options[i].lower().endswith(".pkl"):
                options[i] = options[i][:-4]
            else:
                del options[i]
        return options

    def clearImage(self):
        for i in range(28):
            for j in range(28):
                self.squares[i][j].setClicked(False, 0)
                self.imageInputByteArray[i*28 + j] = 0

        self.labelClassifiedDigit.setText("")
        self.labelClassify.setText("")
        self.labelDigitsProb.setText("")


    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.mouse_down = True
            self.right_button = True
            self.updateSquareUnderMouse(event)
        elif event.button() == Qt.RightButton:
            self.mouse_down = True
            self.right_button = False
            self.updateSquareUnderMouse(event)

    def mouseReleaseEvent(self, event):
        if event.button() in [Qt.LeftButton, Qt.RightButton]:
            self.mouse_down = False

    def mouseMoveEvent(self, event):
        if self.mouse_down:
            self.updateSquareUnderMouse(event)

    def updateSquareUnderMouse(self, event):
        grid_pos = self.imagePaintWidget.mapFromGlobal(event.globalPos())

        for i in range(28):
            for j in range(28):
                square_main = self.squares[i][j]
                squares_surround = []
                if i < 27:
                    squares_surround.append((self.squares[i + 1][j], i + 1, j))
                if i > 0:
                    squares_surround.append((self.squares[i - 1][j], i - 1, j))
                if j < 27:
                    squares_surround.append((self.squares[i][j + 1], i, j + 1))
                if j > 0:
                    squares_surround.append((self.squares[i][j - 1], i, j - 1))

                if square_main.geometry().contains(grid_pos):
                    if self.right_button:
                        square_main.setClicked(False, 0)
                        self.imageInputByteArray[i*28 + j] = 0

                        for square in squares_surround:
                            square[0].setClicked(False, 0)
                            self.imageInputByteArray[square[1]*28 + square[2]] = 0
                    else:
                        square_main.setClicked(True, 255)
                        self.imageInputByteArray[i*28 + j] = 1

                        for square in squares_surround:
                            value = 255
                            #value = np.random.randint(150, 250)
                            square[0].setClicked(True, value)
                            self.imageInputByteArray[square[1]*28 + square[2]] = 1



class ClickableSquare(QWidget):
    def __init__(self, x, y, parent=None):
        super().__init__(parent)
        self.x = x
        self.y = y
        self.clicked = False
        self.colorValue = 0
        self.setFixedSize(15, 15)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(self.colorValue, self.colorValue, self.colorValue) if self.clicked else QColor(Qt.white))
        painter.end()

    def setClicked(self, state, colorValue):
        self.clicked = state
        self.colorValue = 255 - colorValue
        self.update()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec_()
