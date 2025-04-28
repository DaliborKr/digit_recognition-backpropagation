#!/bin/bash

# install requirements

sudo apt install python3-numpy
sudo apt install python3-matplotlib

# get data

wget -P data https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz
wget -P data https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz
wget -P data https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz
wget -P data https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz

# get example pretrained models

wget -P trained_models https://github.com/DaliborKr/sfc_pretrainedModels/raw/refs/heads/main/AG300.pkl
wget -P trained_models https://github.com/DaliborKr/sfc_pretrainedModels/raw/refs/heads/main/basic.pkl
wget -P trained_models https://github.com/DaliborKr/sfc_pretrainedModels/raw/refs/heads/main/AmsDropout.pkl

python3 app.py