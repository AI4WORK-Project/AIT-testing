#!/usr/bin/bash
# This script is used to run all the models in the models_info/train_from_scratch directory
# Ensure to chenge the path to your dataset path
python train.py --lr=0.01 --model vgg16 --datapath '/data/shenghao/dataset/'
python train.py --lr=0.01 --model resnet18 --datapath '/data/shenghao/dataset/'
python train.py --lr=0.01 --model densenet --datapath '/data/shenghao/dataset/'
python train.py --lr=0.01 --model mobilenetv2 --datapath '/data/shenghao/dataset/'
python train.py --lr=0.01 --model efficientnet --datapath '/data/shenghao/dataset/'
python train.py --lr=0.01 --model shufflenetv2 --datapath '/data/shenghao/dataset/'
python train.py --lr=0.01 --model lenet --datapath '/data/shenghao/dataset/'
python train_MNIST.py --lr=0.01 --model lenet --datapath '/data/shenghao/dataset/'
