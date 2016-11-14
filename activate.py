#! --*--coding:utf-8 --*--

"""
@author:liujiezhang
@email:liujiezhangbupt@gmail.com
@time:2016/10/22
@script:基本的激活函数及其导数，包括：
		Sigmoid
		tanh
		Relu
"""
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(x))


def sigmoidDerivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


def tanh(x):
    return np.tanh(x)


def tanhDerivative(x):
    return 1 - np.multiply(x, x)


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))


def softmaxDerivative(x):
    return np.multiply(softmax(x), 1 - softmax(x))


def crossEntropy(real_prob, pre_prob):

    return -np.dot(real_prob, np.log2(pre_prob))
