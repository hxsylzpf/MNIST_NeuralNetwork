# -*- coding:utf-8 -*-

"""
@version: 1.0
@author: kevin
@license: Apache Licence 
@contact: liujiezhang@bupt.edu.cn
@site: 
@software: PyCharm Community Edition
@file: main.py
@time: 16/10/14 上午10:11
"""
import load as input_data
# import MNIST
import network


def train():
    mnist = input_data.readDataSets('data', one_hot=True)
    train_data = mnist.train
    validation_data = mnist.validation
    nn = network.NN(sizes=[784,200,10], epochs=50000, mini_batch_size=10,learning_rate=0.3)
    nn.fit(train_data, validation_data=validation_data)
    nn.save()

def train_MNIST():
    mnist = input_data.readDataSets('data', one_hot=True)
    train_data = mnist.train
    validation_data = mnist.validation
    nn = MNIST.NeuralNetwork(784,250,10)
    nn.train(train_data, epochs=5000,test_set=validation_data,batch_size=20)
    # nn.save()


train_MNIST()
# train()
