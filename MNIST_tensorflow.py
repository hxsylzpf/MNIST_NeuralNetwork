# -*- coding:utf-8 -*-

"""
MNIST data classifier using the tool named tensorflow
The accuracy can up to 92%

@version: 1.0
@author: kevin
@license: Apache Licence 
@contact: liujiezhang@bupt.edu.cn
@site: 
@software: PyCharm Community Edition
@file: MNIST_tensorflow.py
@time: 16/11/15 下午12:23
"""

from __future__ import absolute_import, division, print_function
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


class CNN(object):
    '''
    构建简单的卷积神经网络,实现对手写字体的识别
    '''

    def __int__(self):
        # 载入数据
        self.mnist = input_data.read_data_sets('data/', one_hot=True)
        # 输入
        self.x = tf.placeholder(tf.float32, [None, 784])
        # 真实值
        self.y_ = tf.placeholder(tf.float32, [None, 10])
        # reshape
        self.x_image = tf.reshape(self.x, [-1, 28, 28, 1])

    def weight_variable(self, shape):
        '''
        初始化W
        :param shape: 权重W纬度
        :return:
        '''
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_varibale(self, shape):
        '''
        初始化b
        :param shape: 偏置b的纬度
        :return:
        '''
        # 偏置的默认值为0.1
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(self, x, W):
        '''
        卷积操作,padding取为"SAME",表示输出纬度与输入相同
        :param x: 输入图形向量
        :param W: 权重系数
        :return: 卷积后图形向量
        '''
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(self, x):
        '''
        池化操作,输出纬度与输入纬度相同
        :param x: 输入图形向量,4-D tensor with snape [batch, height, width, channels]
        :return: 选取2x2区域中最大值后的属输出向量
        '''
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def con_operate(self, shape, input):
        '''
        卷积池化层操作
        :param shape: 权重纬度张量
        :param input: 输入张量
        :return: 卷积池化后张量
        '''
        # 初始化W, b向量
        W_con = self.weight_variable(shape)
        b_con = self.bias_varibale([shape[-1]])
        # 卷积操作
        h_con = tf.nn.relu(self.conv2d(input, W_con) + b_con)
        # 池化操作
        h_pool = self.max_pool_2x2(h_con)
        return h_con, h_pool

    def build_model(self):
        '''
        构建卷积神经网络
        :return:
        '''
        # 第一层
        _, h_pool1 = self.con_operate([5, 5, 1, 32], self.x_image)
        # 第二层
        _, h_pool2 = self.con_operate([5, 5, 32, 64], h_pool1)
        # 全连接层
        w_fc1 = self.weight_variable([7 * 7 * 64, 1024])
        b_fc1 = self.bias_varibale([1024])
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)
        # dropout
        self.keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)
        # 输出层
        w_fc2 = self.weight_variable([1024, 10])
        b_fc2 = self.bias_varibale([10])
        # 卷积输出值
        self.y_conv = tf.matmul(h_fc1_drop, w_fc2) + b_fc2

    def evalute_model(self, learn_rate, max_iter_num, mini_batch_size):
        '''
        评估模型结果
        :return:
        '''
        # 构建模型
        self.build_model()
        # 定义损失函数,采用交叉熵
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(self.y_conv, self.y_))
        # 使用ADAM代替SGD
        train_step = tf.train.AdamOptimizer(learn_rate).minimize(cross_entropy)
        # 定义正确率计算函数
        correct_prediction = tf.equal(
            tf.argmax(self.y_conv, 1), tf.argmax(self.y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        # 初始化所有变量
        session = tf.InteractiveSession()
        tf.initialize_all_variables().run()
        # 开始迭代训练
        for epoch in range(max_iter_num):
            batch = self.mnist.train.next_batch(mini_batch_size)
            # 每100次输出一次测试准确率
            if epoch % 100 == 0:
                train_accuray = accuracy.eval(
                    feed_dict={self.x: batch[0], self.y_: batch[1], self.keep_prob: 1.0})
                print("epoch:{0},training accuracy: {1}".format(
                    epoch, train_accuray))
            train_step.run(
                feed_dict={self.x: batch[0], self.y_: batch[1], self.keep_prob: 0.5})
        # 计算全部测试集的准确率
        total_accuracy = accuracy.eval(feed_dict={
                                       self.x: self.mnist.test.images, self.y_: self.mnist.test.labels, self.keep_prob: 1.0})
        print("test accuracy {0}".format(total_accuracy))


def mnist_softmax():
    '''
    简单的两层网络,实现对手写字体识别
    :return:
    '''
    # 导入数据
    mnist = input_data.read_data_sets('data/', one_hot=True)

    # 创建模型
    x = tf.placeholder(tf.float32, [None, 784])
    # W, b 类型是Variable
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    # 计算拟合值
    y = tf.matmul(x, W) + b
    # 实际真实值
    y_ = tf.placeholder(tf.float32, [None, 10])

    # 定义损失函数
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(y, y_))
    # 使用随机梯度算法求解,学习率为0.5
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    # 开始训练
    session = tf.InteractiveSession()
    tf.initialize_all_variables().run()
    # 定义最大迭代次数
    max_iter_num = 2000
    # 定义最小batch尺度
    mini_batch_size = 100
    for _ in range(max_iter_num):
        # 使用mini_batch 进行局部调整W,b
        batch_xs, batch_ys = mnist.train.next_batch(mini_batch_size)
        session.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    # 测试评估模型训练结果
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(session.run(accuracy, feed_dict={
          x: mnist.test.images, y_: mnist.test.labels}))


if __name__ == '__main__':
    # 两层网络
    # mnist_softmax()
    # 卷积CNN网络
    cnn = CNN()
    cnn.__int__()
    cnn.evalute_model(learn_rate=1e-4, max_iter_num=2000, mini_batch_size=50)
