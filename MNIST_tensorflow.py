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

from __future__ import absolute_import,division,print_function
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def run():
    # 导入数据
    mnist = input_data.read_data_sets('data/',one_hot=True)

    # 创建模型
    x = tf.placeholder(tf.float32,[None,784])
    # W, b 类型是Variable
    W = tf.Variable(tf.zeros([784,10]))
    b = tf.Variable(tf.zeros([10]))
    # 计算拟合值
    y = tf.matmul(x,W) + b
    # 实际真实值
    y_ = tf.placeholder(tf.float32,[None,10])

    # 定义损失函数
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y,y_))
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
        batch_xs,batch_ys = mnist.train.next_batch(mini_batch_size)
        session.run(train_step,feed_dict={x:batch_xs,y_:batch_ys})

    # 测试评估模型训练结果
    correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    print(session.run(accuracy,feed_dict={x:mnist.test.images,y_:mnist.test.labels}))


if __name__ == '__main__':
    run()


