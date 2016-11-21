# -*- coding:utf-8 -*-

"""
@version: 1.0
@author: kevin
@license: Apache Licence 
@contact: liujiezhang@bupt.edu.cn
@site: 
@software: PyCharm Community Edition
@file: network_mini_batch.py
@time: 16/10/14 上午09:12
"""
import os
import load_data as input_data

from activate import *


class NN(object):
    '''
    NN model supported any hidden layers by configuring the parameters named "sizes"
    '''

    def __init__(self, sizes=list(), learning_rate=1.0, mini_batch_size=16, epochs=10):
        '''
        参数初始化
        :param sizes: 网络层节点
                        [784,    500,     100,...    10]
                        输入层 / 隐藏层1 / 隐藏层2 ... 输出层
        :param learning_rate:学习率
        :param mini_batch_size:最小batch样本数
        :param epochs:迭代次数
        :return:
        '''
        self.sizes = sizes
        # 网络总层数
        self.num_layers = len(sizes)
        # 权重系数
        self.weights = [np.array([0])] + [np.random.randn(y, x)
                                          for y, x in zip(sizes[1:], sizes[:-1])]
        # 偏置系数
        self.biases = [np.random.randn(y, 1) for y in sizes]
        # _zs = w * x + b
        self._zs = [np.zeros(bias.shape) for bias in self.biases]
        # 激活函数输出值
        self._activations = [np.zeros(bias.shape) for bias in self.biases]
        # 激活函数,导数
        self._activate_func = [sigmoid, sigmoidDerivative]
        # 最小batch尺度
        self.mini_batch_size = mini_batch_size
        # 迭代次数
        self.epochs = epochs
        # 学习率
        self.eta = learning_rate

    def fit(self, train_data, validation_data=None):
        '''
        训练 W, b
        :param train_data: 训练数据
        :param validation_data: 发展集数据
        :return:
        '''

        # for epoch in range(self.epochs):
        accuracy = 0.0
        for epoch in range(self.epochs):
            # 重新加载batch数据集样本
            train_data_images, train_data_labels = train_data.next_batch(
                self.mini_batch_size)
            # 格式化数据
            mini_batches = self.formData(train_data_images, train_data_labels)

            for sample in mini_batches:
                # 初始化每轮batch的 w,b
                nbala_b = [np.zeros(bias.shape) for bias in self.biases]
                nabla_w = [np.zeros(weight.shape) for weight in self.weights]
                # 每个样本数据及其标签
                x, y = sample

                # 前向传播
                self._forward_prop(x)
                # 后向误差传播,得出 w,b偏差量
                data_nabla_b, data_nabla_w = self._back_prob(x, y)
                nbala_b = [nb + dnb for nb,
                           dnb in zip(nbala_b, data_nabla_b)]
                nabla_w = [nw + dnw for nw,
                           dnw in zip(nabla_w, data_nabla_w)]
            # 更新 W,b
            self.weights = [
                w - (self.eta / self.mini_batch_size) * dw for w, dw in zip(self.weights, nabla_w)]
            self.biases = [
                b - (self.eta / self.mini_batch_size) * db for b, db in zip(self.biases, nbala_b)]
            # 对发展集测试训练结果
            if validation_data:
                accuracy = self.validate(validation_data) * 100.0
                print('Epoch {0}, accuracy {1} %.'.format(
                    epoch + 1, accuracy))
            else:
                print('Prcoessed epoch{0}.'.format(epoch))

    def validate(self, validation_data):
        '''
        计算发展集的分类正确率
        :param validation_data: 发展集数据
        :return:准确率
        '''
        validation_results = [(self.predict(x, one_hot=True) == y).all()
                              for x, y in self.formData(validation_data.images, validation_data.labels)]
        # print(sum(validation_results),len(validation_results))
        return sum(validation_results) / len(validation_results)

    def _forward_prop(self, x):
        '''
        前向传播算法
        :param x: 输入向量
        :return:
        '''
        # 输入层
        self._activations[0] = x
        # 逐层计算
        for i in range(1, self.num_layers):
            # _zs = w * x + b
            self._zs[i] = (self.weights[i].dot(
                self._activations[i - 1]) + self.biases[i])
            # 激活值
            self._activations[i] = self._activate_func[0](self._zs[i])

    def _back_prob(self, x, y):
        '''
        后向误差传播
        :param x: 输入向量
        :param y: 样本标签值
        :return:
        '''
        # 初始化所有层 w,b
        nabla_b = [np.zeros(bias.shape) for bias in self.biases]
        nabla_w = [np.zeros(weight.shape) for weight in self.weights]
        # error = (sigmoid(w*x+b) - y) * sigmoidDri(w*x+b)

        error = (self._activations[-1] - y) * \
            self._activate_func[-1](self._zs[-1])

        nabla_b[-1] = error
        nabla_w[-1] = error.dot(self._activations[-2].transpose())
        # 后向计算每层的传播误差
        for l in range(1, self.num_layers - 1)[::-1]:
            # print(l)
            # print(np.shape(self.weights[l + 1].transpose()))
            # print(np.shape(error))
            # print(np.shape(self._activate_func[-1](self._zs[l])))
            # print(np.shape(self._activations[l - 1].transpose()))
            # exit()
            error = np.multiply(
                self.weights[l + 1].transpose().dot(error), self._activate_func[-1](self._zs[l]))
            nabla_b[l] = error
            nabla_w[l] = error.dot(self._activations[l - 1].transpose())

        return nabla_b, nabla_w

    def predict(self, x, one_hot=False):
        '''
        预测函数
        :param x: 输入向量
        :param one_hot: 输出格式是否为one_hot
        :return: 预测值
        '''
        self._forward_prop(x)
        if one_hot:
            y = np.zeros([10, 1])
            y[np.argmax(self._activations[-1])] = 1
            return y
        return np.argmax(self._activations[-1])

    def save(self, file_name='model.npz'):
        '''
        保存模型
        :param file_name:
        :return:
        '''
        np.savez_compressed(file=os.path.join(os.curdir, 'models', file_name),
                            weights=self.weights,
                            biases=self.biases,
                            mini_batch_size=self.mini_batch_size,
                            epochs=self.epochs,
                            eta=self.eta
                            )

    def load(self, file_name='model.npz'):
        '''
        加载模型
        :param file_name:模型名称
        :return:
        '''
        npz_members = np.load(os.path.join(os.curdir, 'models', file_name))

        self.weights = list(npz_members['weights'])
        self.biases = list(npz_members['biases'])

        self.sizes = [b.shape[0] for b in self.biases]
        self.num_layers = len(self.sizes)

        self._zs = [np.zeros(bias.shape) for bias in self.biases]
        self._activations = [np.zeros(bias.shape) for bias in self.biases]

        self.mini_batch_size = int(npz_members['mini_batch_size'])
        self.epochs = int(npz_members['epochs'])
        self.eta = float(npz_members['eta'])

    def formData(self, data_images, data_labels):
        '''
        格式化数据
        :param data_images: 图片向量集
        :param data_labels: 图片标签集
        :return:格式化后数据
        '''

        data = zip([np.reshape(x, (784, 1)) for x in data_images],
                   [np.reshape(y, (10, 1)) for y in data_labels])
        return data

if __name__ == '__main__':
    mnist = input_data.readDataSets('data', one_hot=True)
    train_data = mnist.train
    validation_data = mnist.validation
    nn = NN(sizes=[784, 70, 10], epochs=50000,
            mini_batch_size=10, learning_rate=0.2)
    nn.fit(train_data, validation_data=validation_data)
    nn.save()
