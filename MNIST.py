#!/usr/bin/python3

import time
from activate import *
import load as input_data


class NeuralNetwork(object):

    def __init__(self, in_units, hidden_units, out_units):
        """Returns a new 3-layer neural network with the specified layer sizes."""

        # Hyper parameters
        self.input_size = in_units
        self.output_size = out_units
        self.hidden_size = hidden_units
        self.activate_func = [sigmoid,sigmoidDerivative]
        # Learning parameters
        self.rate = 6.0

        # Weight parameters, randomly initialized
        self.W1 = np.random.uniform(-0.5, 0.5, (self.input_size, self.hidden_size))
        self.W2 = np.random.uniform(-0.5, 0.5, (self.hidden_size, self.output_size))

    def configure(self, rate=None):
        """Change the learning parameters of the network."""
        self.rate = self.rate if rate is None else rate

    def init_weights(self):
        """Initialize weights using Nguyen-Widrow."""
        self.W1 = np.random.uniform(-0.5, 0.5, (self.input_size, self.hidden_size))
        self.W2 = np.random.uniform(-0.5, 0.5, (self.hidden_size, self.output_size))

        # Initialize the hidden layer weights
        beta = 0.7 * (self.hidden_size ** (1.0 / self.input_size))
        for n in range(self.hidden_size):
            norm_val = np.linalg.norm(self.W1[:,n])
            self.W1[:,n] = np.multiply(self.W1[:,n], beta / norm_val)

        # Initialize the output layer weights
        beta = 0.7 * (self.output_size ** (1.0 / self.hidden_size))
        for n in range(self.output_size):
            norm_val = np.linalg.norm(self.W2[:,n])
            self.W2[:,n] = np.multiply(self.W2[:,n], beta / norm_val)

    def forward(self, sample):
        """Forward propagation through the network.
        sample: ndarray of shape (n, input_size), where n is number of samples
        """
        self.Z2 = np.dot(sample.T, self.W1).T
        self.A2 = self.activate_func[0](self.Z2)
        self.Z3 = np.dot(self.A2.T, self.W2).T
        self.y_hat = self.activate_func[0](self.Z3)
        return self.y_hat

    def cost(self, estimate, target):
        """Sum Squared Error cost function.
        estimate: ndarray of shape (output_size,n), where n is number of samples
        target  : ndarray of shape (output_size,n)
        """
        return np.mean(np.mean((target - estimate) ** 2,axis=0))

    def cost_prime(self, sample, target, estimate):
        """Gradient descent derivative.
        sample  : ndarray of shape (n, input_size), where n is number of samples
        target  : ndarray of shape (n, output_size)
        estimate: ndarray of shape (n, output_size)
        """
        total = len(sample)

        delta3 = np.multiply(-(target - estimate), self.activate_func[-1](self.Z3))
        dW2 = np.multiply(np.dot(self.A2, delta3.T), 2 / total)

        delta2 = np.dot(self.W2,delta3) * self.activate_func[-1](self.Z2)
        dW1 = np.multiply(np.dot(sample, delta2.T), 2 / total)

        return dW1, dW2

    def evaluate(self, sample, target):
        """Evaluate network performace using given data."""
        results = self.forward(sample.T)
        pairs = [(np.argmax(x), np.argmax(y)) for x, y in zip(results.T, target.T)]
        correct = sum(int(x == y) for x, y in pairs)
        return correct

    def backprop(self, images, labels):
        """Update weights using batch backpropagation."""
        size = len(labels)
        dW1s = []
        dW2s = []

        for i in range(size):
            label = labels[i]
            image = images[i]

            estimate = self.forward(image)
            dW1, dW2 = self.cost_prime(image, label, estimate)

            dW1s.append(dW1)
            dW2s.append(dW2)

        self.W1 = self.W1 - (self.rate / size) * sum(dW1s)
        self.W2 = self.W2 - (self.rate / size) * sum(dW2s)

    def train(self, train_data, epochs, batch_size, test_set=None):
        """Train the neural network using given data and parameters."""
        if test_set is not None:
            size_test = len(test_set.labels)
        size = len(train_data.labels)
        print("num training data: {}".format(size))

        self.costs = []
        start = time.time()
        for r in range(epochs):
            batch_datas = []
            for i in range(10):
                # 重新加载batch数据集样本
                train_data_images, train_data_labels = train_data.next_batch(
                    batch_size)
                # 格式化数据
                mini_batches = self.formData(train_data_images, train_data_labels)
                batch_datas.append(mini_batches)


            for batch_data in batch_datas:
                images,labels = [],[]
                for data in batch_data:
                    images.append(data[0])
                    labels.append(data[1])
                self.backprop(images, labels)

            # target = train_data.labels
            # sample = train_data.images
            # estimate = self.forward(sample.T)
            # cost = self.cost(estimate, target.T)
            # self.costs.append(cost)
            # print("Epoch {} complete: cost {}".format(r, cost))

            if test_set is not None:
                target = test_set.labels
                sample = test_set.images
                correct = self.evaluate(sample, target.T)
                print("  {} / {}".format(correct, size_test))
        stop = time.time()
        elapsed = stop - start
        print("Time elapsed: {} sec".format(elapsed))

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
    nn = NeuralNetwork(784,80,10)
    nn.train(train_data, epochs=5000,test_set=validation_data,batch_size=20)
