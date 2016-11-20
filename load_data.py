# -*- coding:utf-8 -*-

"""
@version: 1.0
@author: kevin
@license: Apache Licence
@contact: liujiezhang@bupt.edu.cn
@site:
@software: PyCharm Community Edition
@file: network_mini_batch.py
@time: 16/10/11 上午11:52
"""

from __future__ import absolute_import, absolute_import, print_function
import os
import gzip
import numpy
from six.moves import urllib, xrange

# the datafile url
SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'


def _isDownload(file_name, work_dir):
    """
    download the data file if they don't exist
    return file_path
    """
    if not os.path.exists(work_dir):
        os.mkdir(work_dir)
    file_path = os.path.join(work_dir, file_name)
    if not os.path.exists(file_path):
        # download file
        file_path, _ = urllib.request.urlretrieve(
            SOURCE_URL + file_name, file_path)
        state_info = os.stat(file_path)
        print('Successfully downloaded!!', file_name,
              state_info.st_size, 'bytes.')
    return file_path


def _read32(bytestream):

    dt = numpy.dtype(numpy.uint32).newbyteorder('>')
    return numpy.frombuffer(bytestream.read(4), dtype=dt)[0]


def extractImages(file_name):
    """
    Extract the images into a 4D unit8 numpy.array
    like [index,y,x,depty]
    """
    print('Extracting ', file_name)
    with gzip.open(file_name) as bytestream:
        magic = _read32(bytestream)
        if magic != 2051:
            raise ValueError(
                'Invalid magic number %d in MNIST image file:%s' %
                (magic, file_name)
            )
        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        buf = bytestream.read(rows * cols * num_images)
        data = numpy.frombuffer(buf, dtype=numpy.uint8)
        data = data.reshape(num_images, rows, cols, 1)

        return data


def extractLabels(file_name, one_hot=False):
    """
    ===> 1D unit8 numpy.array [index]
    """
    print('Extracting ', file_name)
    with gzip.open(file_name) as bytestream:
        magic = _read32(bytestream)
        if magic != 2049:
            raise ValueError(
                'Invalid magic number %d in MNIST image file:%s' %
                (magic, file_name)
            )
        num_items = _read32(bytestream)
        buf = bytestream.read(num_items)
        labels = numpy.frombuffer(buf, dtype=numpy.uint8)
        if one_hot:
            return denseToOneHot(labels)
        return labels


def denseToOneHot(labels_dense, number_classes=10):
    """
    class lables ==> one hot vectors
    """
    number_labels = labels_dense.shape[0]
    index_offset = numpy.arange(number_labels) * number_classes
    labels_ont_hot = numpy.zeros((number_labels, number_classes))
    labels_ont_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_ont_hot


class DataSet(object):

    def __init__(self, images, labels, fake_data=False):
        if fake_data:
            self._num_examples = 10000
        else:
            assert images.shape[0] == labels.shape[0], (
                'images.shape:%s labels.shape:%s' % (
                    images.shape, labels.shape)
            )
            self._num_examples = images.shape[0]
            # [num examples,rows ,cols,depth] ====> [num examples,rows * cols] assuming depth=1
            assert images.shape[3] == 1
            images = images.reshape(images.shape[0],
                                    images.shape[1] * images.shape[2]
                                    )
            # [0,255] ===> [0.0,1.0]
            images = images.astype(numpy.float32)
            images = numpy.multiply(images, 1.0 / 255.0)
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

        # __build_in__ fget and fset
    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epoch_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, fake_data=False):
        """
        return next "batch_size" examples from data
        """
        if fake_data:
            # 28 * 28 =784
            fake_image = [1.0 for _ in xrange(784)]
            fake_label = 0
            return[fake_image for _ in xrange(batch_size)], [fake_label for _ in xrange(batch_size)]
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        # if not over size
        if self._index_in_epoch > self._num_examples:
            self._epochs_completed += 1

            perm = numpy.arange(self._num_examples)
            # shuffle the data
            numpy.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]

            # start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]


def readDataSets(train_dir, fake_data=False, one_hot=False):
    class DataSets(object):
        pass
    data_sets = DataSets()
    if fake_data:
        data_sets.train = DataSet([], [], fake_data=True)
        data_sets.validation = DataSet([], [], fake_data=True)
        data_sets.test = DataSet([], [], fake_data=True)
        return data_sets

    TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
    TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
    TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
    TEST_LABELS = 't10k-labels-idx1-ubyte.gz'
    VALIDVATION_SIZE = 5000

    local_file = _isDownload(TRAIN_IMAGES, train_dir)
    train_images = extractImages(local_file)

    local_file = _isDownload(TRAIN_LABELS, train_dir)
    train_labels = extractLabels(local_file, one_hot=one_hot)

    local_file = _isDownload(TEST_IMAGES, train_dir)
    test_images = extractImages(local_file)

    local_file = _isDownload(TEST_LABELS, train_dir)
    test_labels = extractLabels(local_file, one_hot=one_hot)

    validation_images = train_images[:VALIDVATION_SIZE]
    validation_labels = train_labels[:VALIDVATION_SIZE]

    train_images = train_images[VALIDVATION_SIZE:]
    train_labels = train_labels[VALIDVATION_SIZE:]

    data_sets.train = DataSet(train_images, train_labels)
    data_sets.validation = DataSet(validation_images, validation_labels)
    data_sets.test = DataSet(test_images, test_labels)

    return data_sets
