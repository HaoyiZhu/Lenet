#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = ''
__author__ = 'zhuhaoyi'
__mtime__ = '2020/7/16'

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import time
from functools import reduce

np.random.seed(1337)


class Conv:
    def __init__(self, inputs_shape, output_channels=6, ksize=5, stride=1):
        self.inputs_shape = inputs_shape
        self.output_channels = output_channels
        self.stride = stride
        self.ksize = ksize
        self.activation = Relu()

        self.lr = None
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-08
        self.m = None
        self.v = None
        self.n = 0
        self.m_b = None
        self.v_b = None

    def init_weights(self):
        self.inputs_channels = self.inputs_shape[1]
        weights_scale = np.sqrt(reduce(lambda x, y: x * y, self.inputs_shape) / self.output_channels)
        self.weights = np.random.standard_normal(
            (self.ksize, self.ksize, self.inputs_channels, self.output_channels)) / weights_scale
        self.bias = np.random.standard_normal((1, self.output_channels)) / weights_scale

    def forward(self, inputs):
        inputs = inputs.transpose(0, 2, 3, 1)
        self.batchsize = inputs.shape[0]
        self.col_image = self.split_by_strides(inputs)
        self.outputs = np.tensordot(self.col_image, self.weights, axes=([3,4,5],[0,1,2]))
        self.outputs += self.bias
        self.outputs = self.outputs.transpose(0, 3, 1, 2)
        self.outputs = self.activation.forward(self.outputs)

    def backward(self, delta_in, learning_rate):
        delta_in = self.activation.backward(delta_in)
        b_gradient = np.sum(delta_in, axis=(0, 2, 3))
        b_gradient = b_gradient.reshape((1, -1))
        self.delta_in = delta_in.transpose(0, 2, 3, 1)
        w_gradient = np.tensordot(self.col_image,self.delta_in,axes=([0,1,2],[0,1,2]))
        pad_delta = np.pad(self.delta_in, (
            (0, 0), (self.ksize - 1, self.ksize - 1), (self.ksize - 1, self.ksize - 1), (0, 0)),
                         'constant', constant_values=0)
        pad_delta = self.split_by_strides(pad_delta)
        self.delta_out = np.tensordot(pad_delta, self.weights, axes=([3, 4, 5], [0, 1, 3]))
        self.delta_out = self.delta_out.transpose(0, 3, 1, 2)

        if self.lr is None:
            self.lr = learning_rate
        if self.m is None:
            self.m = np.zeros_like(self.weights)
        if self.v is None:
            self.v = np.zeros_like(self.weights)
        self.n += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * w_gradient
        self.v = self.beta2 * self.v + (1 - self.beta2) * np.square(w_gradient)
        alpha = self.lr * np.sqrt(1 - np.power(self.beta2, self.n))
        alpha = alpha / (1 - np.power(self.beta1, self.n))
        self.weights -= alpha * self.m / (np.sqrt(self.v) + self.eps)

        if self.m_b is None:
            self.m_b = np.zeros_like(self.bias)
        if self.v_b is None:
            self.v_b = np.zeros_like(self.bias)
        self.m_b = self.beta1 * self.m_b + (1 - self.beta1) * b_gradient
        self.v_b = self.beta2 * self.v_b + (1 - self.beta2) * np.square(b_gradient)
        self.bias -= alpha * self.m_b / (np.sqrt(self.v_b) + self.eps)

    def split_by_strides(self, x):
        N, H, W, C = x.shape
        oh = (H - self.ksize) // self.stride + 1
        ow = (W - self.ksize) // self.stride + 1
        shape = (N, oh, ow, self.ksize, self.ksize, C)
        strides = (x.strides[0], x.strides[1] * self.stride, x.strides[2] * self.stride, *x.strides[1:])
        return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)

class AvgPool:
    def __init__(self, size):
        self.size = size
        self.activation = Relu()

    def forward(self, inputs):
        self.outputs = inputs.reshape(inputs.shape[0], inputs.shape[1], inputs.shape[2] // self.size, self.size, inputs.shape[3] // self.size, self.size)
        self.outputs = self.outputs.mean(axis=(3,5))
    def backward(self, delta_in,lr):
        self.delta_in = delta_in
        self.delta_out = (self.delta_in / self.size**2).repeat(self.size, axis=2).repeat(self.size, axis=3)

class Relu():
    def __init__(self):
        pass

    def forward(self, x):
        self.x = x
        return np.maximum(0, x)

    def backward(self, delta_in):
        delta_in[self.x <= 0] = 0
        return delta_in

class SoftMax():
    def __init__(self):
        pass

    def forward(self, x):
        x = x - np.max(x, axis=-1, keepdims = True)
        exp = np.exp(x)
        s = exp / np.sum(exp, axis=-1, keepdims = True)
        self.outputs = s
        return s

    def backward(self, delta_in, lr):
        self.delta_out =  delta_in

class FC():
    def __init__(self, weight_size, activation='relu'):
        self.weight_size = weight_size
        if activation == 'relu':
            self.activation = Relu()
        else:
            self.activation = SoftMax()

        self.lr = None
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-08
        self.m = None
        self.v = None
        self.n = 0
        self.m_b = None
        self.v_b = None

    def init_weights(self):
        H, W = self.weight_size
        stdw = np.sqrt(1 / H)
        stdb = np.sqrt(1 / W)
        self.weights = np.random.normal(0, stdw, (H, W))
        self.bias = np.random.normal(0, stdb, (1, W))

    def forward(self, inputs):
        self.inputs_shape = inputs.shape
        if len(inputs.shape) == 4:
            N, C, H, W = inputs.shape
            inputs = inputs.reshape((N,256))
        self.inputs = inputs
        self.outputu = self.inputs.dot(self.weights) + self.bias
        self.outputs = self.activation.forward(self.outputu)

    def backward(self, delta_in, learning_rate):
        self.delta_in = self.activation.backward(delta_in)
        self.delta_out = np.dot( self.delta_in,self.weights.T)
        self.delta_out = self.delta_out.reshape(self.inputs_shape)
        w_gradient = np.dot(self.inputs.T, self.delta_in)
        b_gradient = np.sum(self.delta_in, axis=0, keepdims=True)

        if self.lr is None:
            self.lr = learning_rate / self.inputs.shape[0]
        if self.m is None:
            self.m = np.zeros_like(self.weights)
        if self.v is None:
            self.v = np.zeros_like(self.weights)
        self.n += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * w_gradient
        self.v = self.beta2 * self.v + (1 - self.beta2) * np.square(w_gradient)
        alpha = self.lr * np.sqrt(1 - np.power(self.beta2, self.n))
        alpha = alpha / (1 - np.power(self.beta1, self.n))
        self.weights -= alpha * self.m / (np.sqrt(self.v) + self.eps)

        if self.m_b is None:
            self.m_b = np.zeros_like(self.bias)
        if self.v_b is None:
            self.v_b = np.zeros_like(self.bias)
        self.m_b = self.beta1 * self.m_b + (1 - self.beta1) * b_gradient
        self.v_b = self.beta2 * self.v_b + (1 - self.beta2) * np.square(b_gradient)
        self.bias -= alpha * self.m_b / (np.sqrt(self.v_b) + self.eps)

class BN():
    def __init__(self, training=True):
        self.gamma=1
        self.beta=0.0
        self.epsilon=1e-5
        self.mean=None
        self.var=None
        self.training = training

    def forward(self,input,axis=1,momentum=0.95):
        if self.training:
            shape=list(input.shape)
            ax=list(np.arange(len(shape)))
            shape.pop(axis)
            ax.pop(axis)
            self.axis=tuple(ax)
            self.m=reduce(lambda x, y: x * y, shape)


            mu=np.mean(input,axis=self.axis,keepdims=True)
            self.xmu=input-mu
            var = np.var(input,axis=self.axis,keepdims=True)
            self.ivar=1/np.sqrt(var+self.epsilon)
            self.xhut=self.xmu*self.ivar

            if self.mean is None: self.mean=mu
            if self.var is None: self.var =var

            self.mean=self.mean*momentum+mu*(1-momentum)
            self.var = self.var * momentum + var * (1 - momentum)

            self.outputu = self.gamma*self.xhut+self.beta
        else:
            self.outputu = self.test(input=input)
        self.outputs = self.outputu

    def test(self,input):
        xmu = input - self.mean
        ivar = 1 / np.sqrt(self.var + self.epsilon)
        xhut = xmu * ivar
        return self.gamma*xhut+self.beta

    def backward(self,delta_in,lr=0.09):
        dxhut=delta_in*self.gamma
        dx1=self.m*dxhut
        dx2=self.ivar**2*np.sum(dxhut*self.xmu,axis=self.axis,keepdims=True)*self.xmu
        dx3=np.sum(dxhut,axis=self.axis,keepdims=True)
        dx=self.ivar/self.m*(dx1-dx2-dx3)

        dbeta=np.sum(delta_in,axis=self.axis,keepdims=True)
        self.beta-=lr*dbeta
        dgmama=np.sum(delta_in*self.xhut,axis=self.axis,keepdims=True)
        self.gamma-=lr*dgmama
        self.delta_out = dx

class DropOut(object):
    def __init__(self,  p=0.2, training=True):
        self.p = p
        self.training = training

    def forward(self, inputs):
        scaler, mask = 1.0, np.ones(inputs.shape).astype(bool)
        if self.training:
            scaler = 1.0 / (1.0 - self.p)
            mask = np.random.rand(*inputs.shape) >= self.p
            inputs = mask * inputs
        self.mask = mask
        self.outputs = scaler * inputs

    def backward(self, delta_in, lr):
        delta_in=self.mask*delta_in
        delta_in *= 1.0 / (1.0 - self.p)
        self.delta_out = delta_in

class LeNet(object):
    def __init__(self):
        np.random.seed(1337)
        # suppose batchsize = 32
        self.net = [Conv(inputs_shape=[32,1,28,28], output_channels=6, ksize=5, stride=1),
                    BN(training=True),
                    AvgPool(size=2),
                    Conv(inputs_shape=[32,6,5,5], output_channels=16, ksize=5, stride=1),
                    BN(training=True),
                    AvgPool(size=2),
                    DropOut(p=0.2, training=True),
                    FC(weight_size=[256, 128], activation='relu'),
                    FC(weight_size=[128, 64], activation='relu'),
                    FC(weight_size=[64, 10], activation='relu'),
                    SoftMax()
                    ]
        self.init_layers = [0, 3, 7, 8, 9]
        self.train_layers = [1, 4, 6]
        print("initialize")

    def init_weight(self):
        for i in self.init_layers:
            self.net[i].init_weights()

    def forward(self, x):
        """前向传播
        x是训练样本， shape是 B,C,H,W
        这里的C是单通道 c=1 因为 MNIST中都是灰度图像
        返回的是最后一层 softmax后的结果
        也就是 以 One-Hot 表示的类别概率
        Arguments:
            x {np.array} --shape为 B，C，H，W
        """
        for k, layer in enumerate(self.net):
            if k == 0:
                layer.forward(x)
            else:
                layer.forward(self.net[k - 1].outputs)
        outputs = self.net[-1].outputs

        return outputs

    def backward(self, error, lr=0.8e-3):
        idx = [i for i in range(len(self.net))]
        idx.reverse()
        for i in idx:
            if i == len(self.net) - 1:
                self.net[i].backward(error, lr)
            else:
                self.net[i].backward(self.net[i+1].delta_out, lr)

    def compute_accuracy(self, output, labels):
        correct_output = np.argmax(output, axis=1)
        correct_labels = np.argmax(labels, axis=1)

        is_correct = [a == b for a, b in zip(correct_output, correct_labels)]

        accuracy = sum(is_correct) * 1. / labels.shape[0]
        return accuracy

    def evaluate(self, x, labels):
        """
        x是测试样本， shape 是BCHW
        labels是测试集中的标注， 为one-hot的向量
        返回的是分类正确的百分比

        在这个函数中，建议直接调用一次forward得到pred_labels,
        再与 labels 做判断

        Arguments:
            x {np array} -- BCWH
            labels {np array} -- B x 10
        """
        if len(x.shape) == 3:
            x = np.expand_dims(x, 1)
        for i in self.train_layers:
            self.net[i].training = False
        #self.net[6].mode = 'test'
        #self.net[9].mode = 'test'
        outputs = self.forward(x)
        accuracy = self.compute_accuracy(outputs, labels)
        return accuracy

    def data_augmentation(self, images):
        '''
        数据增强，可选操作，非强制，但是需要合理
        一些常用的数据增强选项： ramdom scale， translate， color(grayscale) jittering， rotation, gaussian noise,
        这一块儿允许使用 opencv 库或者 PIL image库
        比如把6旋转90度变成了9，但是仍然标签为6 就不合理了
        '''
        return images

    def fit(
        self,
        train_image,
        train_label,
        test_image = None,
        test_label = None,
        epoches = 10,
        batch_size = 16,
        lr = 0.8e-3
    ):
        self.net[0].inputs_shape[0] = batch_size
        self.net[3].inputs_shape[0] = batch_size
        sum_time = 0
        accuracies = []

        self.init_weight()

        lr = lr
        for epoch in range(epoches):
            ## 可选操作，数据增强
            #print('epoch ', epoch)
            train_image = self.data_augmentation(train_image)
            shuffle_ix = np.random.permutation(np.arange(len(train_image)))
            train_image = train_image[shuffle_ix]
            train_label = train_label[shuffle_ix]
            ## 随机打乱 train_image 的顺序， 但是注意train_image 和 test_label 仍需对应

            batch_images = [] # 请实现 step #1
            batch_labels = [] # 请实现 step #1

            for batch in range(len(train_image) // batch_size):
                batch_img = train_image[batch * batch_size:(batch + 1) * batch_size]
                batch_label = train_label[batch * batch_size: (batch + 1) * batch_size]
                batch_images.append(batch_img)
                batch_labels.append(batch_label)

            number = 0

            last = time.time() #计时开始
            for imgs, labels in zip(batch_images, batch_labels):
                number += 1
                #print(number)
                if len(imgs.shape) == 3:
                    imgs = np.expand_dims(imgs, 1)

                outputs = self.forward(imgs)
                assert outputs.shape == labels.shape, 'outshape not match!'
                self.backward(outputs-labels, lr)

            duration = time.time() - last
            sum_time += duration

            if epoch % 5 == 0:
                for i in self.train_layers:
                    self.net[i].training = False

                accuracy = self.evaluate(np.expand_dims(test_image,1), test_label)
                print("epoch{} accuracy{}".format(epoch, accuracy))
                accuracies.append(accuracy)

                for i in self.train_layers:
                    self.net[i].training = True
        avg_time = sum_time / epoches
        return avg_time, accuracies


