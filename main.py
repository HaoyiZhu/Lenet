from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from lenet import LeNet

def normalize_image(images):
    ''' 对图像做归一化处理 '''
    mean = np.mean(images)
    std = np.std(images)
    images = np.float64(images)
    images -= mean
    images /= std
    # maybe pca will be better?
    return images

def one_hot_labels(labels):
    '''
    将labels 转换成 one-hot向量
    eg:  label: 3 --> [0,0,0,1,0,0,0,0,0,0]
    '''
    one_hot_labels = np.zeros((labels.size, 10))
    one_hot_labels[np.arange(labels.size), labels.astype(int)] = 1
    return one_hot_labels


def main():
    # image shape: N x H x W, pixel [0, 255]
    # label shape: N x 10
    with np.load('mnist.npz', allow_pickle=True) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']

    print(x_train.shape, x_train[0].max(), x_train[0].min()) #(60000, 28, 28) 255 0 5
    print(x_test.shape, x_test[0].max(), x_test[0].min()) #(10000, 28, 28) 255 0 7

    x_train = normalize_image(x_train)
    x_test = normalize_image(x_test)
    y_train = one_hot_labels(y_train)
    y_test = one_hot_labels(y_test)


    lr = 1.5e-4
    batch_size = 8

    net = LeNet()

    avgtime, accuracy = \
        net.fit(x_train, y_train, x_test, y_test, epoches=40, batch_size=batch_size, lr=lr)
    accu = net.evaluate(x_test, labels=y_test)
    print('avgtime: ', avgtime)
    #print('accuracy: ', accuracy)
    '''
    plt.plot(accuracy)
    plt.savefig('lr={}.jpg'.format(lr))
    plt.show()
	'''
    print("final accuracy {}".format(accu))


if __name__ == "__main__":
    main()


