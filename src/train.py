# -*- coding: utf-8 -*-
from __future__ import print_function

"""
    Date: 8/28/16
    &copy;2016 Takanori Ogata. All rights reserved.
"""
__author__ = 'ogata'


from keras.datasets import cifar10
from keras.utils import np_utils
from model import create_model
from keras.optimizers import SGD, Adam
from keras.callbacks import LearningRateScheduler


def main():
    batch_size = 64
    nb_classes = 10
    nb_epoch = 300
    num_first_filter = 16
    num_growth = 12
    depth = 40
    output_dim = 10
    dropout_rate = 0.2  # None

    # input image dimensions
    img_rows, img_cols = 32, 32
    # the CIFAR10 images are RGB
    img_channels = 3

    # the data, shuffled and split between train and test sets
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    model = create_model((img_channels, img_rows, img_cols),
                         num_first_filter, num_growth, depth,
                         output_dim, dropout_rate)
    model.summary()
    # sgd = Adam()
    sgd = SGD(lr=0.1, decay=0, momentum=0.9, nesterov=True)

    def _lr_scheduler(epoch):
        if epoch >= 225:
            return 0.001
        if epoch >= 150:
            return 0.01
        return 0.1

    lr_scheduler = LearningRateScheduler(_lr_scheduler)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    print('Not using data augmentation.')
    model.fit(X_train, Y_train,
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              validation_data=(X_test, Y_test),
              shuffle=True,
              callbacks=[lr_scheduler])
    model.save('./output_model')

if __name__ == '__main__':
    main()
