# -*- coding: utf-8 -*-
from __future__ import print_function

"""
    Date: 8/28/16
    &copy;2016 Takanori Ogata. All rights reserved.
"""
__author__ = 'ogata'

from keras.layers import (
    Input, Convolution2D, MaxPooling2D, BatchNormalization, Flatten,
    Dense, Dropout, Activation, merge, AveragePooling2D
)
from keras.models import Model
from keras import regularizers


def bn_relu_conv(nb_filter, nb_row, nb_col, dropout_rate=None, regularizer=None):
    def f(input):
        norm = BatchNormalization(mode=0, axis=1)(input)
        activation = Activation("relu")(norm)
        conv = Convolution2D(nb_filter=nb_filter,
                             nb_row=nb_row, nb_col=nb_col, subsample=(1, 1),
                             init="he_normal", border_mode="same",
                             W_regularizer=regularizer,
                             b_regularizer=regularizer)(activation)
        if dropout_rate:
            conv = Dropout(p=dropout_rate)(conv)
        return conv
    return f


def layer_block(input_layers, num_layers_of_each_block, num_filter,
                dropout_rate=None, regularizer=None):
    num_filters_added = 0
    output_layers = []
    output_layers += input_layers
    for i in range(num_layers_of_each_block):
        if len(output_layers) == 1:
            merged = input_layers[0]
        else:
            merged = merge(output_layers, mode='concat', concat_axis=1)
        x = bn_relu_conv(num_filter, 3, 3, dropout_rate, regularizer)(merged)
        output_layers.append(x)
        num_filters_added += num_filter
    return output_layers, num_filters_added


def transition_layer(input_layers, num_filters, dropout_rate=None,
                     regularizer=None):
    merged = merge(input_layers, mode='concat', concat_axis=1)
    trans = bn_relu_conv(num_filters, 1, 1, dropout_rate, regularizer)(merged)
    pool = AveragePooling2D((2, 2))(trans)
    return pool


def create_model(input_shape, num_first_filter, num_growth, depth, output_dim,
                 dropout_rate):

    if (depth - 4) % 3 != 0:
        raise ValueError('Depth must be 3N + 4. depth: {}'.format(depth))
    num_layers_of_each_block = (depth - 4) / 3

    regularizer = regularizers.WeightRegularizer(l2=1e-4)

    img_input = Input(shape=input_shape, name='input')
    conv1 = Convolution2D(nb_filter=num_first_filter,
                          nb_row=3, nb_col=3, subsample=(1, 1),
                          init="he_normal", border_mode="same",
                          W_regularizer=regularizer,
                          b_regularizer=regularizer)(img_input)

    input_layers = [img_input, conv1]
    num_filters = num_first_filter

    # 1st block
    output_layers, num_added = layer_block(input_layers,
                                           num_layers_of_each_block,
                                           num_growth, dropout_rate,
                                           regularizer)

    # transition
    num_filters += num_added
    pool = transition_layer(output_layers, num_filters, regularizer)

    # 2nd block
    input_layers = [pool]
    output_layers, num_added = layer_block(input_layers,
                                           num_layers_of_each_block,
                                           num_growth, dropout_rate,
                                           regularizer)

    # transition
    num_filters += num_added
    pool = transition_layer(output_layers, num_filters, regularizer)

    # 3rd block
    input_layers = [pool]
    output_layers, num_added = layer_block(input_layers,
                                           num_layers_of_each_block,
                                           num_growth, dropout_rate,
                                           regularizer)
    # transition
    merged = merge(output_layers, mode='concat', concat_axis=1)
    x = BatchNormalization(mode=0, axis=1)(merged)
    # print('out ', x, type(x))
    # import theano.tensor
    # print('shape', theano.tensor.shape(x))
    x = Activation("relu")(x)

    x = AveragePooling2D((8, 8))(x)
    x = Flatten()(x)

    x = Dense(output_dim, init="he_normal")(x)

    model = Model(input=img_input, output=x)
    return model


def test():
    num_first_filter = 16
    num_growth = 12
    depth = 10
    output_dim = 10
    model = create_model((3, 32, 32), num_first_filter, num_growth, depth, output_dim)
    # model.compile(optimizer='sgd', loss='mse')  # dummy
    model.summary()


if __name__ == '__main__':
    test()
