#!/usr/bin/env python3
"""Trains an action predictor on estimated IkeaDB poses."""

import os

import numpy as np

from keras.models import Sequential
from keras.layers import Bidirectional, GRU, Dropout, Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.regularizers import l2
from keras.utils.np_utils import to_categorical

import addpaths  # noqa
from load import loadDataset


def make_model(seq_len, num_channels, num_actions):
    # model from Anoop
    model = Sequential()
    model.add(
        Bidirectional(
            GRU(96,
                return_sequences=True,
                dropout_W=0.1,
                dropout_U=0.1,
                W_regularizer=l2(0.001),
                activation='relu',
                init='uniform'),
            input_shape=(seq_len, num_channels)))
    model.add(Dropout(0.2))

    model.add(
        Bidirectional(
            GRU(96,
                return_sequences=False,
                dropout_W=0.1,
                dropout_U=0.1,
                W_regularizer=l2(0.001),
                activation='relu',
                init='uniform')))
    model.add(Dropout(0.2))

    model.add(Dense(96, W_regularizer=l2(0.001), activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(96, W_regularizer=l2(0.001)))
    model.add(Dropout(0.2))

    model.add(Dense(num_actions, activation='softmax'))
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])

    return model


def to_XY(ds):
    num_classes = len(dataset['p2d_action_names'])
    Y_ints = np.array([y for _, y in ds])
    Y = to_categorical(Y_ints, num_classes)

    T, D = ds[0][0].shape
    X = np.empty((len(ds), T, D))
    for idx, pair in enumerate(ds):
        X[idx] = pair[0]

    return X, Y


if __name__ == '__main__':
    dataset = loadDataset()
    train_X, train_Y = to_XY(dataset['train_aclass_ds'])
    val_X, val_Y = to_XY(dataset['val_aclass_ds'])

    checkpoint_dir = './chkpt-aclass/'
    try:
        os.makedirs(checkpoint_dir)
    except FileExistsError:
        pass

    seq_len, num_channels = train_X.shape[1:]
    num_actions = val_Y.shape[1]
    model = make_model(seq_len, num_channels, num_actions)
    model.fit(
        train_X,
        train_Y,
        batch_size=64,
        nb_epoch=1000,
        validation_data=(val_X, val_Y),
        callbacks=[
            EarlyStopping(), ModelCheckpoint(
                checkpoint_dir +
                'action-classifier-{epoch:02d}-{val_loss:.2f}.hdf5',
                save_best_only=True)
        ],
        shuffle=True)
