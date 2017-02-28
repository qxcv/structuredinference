#!/usr/bin/env python3
"""Trains an action predictor on estimated IkeaDB poses."""

import os

import numpy as np

from keras.models import Sequential
from keras.layers import Bidirectional, GRU, Dropout, Dense, GaussianNoise
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.regularizers import l2
from keras.utils.np_utils import to_categorical

import addpaths  # noqa
from load import loadDataset

MERGE_ACTIONS = {
    'attach leg':
    ['attach leg 1', 'attach leg 2', 'attach leg 3', 'attach leg 4'],
    'detach leg':
    ['detach leg 1', 'detach leg 2', 'detach leg 3', 'detach leg 4'],
}


def make_model(seq_len, num_channels, num_actions):
    # model from Anoop
    model = Sequential()
    model.add(GaussianNoise(0.05, input_shape=(seq_len, num_channels)))
    model.add(
        Bidirectional(
            GRU(96,
                return_sequences=True,
                dropout_W=0.2,
                dropout_U=0.2,
                W_regularizer=l2(0.001),
                activation='relu',
                init='uniform')))
    model.add(Dropout(0.2))

    model.add(
        Bidirectional(
            GRU(96,
                return_sequences=False,
                dropout_W=0.2,
                dropout_U=0.2,
                W_regularizer=l2(0.001),
                activation='relu',
                init='uniform')))
    model.add(Dropout(0.2))

    model.add(Dense(96, W_regularizer=l2(0.001), activation='relu'))
    model.add(Dropout(0.3))

    model.add(Dense(96, W_regularizer=l2(0.001)))
    model.add(Dropout(0.3))

    model.add(Dense(num_actions, activation='softmax'))
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])

    return model


def merge_actions(act_labels, act_names, merges):
    # TODO: all of this is broken. Need to finish it.
    raise NotImplementedError()

    act_labels = np.asarray(act_labels)
    new_labels = np.empty_like(act_labels)

    # start by making new labels array this will help us figure out which
    # indices should map to which
    inverse_merges = {}
    for target, sources in merges.items():
        for source in sources:
            assert source not in inverse_merges
            inverse_merges[source] = target

    # ind_map should be a len(act_names) array of integers representing labels
    # under the newly constructed labelling system
    ind_map = []


def to_XY(ds):
    num_classes = len(dataset['p2d_action_names'])
    Y_ints = np.array([y for _, y in ds])
    good_samples = Y_ints != 0
    Y = to_categorical(Y_ints, num_classes)

    # TODO: try the following:
    # 1) Reconstruct poses first, then subtract out the mean pose
    # 2) Extend pose setup by concatenating with previous frame (done)
    T, D = ds[0][0].shape
    X = np.empty((len(ds), T, D))
    for idx, pair in enumerate(ds):
        X[idx] = pair[0]

    # add differences from previous time step
    X_delta = X[:, 1:] - X[:, :-1]
    X_cat = np.concatenate((X[:, 1:], X_delta), axis=2)
    assert X_cat.shape == (X.shape[0], X.shape[1] - 1,
                           X.shape[2] * 2), X_cat.shape

    return X_cat[good_samples], Y[good_samples]


if __name__ == '__main__':
    dataset = loadDataset()
    train_X, train_Y = to_XY(dataset['train_aclass_ds'])
    val_X, val_Y = to_XY(dataset['val_aclass_ds'])

    assert train_Y[:, 0].sum() == 0, "should have nothing of class 0"
    assert val_Y[:, 0].sum() == 0, "should have nothing of class 0"

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
            EarlyStopping(monitor='val_acc', patience=50), ModelCheckpoint(
                checkpoint_dir +
                'action-classifier-{epoch:02d}-{val_loss:.2f}.hdf5',
                save_best_only=True)
        ],
        shuffle=True)
