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
            GRU(50,
                return_sequences=True,
                dropout_W=0.2,
                dropout_U=0.2,
                W_regularizer=l2(0.001),
                activation='relu',
                init='uniform')))
    model.add(Dropout(0.2))

    model.add(
        Bidirectional(
            GRU(50,
                return_sequences=False,
                dropout_W=0.2,
                dropout_U=0.2,
                W_regularizer=l2(0.001),
                activation='relu',
                init='uniform')))
    model.add(Dropout(0.2))

    model.add(Dense(50, W_regularizer=l2(0.001), activation='relu'))
    model.add(Dropout(0.3))

    model.add(Dense(50, W_regularizer=l2(0.001)))
    model.add(Dropout(0.3))

    model.add(Dense(num_actions, activation='softmax'))
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])

    return model


def balance_aclass_ds(aclass_ds, act_names):
    # find appropriate number of samples for a single action class,
    # then trim "heavy" action classes to have no more than
    # that number of samples
    class_map = np.zeros((len(aclass_ds), len(act_names)))
    for ds_idx, item in enumerate(aclass_ds):
        _, class_num = item
        class_map[ds_idx, class_num] = 1
    support = class_map.sum(axis=0)
    support_target = int(np.min(support))
    to_keep = np.zeros((len(aclass_ds), ))
    for class_num in range(len(act_names)):
        if support[class_num] <= support_target:
            to_keep[class_map[:, class_num] == 1] = 1
        else:
            # drop all but [:median_support] of these
            class_inds, = np.nonzero(class_map[:, class_num])
            perm = np.random.permutation(len(class_inds))[:support_target]
            chosen_inds = class_inds[perm]
            to_keep[chosen_inds] = 1
    rv = []
    for choose_ind in np.nonzero(to_keep)[0]:
        rv.append(aclass_ds[choose_ind])
    return rv


def merge_actions(aclass_ds, merge_map, act_names):
    for class_name in act_names:
        if class_name not in merge_map:
            merge_map[class_name] = class_name
    new_class_names = sorted({
        class_name
        for class_name in merge_map.values() if class_name is not None
    })
    new_class_nums = []
    for class_name in act_names:
        new_name = merge_map[class_name]
        if new_name is None:
            new_num = None
        else:
            new_num = new_class_names.index(new_name)
        new_class_nums.append(new_num)
    new_aclass_ds = []
    for poses, action in aclass_ds:
        new_action = new_class_nums[action]
        if new_action is None:
            continue
        new_aclass_ds.append((poses, new_action))
    return new_class_names, new_aclass_ds


def to_XY(ds):
    num_classes = len(dataset['p2d_action_names'])
    Y_ints = np.array([y for _, y in ds])
    good_samples = Y_ints != 0
    Y = to_categorical(Y_ints, num_classes)

    # TODO: try to reconstruct poses before learning on them. Maybe subtract
    # out mean pose of each sequence.
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


def standardise(features, mean=None, std=None):
    assert features.ndim == 3, \
        "expected features to be 3D (N*T*D), but shape was %s" \
        % (features.shape,)

    if mean is None:
        mean = features.reshape((-1, features.shape[-1])) \
                       .mean(axis=0)

    if std is None:
        std = features.reshape((-1, features.shape[-1])) \
                       .std(axis=0)

    std[std < 1e-5] = 1
    shape_mean = mean[np.newaxis, np.newaxis, ...]
    shape_std = std[np.newaxis, np.newaxis, ...]
    features = (features - shape_mean) / shape_std

    return features, mean, std


if __name__ == '__main__':
    dataset = loadDataset()
    merge_map = {
        'attach leg 1': 'attach leg',
        'attach leg 2': 'attach leg',
        'attach leg 3': 'attach leg',
        'attach leg 4': 'attach leg',
        'detach leg 1': 'detach leg',
        'detach leg 2': 'detach leg',
        'detach leg 3': 'detach leg',
        'detach leg 4': 'detach leg',
        'n/a': None
    }
    old_act_names = dataset['p2d_action_names']
    _, train_aclass_ds \
        = merge_actions(dataset['train_aclass_ds'], merge_map, old_act_names)
    aclass_target_names, val_aclass_ds \
        = merge_actions(dataset['val_aclass_ds'], merge_map, old_act_names)
    train_aclass_ds_bal = balance_aclass_ds(train_aclass_ds,
                                            aclass_target_names)
    val_aclass_ds_bal = balance_aclass_ds(val_aclass_ds, aclass_target_names)
    train_X, train_Y = to_XY(train_aclass_ds_bal)
    val_X, val_Y = to_XY(val_aclass_ds_bal)

    train_X, mean, std = standardise(train_X)
    val_X, _, _ = standardise(val_X, mean, std)

    # assert train_Y[:, 0].sum() == 0, "should have nothing of class 0"
    # assert val_Y[:, 0].sum() == 0, "should have nothing of class 0"

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
