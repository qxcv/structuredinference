#!/usr/bin/env python3
"""Trains an action predictor on estimated IkeaDB poses."""

import addpaths  # noqa
from load import loadDataset
from common_pp.act_class_model import train_act_class_model


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
    train_act_class_model(dataset, merge_map)
