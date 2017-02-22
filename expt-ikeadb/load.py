"""Load a 2D pose dataset (probably IkeaDB) and (optionally) associated
actions."""

from collections import namedtuple
import json

import h5py

import numpy as np

from p2d_loader import preprocess_sequence, extract_action_dataset

VAL_FRAC = 0.2

Data = namedtuple('Data', [
    'train_poses', 'train_actions', 'val_poses', 'val_actions', 'mean', 'std',
    'action_names', 'train_vids', 'val_vids', 'data_path', 'parents',
    'train_aclass_ds', 'val_aclass_ds'
])


def load_data(data_file, seq_length, seq_skip):
    train_pose_blocks = []
    train_action_blocks = []
    train_aclass_ds = []
    val_pose_blocks = []
    val_action_blocks = []
    val_aclass_ds = []

    # for deterministic val set split
    srng = np.random.RandomState(seed=8904511)

    with h5py.File(data_file, 'r') as fp:
        parents = fp['/parents'].value
        num_actions = fp['/num_actions'].value.flatten()[0]

        action_json_string = fp['/action_names'].value.tostring().decode(
            'utf8')
        action_names = ['n/a'] + json.loads(action_json_string)

        vid_names = list(fp['seqs'])
        val_vid_list = list(vid_names)
        srng.shuffle(val_vid_list)
        val_count = max(1, int(VAL_FRAC * len(val_vid_list)))
        val_vids = set(val_vid_list[:val_count])
        train_vids = set(val_vid_list) - val_vids

        for vid_name in fp['seqs']:
            actions = fp['/seqs/' + vid_name + '/actions'].value
            # `cert` chance of choosing correct action directly, `1 - cert`
            # chance of choosing randomly (maybe gets correct action)
            cert = 0.6
            one_hot_acts = (1 - cert) * np.ones(
                (len(actions), num_actions + 1)) / (num_actions + 1)
            # XXX: This is an extremely hacky way of injecting noise :/
            one_hot_acts[(range(len(actions)), actions)] += cert
            # actions should form prob dist., roughly
            assert np.all(np.abs(1 - one_hot_acts.sum(axis=1)) < 0.001)

            poses = fp['/seqs/' + vid_name + '/poses'].value
            relposes = preprocess_sequence(poses, parents, smooth=True)

            assert len(relposes) == len(one_hot_acts)

            aclass_list = extract_action_dataset(relposes, actions)
            if vid_name in val_vids:
                val_aclass_ds.extend(aclass_list)
            else:
                train_aclass_ds.extend(aclass_list)

            for i in range(len(relposes) - seq_skip * seq_length + 1):
                pose_block = relposes[i:i + seq_skip * seq_length:seq_skip]
                act_block = one_hot_acts[i:i + seq_skip * seq_length:seq_skip]

                if vid_name in val_vids:
                    train_pose_blocks.append(pose_block)
                    train_action_blocks.append(act_block)
                else:
                    val_pose_blocks.append(pose_block)
                    val_action_blocks.append(act_block)

    train_poses = np.stack(train_pose_blocks, axis=0).astype('float32')
    train_actions = np.stack(train_action_blocks, axis=0).astype('float32')
    val_poses = np.stack(val_pose_blocks, axis=0).astype('float32')
    val_actions = np.stack(val_action_blocks, axis=0).astype('float32')

    flat_poses = train_poses.reshape((-1, train_poses.shape[-1]))
    mean = flat_poses.mean(axis=0).reshape((1, 1, -1))
    std = flat_poses.std(axis=0).reshape((1, 1, -1))
    # TODO: Smarter handling of std. Will also need to use smarter
    # handling in actual loader script used by train.py
    std[std < 1e-5] = 1
    train_poses = (train_poses - mean) / std
    val_poses = (val_poses - mean) / std

    return Data(train_poses, train_actions, val_poses, val_actions, mean, std,
                action_names, sorted(train_vids), sorted(val_vids), data_file,
                parents, train_aclass_ds, val_aclass_ds)


def loadDataset():
    seq_length = 32
    seq_skip = 3
    data = load_data('./ikea_action_data.h5', seq_length, seq_skip)

    dim_observations = data.train_poses.shape[2]

    dataset = {}

    dataset['train'] = data.train_poses
    dataset['mask_train'] = np.ones(data.train_poses.shape[:2])

    dataset['valid'] = data.val_poses
    dataset['mask_valid'] = np.ones(data.val_poses.shape[:2])

    dataset['test'] = dataset['valid']
    dataset['mask_test'] = dataset['mask_valid']

    dataset['dim_observations'] = dim_observations
    dataset['data_type'] = 'real'

    dataset['p2d_mean'] = data.mean
    dataset['p2d_std'] = data.std

    dataset['train_cond_vals'] = data.train_actions
    dataset['val_cond_vals'] = data.val_actions
    dataset['test_cond_vals'] = data.val_actions
    dataset['p2d_action_names'] = data.action_names

    dataset['p2d_parents'] = data.parents

    # for action prediction
    dataset['train_aclass_ds'] = data.train_aclass_ds
    dataset['val_aclass_ds'] = data.val_aclass_ds

    print('Shapes of various things:')
    to_check = [
        'train', 'valid', 'test', 'train_cond_vals', 'val_cond_vals',
        'test_cond_vals'
    ]
    for to_shape in to_check:
        print('%s: %s' % (to_shape, dataset[to_shape].shape))
    for name in ['train_aclass_ds', 'val_aclass_ds']:
        print('%s: %d (list)' % (name, len(dataset[name])))

    return dataset
