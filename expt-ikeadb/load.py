"""Load a 2D pose dataset (probably IkeaDB) and (optionally) associated
actions."""

from p2d_loader import load_p2d_data


def loadDataset():
    seq_length = 32
    seq_skip = 3
    data = load_p2d_data(
        './ikea_action_data.h5',
        seq_length,
        seq_skip,
        gap=1,
        val_frac=0.2,
        completion_length=256,
        add_noise=0.9,
        remove_head=True)

    dim_observations = data["train_poses"].shape[2]

    dataset = {}

    dataset['train'] = data["train_poses"]
    dataset['mask_train'] = data["train_mask"]

    dataset['valid'] = data["val_poses"]
    dataset['mask_valid'] = data["val_mask"]

    dataset['test'] = dataset['valid']
    dataset['mask_test'] = dataset['mask_valid']

    dataset['dim_observations'] = dim_observations
    dataset['data_type'] = 'real'

    dataset['p2d_mean'] = data["mean"]
    dataset['p2d_std'] = data["std"]

    dataset['train_cond_vals'] = data["train_actions"]
    dataset['val_cond_vals'] = data["val_actions"]
    dataset['test_cond_vals'] = data["val_actions"]
    dataset['p2d_action_names'] = data["action_names"]

    dataset['p2d_parents'] = data["parents"]

    # for action prediction
    dataset['train_aclass_ds'] = data["train_aclass_ds"]
    dataset['val_aclass_ds'] = data["val_aclass_ds"]

    # for sequence completion
    dataset['train_completions'] = data['train_completions']
    dataset['val_completions'] = data['val_completions']

    print('Shapes of various things:')
    to_check_shape = [
        'train', 'valid', 'test', 'train_cond_vals', 'val_cond_vals',
        'test_cond_vals'
    ]
    for to_shape in to_check_shape:
        print('%s: %s' % (to_shape, dataset[to_shape].shape))
    to_check_len = [
        'train_aclass_ds', 'val_aclass_ds', 'train_completions',
        'val_completions'
    ]
    for name in to_check_len:
        print('%s: %d (list)' % (name, len(dataset[name])))

    return dataset
