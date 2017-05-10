#!/usr/bin/env python2
"""Write out predictions file for this dataset and method. Output will be
processed by ``make_stats.py`` in ``pose-prediction`` repo."""

import addpaths  # noqa: F401

from argparse import ArgumentParser

import h5py
import json
import numpy as np
import os
import shlex
import sys
from theano import config
import tqdm

from stinfmodel_fast import evaluate as DKF_evaluate
from stinfmodel_fast.dkf import DKF
from utils.misc import removeIfExists
import p2d_loader

# do this last because it's in the current dir
sys.path.append(os.getcwd())
from load import loadDataset  # noqa: E402


def f32(x):
    return np.asarray(x, dtype='float32')


def forecast_on_batch(dkf, poses, forecast_length):
    """Extends a batch of pose sequences by the desired forecast length."""
    assert poses.ndim == 3, "Poses should be batch*time*dim"
    prefix_length = poses.shape[1]
    poses = poses.astype('float32')
    batch_size = len(poses)

    forecast = np.zeros(
        (batch_size, forecast_length, poses.shape[-1]), dtype=config.floatX)
    mask = np.ones_like(poses, dtype=config.floatX)
    # ignore mu_z/logcov_z for now. will have to test later whether
    # using mu_z in place of z improves performance
    z, mu_z, logcov_z = DKF_evaluate.infer(dkf, poses, mask)
    # take just last timestep
    z = z[:, -1:]
    # z needs to be 3D (nsamples, time, stochdim)
    assert z.ndim == 3 and z.shape[:2] == (batch_size, 1)

    for t in range(prefix_length, prefix_length + forecast_length):
        # completion based on transition prior only
        mu, logcov = dkf.transition_fxn(z)
        z = DKF_evaluate.sampleGaussian(mu, logcov).astype(config.floatX)
        e = dkf.emission_fxn(z)
        assert e.ndim == 3 and e.shape[:2] == (batch_size, 1)
        forecast[:, t - prefix_length] = e[:, 0]

    return forecast


def forecast_batches(dkf, poses, forecast_length, batch_size=256):
    """Splits input into batches and forecasts on each."""
    assert poses.ndim == 3, poses.shape

    num_batches = int(np.ceil(len(poses) / float(batch_size)))
    forecast_list = []
    for bnum in tqdm.tqdm(range(num_batches)):
        start = bnum * batch_size
        stop = start + batch_size
        batch_poses = poses[start:stop]
        forecast = forecast_on_batch(dkf, batch_poses, forecast_length)
        forecast_list.append(forecast)

    joined_forecasts = np.concatenate(forecast_list, axis=0)

    return joined_forecasts


def parse_dkf_args(runme_path, conf_path, weight_path):
    # DKF uses arguments like "-vm LR -infm structured", etc. This script
    # loads those arguments.
    new_argv = get_args(runme_path)
    new_argv.extend(['-reload', weight_path, '-params', conf_path])
    from parse_args_dkf import parse
    params = parse(new_argv)

    return params


def load_dkf(dataset, runme_path, conf_path, weight_path):
    # This is pretty much just copied from train.py. Mostly voodoo.
    params = parse_dkf_args(runme_path, conf_path, weight_path)

    # Add dataset and NADE parameters, which will become part of the model
    for k in ['dim_observations', 'data_type']:
        params[k] = dataset[k]
    if params['use_nade']:
        params['data_type'] = 'real_nade'

    # Remove from params
    removeIfExists('./NOSUCHFILE')
    reloadFile = params.pop('reloadFile')
    pfile = params.pop('paramFile')
    assert os.path.exists(pfile), pfile + ' not found. Need paramfile'
    dkf = DKF(params, paramFile=pfile, reloadFile=reloadFile)

    return dkf


def get_args(runme_path):
    """Parse a runme.sh script to get train.py arguments."""
    with open(runme_path, 'r') as fp:
        data = fp.read()
    tokens = shlex.split(data)
    # get everything after "train.py", omitting one-char whitespace (which
    # sometimes occur when you have "\<newline>")
    train_idx = tokens.index('train.py')
    assert train_idx >= 0
    rest = tokens[train_idx + 1:]
    no_nl = [t for t in rest if not (len(t) == 1 and t.isspace())]
    return no_nl


def get_all_preds(dkf, dataset, for_cond, for_pred, num_samples, is_2d,
                  seq_ids):
    """Get a bunch of predictions for validation set. Tries to manage memory
    carefully!"""
    N, T, D = for_pred.shape
    flat_samples = np.zeros((N, num_samples, T, D), dtype='float32')
    for sample_num in range(num_samples):
        flat_samples[:, sample_num] = forecast_on_batch(dkf, for_cond, T)
    # squash so that different samples appear in different rows
    by_row = flat_samples.reshape((N * num_samples, T, D))
    del flat_samples
    # gotta make this the same size as flat_samples
    seq_ids_flat = np.concatenate([[r] * num_samples for r in seq_ids])
    if is_2d:
        rec_by_row = dataset.reconstruct_poses(by_row, seq_ids_flat)
    else:
        rec_by_row = dataset.reconstruct_skeletons(by_row, seq_ids_flat)
    del by_row
    rv_shape = (N, num_samples) + rec_by_row.shape[1:]
    return rec_by_row.reshape(rv_shape)


parser = ArgumentParser()
parser.add_argument(
    '--num-samples',
    type=int,
    default=5,
    help='number of predictions to make for each test item')
parser.add_argument('runme_path', help='path to relevant runme.sh script')
parser.add_argument(
    'conf_path', help='path to *-config.pkl file in checkpoints')
parser.add_argument(
    'weight_path', help='path to *-params.h5 file in checkpoints')
parser.add_argument('dest_h5', help='.h5 file to write predictions to')

if __name__ == '__main__':
    args = parser.parse_args()

    print('Loading dataset')
    ds_dict = loadDataset()
    if 'p2d' in ds_dict:
        dataset = ds_dict['p2d']
    else:
        dataset = ds_dict['p3d']

    print('Loading DKF')
    dkf = load_dkf(ds_dict, args.runme_path, args.conf_path, args.weight_path)

    print('Generating eval data')
    is_2d = isinstance(dataset, p2d_loader.P2DDataset)
    pred_usable = None
    if is_2d:
        result = dataset.get_ds_for_eval(train=False, discard_no_annos=True)
        for_cond, for_pred = result['conditioning'], result['prediction']
        pred_scales = result['prediction_scales']
        if dataset.has_sparse_annos:
            pred_usable = result['prediction_valids']
        seq_ids = result['prediction_seq_ids']
        orig_frame_numbers = result['prediction_frame_nums']
        # XXX: These WON'T be scaled correctly, as I'm not passing in video ID.
        # Later code also won't be able to scale because it's missing the
        # per-sequence offset :(
        for_pred_recon = dataset.reconstruct_poses(for_pred, seq_ids)
    else:
        for_cond, for_pred = dataset.get_ds_for_eval(train=False)
        for_pred_recon = dataset.reconstruct_skeletons(for_pred)
    print('Getting predictions')
    dkf_preds = get_all_preds(dkf, dataset, for_cond, for_pred,
                              args.num_samples, is_2d, seq_ids)
    print('Writing predictions')
    with h5py.File(args.dest_h5, 'w') as fp:
        extra_data = {}
        fp['/method_name'] = 'dkf'
        if not is_2d:
            fp['/parents_3d'] = dataset.parents
            fp.create_dataset(
                '/skeletons_3d_true',
                compression='gzip',
                shuffle=True,
                data=for_pred_recon)
            fp.create_dataset(
                '/skeletons_3d_pred',
                compression='gzip',
                shuffle=True,
                data=dkf_preds)
            # fp.create_dataset(
            #     '/seq_ids_3d_json',
            #     compression='gzip',
            #     data=json.dumps(seq_ids.tolist()))
            # fp.create_dataset(
            #     '/orig_frame_numbers_3d',
            #     compression='gzip',
            #     data=orig_frame_numbers)
        else:
            fp['/parents_2d'] = dataset.parents
            fp.create_dataset(
                '/poses_2d_true',
                compression='gzip',
                shuffle=True,
                data=for_pred_recon)
            fp['/scales_2d'] = f32(pred_scales)
            fp.create_dataset(
                '/poses_2d_pred',
                compression='gzip',
                shuffle=True,
                data=dkf_preds)
            extra_data['pck_joints'] = dataset.pck_joints

            # next two params are for making videos of predictions on top of
            # original frames

            # tells us name of sequence in original file which contained poses
            # used for prediction
            fp['/seq_ids_2d_json'] = json.dumps(seq_ids.tolist())
            # tells us the sequence number of each frame in that original file
            fp.create_dataset(
                '/orig_frame_numbers_2d',
                compression='gzip',
                data=orig_frame_numbers)
        if pred_usable is not None:
            fp['/is_usable'] = pred_usable
        fp['/extra_data'] = json.dumps(extra_data)
