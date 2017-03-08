#!/usr/bin/env python3
"""Turn completion files into actual SxS videos of the completion, the ground
truth, etc. Probably shares a lot with the other completion video scripts."""

import argparse
import json
import os
import re

import h5py

import numpy as np

import matplotlib.pyplot as plt

from scipy.io import loadmat

from scipy.optimize import fmin

import addpaths  # noqa
from plot_2d_seqs import draw_poses
from common_pp.completion_video_common import load_sorted_paths, alignment_constant

FRAME_DIR = '/data/home/cherian/IkeaDataset/Frames/'
FRAME_DIR = '/home/sam/sshfs/paloalto' + FRAME_DIR  # XXX
DB_PATH = '/data/home/cherian/IkeaDataset/IkeaClipsDB_withactions.mat'
DB_PATH = '/home/sam/sshfs/paloalto' + DB_PATH  #  XXX
POSE_DIR = '/home/sam/sshfs/paloalto/etc/cpm-keras/ikea-mat-poses/'  # XXX

parser = argparse.ArgumentParser()
parser.add_argument('completion_path', help='path to .json completion file')

if __name__ == '__main__':
    args = parser.parse_args()

    db = loadmat(DB_PATH, squeeze_me=True)['IkeaDB']
    # could get just one entry (one relevant to our vid) instead of looping
    # over all. oh well
    meta_dict = {}
    for video_entry in db:
        clip_path = video_entry['clip_path']
        prefix = '/data/home/cherian/IkeaDataset/Frames/'
        assert clip_path.startswith(prefix)
        path_suffix = clip_path[len(prefix):]
        # This is same number used to identify pose clip (not sequential!)
        tmp2_id = video_entry['video_id']
        new_name = 'vid%d' % tmp2_id
        meta_dict[new_name] = {'path_suffix': path_suffix, 'tmp2_id': tmp2_id}

    with open(args.completion_path) as fp:
        d = json.load(fp)
    vid_name = d['vid_name']
    meta = meta_dict[vid_name]
    path_suffix = meta['path_suffix']
    tmp2_id = meta['tmp2_id']
    # tmp2_id = int(re.match(r'^vid(\d+)$', vid_name).groups()[0])
    all_frame_fns = load_sorted_paths(os.path.join(FRAME_DIR, path_suffix))
    # for some reason there is one video directory with a subdirectory that has
    # a numeric name
    all_frame_fns = [f for f in all_frame_fns if f.endswith('.jpg')]
    frame_paths = [all_frame_fns[i] for i in d['frame_inds']]

    pose_seqs = np.stack(
        (d['true_poses'], d['prior_poses'], d['posterior_poses']), axis=0)
    seq_names = ['True poses', 'Prior prediction', 'Posterior prediction']

    pose_mat_path = os.path.join(POSE_DIR, 'pose_clip_%d.mat' % tmp2_id)
    pose_mat = loadmat(pose_mat_path, squeeze_me=True)
    ref_pose = pose_mat['pose'][1:8, :, 0].astype('float').T
    alpha, beta = alignment_constant(pose_seqs[0, 0], ref_pose)

    pose_seqs = pose_seqs * alpha + beta[None, None, :, None]

    # important not to let return value be gc'd (anims won't run otherwise!)
    anims = draw_poses(
        'Completed poses in %s' % args.completion_path,
        d['parents'],
        pose_seqs,
        frame_paths=[frame_paths] * 3,
        subplot_titles=seq_names,
        fps=50 / 9.0,
        crossover=d['crossover_time'])
    plt.show()
