#!/usr/bin/env python3
"""Turn completion files into actual SxS videos of the completion, the ground
truth, etc."""

import argparse
import json
import os

import h5py

import numpy as np

import matplotlib.pyplot as plt

from scipy.optimize import fmin

import addpaths  # noqa
from plot_2d_seqs import draw_poses
from common_pp.completion_video_common import load_sorted_paths, alignment_constant

FRAME_DIR = '/data/home/cherian/MPII/Cheng-MPII-Pose-Action/frames/'
FRAME_DIR = '/home/sam/sshfs/paloalto' + FRAME_DIR  # XXX
POSE_DIR = '/home/sam/sshfs/paloalto/etc/cpm-keras/mpii-ca2-mat-poses'  # XXX

parser = argparse.ArgumentParser()
parser.add_argument('completion_path', help='path to .json completion file')

if __name__ == '__main__':
    args = parser.parse_args()

    with open(args.completion_path) as fp:
        d = json.load(fp)
    vid_name = d['vid_name']
    all_frame_fns = load_sorted_paths(os.path.join(FRAME_DIR, vid_name))
    frame_paths = [all_frame_fns[i] for i in d['frame_inds']]

    pose_seqs = np.stack(
        (d['true_poses'], d['prior_poses'], d['posterior_poses']), axis=0)
    seq_names = ['True poses', 'Prior prediction', 'Posterior prediction']

    all_mat_pose_paths = load_sorted_paths(os.path.join(POSE_DIR, vid_name))
    mat_fst_pose_path = all_mat_pose_paths[d['frame_inds'][0]]
    with h5py.File(mat_fst_pose_path) as fp:
        # gives us 2*14
        ref_pose = fp['pose'].value[:, :8].astype('float')
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
