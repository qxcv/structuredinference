#!/usr/bin/env python3
"""Turn evaluation file into sequences of videos. Not sure how I'm going to
make this method-agnostic."""

import argparse
import json
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import h5py

import addpaths  # noqa
from plot_seqs import draw_poses
from common_pp.completion_video_common import load_sorted_paths

FRAME_DIR = '/data/home/cherian/IkeaDataset/Frames/'
DB_PATH = '/data/home/cherian/IkeaDataset/IkeaClipsDB_withactions.mat'
POSE_DIR = '/home/sam/etc/cpm-keras/ikea-mat-poses/'

parser = argparse.ArgumentParser()
parser.add_argument('results_h5_path', help='path to .h5 from eval code')
parser.add_argument(
    '--vid-dir',
    type=str,
    default=None,
    help='save videos to this directory instead of showing poses')

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

    with h5py.File(args.results_h5_path, 'r') as fp:
        num_seqs = len(fp['/poses_2d_pred'])
        # pick a random one!
        sel_idx = np.random.randint(num_seqs)
        json_vid_names = json.loads(
            fp['/seq_ids_2d_json'].value.decode('utf8'))
        vid_name = json_vid_names[sel_idx]
        print('Selected sample set %d (video name %s)' % (sel_idx, vid_name))
        seq_frame_inds = fp['/orig_frame_numbers_2d'].value[sel_idx]
        true_poses = fp['/poses_2d_true'].value[sel_idx]
        pred_poses = fp['/poses_2d_pred'].value[sel_idx]
        num_samples = pred_poses.shape[0]
        seq_names = ['True poses'] + \
                    ['Sample %d' % d for d in range(num_samples)]
        pose_seqs = np.stack([true_poses] + [r for r in pred_poses], axis=0)
        parents = fp['/parents_2d'].value
    meta = meta_dict[vid_name]
    path_suffix = meta['path_suffix']
    tmp2_id = meta['tmp2_id']
    # tmp2_id = int(re.match(r'^vid(\d+)$', vid_name).groups()[0])
    all_frame_fns = load_sorted_paths(os.path.join(FRAME_DIR, path_suffix))
    # for some reason there is one video directory with a subdirectory that has
    # a numeric name
    all_frame_fns = [f for f in all_frame_fns if f.endswith('.jpg')]
    frame_paths = [all_frame_fns[i] for i in seq_frame_inds]

    pose_mat_path = os.path.join(POSE_DIR, 'pose_clip_%d.mat' % tmp2_id)
    pose_mat = loadmat(pose_mat_path, squeeze_me=True)

    # important not to let return value be gc'd (anims won't run otherwise!)
    anim = draw_poses(
        'Completed poses in %s' % args.results_h5_path,
        parents,
        pose_seqs,
        frame_paths=[frame_paths] * (1 + num_samples),
        subplot_titles=seq_names,
        fps=50 / 9.0)
    if args.vid_dir is not None:
        # save video
        print('Saving video')
        try:
            os.makedirs(args.vid_dir)
        except FileExistsError:
            pass

        key = vid_name + '-' + str(sel_idx)
        anim.save(
            os.path.join(args.vid_dir, key + '.mp4'),
            writer='avconv',
            # no idea what bitrate defaults to, but empirically it seems to be
            # around 1000 (?)
            bitrate=3000,
            # dpi defaults to 300
            dpi=300,
            # go at one-third of original speed
            fps=50 / (3*3))
    else:
        print('Showing sequence')
        plt.show()
