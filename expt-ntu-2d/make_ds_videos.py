#!/usr/bin/env python3
"""Turn original dataset sequences into videos. Super duper cut-and-pasted."""

import argparse
import json
import os
import random
import re
import subprocess
import shutil
import tempfile
import zipfile

import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread

import addpaths  # noqa
from plot_seqs import draw_poses

H5_PATH = './ntu_data.h5'
ZIPS_DIR = '/data/home/sam/ntu-rgbd/'
ZIPS_DIR = '/home/sam/sshfs/paloalto' + ZIPS_DIR  # XXX

parser = argparse.ArgumentParser()
parser.add_argument(
    '--vid_name', help='name of video sequence to show (otherwise random)')
parser.add_argument(
    '--save',
    default=None,
    metavar='DEST',
    help='if supplied, save to video file instead of showing')


class ZipVideo:
    FRAME_TEMPLATE = '%04d.jpg'

    def __init__(self, zip_path, path_in_zip):
        # read the video
        zf = zipfile.ZipFile(zip_path)
        inner_fp = zf.open(path_in_zip)

        # make temporary file and extract video to it
        ext = os.path.splitext(path_in_zip)[1]
        with tempfile.NamedTemporaryFile(suffix=ext) as intermediate_fp:
            while True:
                # copy across 1MiB at a time
                buf = inner_fp.read(1024 * 1024)
                if not buf:
                    break
                intermediate_fp.write(buf)
            intermediate_fp.flush()

            # just copy the frames out to disk, I give up trying to do anything
            # more fancy
            self.tmp_dir = intermediate_fp.name + '-frames'
            os.makedirs(self.tmp_dir, exist_ok=True)
            subprocess.run(
                [
                    'ffmpeg', '-i', intermediate_fp.name,
                    os.path.join(self.tmp_dir, self.FRAME_TEMPLATE)
                ],
                check=True)

    def get_frame(self, fr_ind):
        # ffmpeg starts at 1, we start at 0
        fr_num = fr_ind + 1
        frame_path = os.path.join(self.tmp_dir, self.FRAME_TEMPLATE % fr_num)
        return imread(frame_path)

    def __del__(self):
        if self.tmp_dir is not None:
            shutil.rmtree(self.tmp_dir)


_vid_name_re = re.compile(r'^S(?P<setup>\d{3})C(?P<camera>\d{3})'
                          r'P(?P<performer>\d{3})R(?P<replication>\d{3})'
                          r'A(?P<action>\d{3})_I(?P<orig_id>\d+)'
                          r'SF(?P<start_frame>\d+)EF(?P<end_frame>\d+)$')


def parse_name(name):
    # takes a sequence name and returns video path, start frame, and end frame
    groups = _vid_name_re.match(name).groupdict()
    # this holds videos for the current setup
    archive_name = 'nturgbd_rgb_s%s.zip' % groups['setup']
    # name of video in archive
    name_pfx, _ = name.split('_', 1)
    video_name = 'nturgb+d_rgb/%s_rgb.avi' % name_pfx
    start_frame = int(groups['start_frame'])
    end_frame = int(groups['end_frame'])
    return archive_name, video_name, start_frame, end_frame


if __name__ == '__main__':
    args = parser.parse_args()
    vid_name = args.vid_name

    with h5py.File(H5_PATH) as fp:
        if vid_name is None:
            vid_name = random.choice(list(fp['/seqs']))
            print("Selected video '%s'" % vid_name)
        poses = fp['/seqs/' + vid_name + '/poses'].value
        action_ids = fp['/seqs/' + vid_name + '/actions'].value
        action_name_map = np.asarray(
            json.loads(fp['/action_names'].value.tostring().decode('utf8')))
        parents = fp['/parents'].value

    # read out all the frames we need
    zip_name, avi_name, start_frame, end_frame = parse_name(vid_name)
    zip_path = os.path.join(ZIPS_DIR, zip_name)
    video = ZipVideo(zip_path, avi_name)
    frames = [
        video.get_frame(frame_idx)
        for frame_idx in range(start_frame, end_frame + 1)
    ]

    pose_seqs = poses[None, ...]
    action_labels = action_name_map[action_ids]
    seq_names = ['True poses']

    assert len(action_labels) == len(poses)
    print('Action names: ' + ', '.join(action_name_map))
    print('Actions: ' + ', '.join(action_labels))

    # important not to let return value be gc'd (anims won't run otherwise!)
    anims = draw_poses(
        'Poses in %s' % vid_name,
        parents,
        pose_seqs,
        frames=[frames],
        subplot_titles=seq_names,
        # always plot at 10fps so that we can actually see the action :P
        fps=10,
        action_labels=action_labels)
    if args.save is not None:
        print('Saving video to %s' % args.save)
        anims.save(args.save)
    else:
        plt.show()
