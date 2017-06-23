#!/usr/bin/env python3
"""Turn original dataset sequences into videos. Super duper cut-and-pasted."""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import zipfile

import h5py

import numpy as np

import matplotlib.pyplot as plt

import cv2

import addpaths  # noqa
from plot_seqs import draw_poses

H5_PATH = './ntu_data.h5'
ZIPS_DIR = '/data/home/sam/ntu-rgbd/'
ZIPS_DIR = '/home/sam/sshfs/paloalto' + ZIPS_DIR  # XXX

parser = argparse.ArgumentParser()
parser.add_argument('vid_name', help='name of video sequence to show')


class CV2Video(object):
    def __init__(self, file_path):
        self.cap = cv2.VideoCapture(file_path)
        if not self.cap.isOpened():
            exists = os.path.exists(file_path)
            print(
                "Couldn't read '%s', file exists? %s" % (file_path, 'yes'
                                                         if exists else 'no'),
                file=sys.stderr)
            raise IOError("Failed to open '%s' (OpenCV doesn't give any error "
                          "message beyond the implicit \"fuck you\" of "
                          "failure)" % file_path)
        self.n_frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)

    def get_frame(self, fr_ind):
        assert 0 <= fr_ind < self.n_frames, 'frame %d out of range [0, %d)' \
            % (fr_ind, self.n_frames)
        assert self.cap.set(cv2.CAP_PROP_POS_FRAMES, fr_ind), \
            "could not skip to frame %d" % fr_ind
        succ, frame = self.cap.read()
        assert succ, "frame-reading failed on frame %d" % fr_ind
        # OpenCV has weird BGR loading convention; need RGB instead
        rgb_frame = frame[:, :, ::-1]
        return rgb_frame


class ZipCV2Video(CV2Video):
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

            # transcode to .mp4 (was in DivX, the RAR of videos)
            self.tmp_fp = tempfile.NamedTemporaryFile(suffix='.mjpg')
            subprocess.run(
                ['ffmpeg', '-y', '-i', intermediate_fp.name, self.tmp_fp.name],
                check=True)

        # initialise with the new file
        super().__init__(self.tmp_fp.name)


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
    assert int(cv2.__version__.split('.')[0]) >= 3, \
        "Only tested with OpenCV 3"
    args = parser.parse_args()
    vid_name = args.vid_name

    with h5py.File(H5_PATH) as fp:
        poses = fp['/seqs/' + vid_name + '/poses'].value
        action_ids = fp['/seqs/' + vid_name + '/actions'].value
        action_name_map = np.asarray(
            json.loads(fp['/action_names'].value.tostring().decode('utf8')))
        parents = fp['/parents'].value

    # read out all the frames we need
    zip_name, avi_name, start_frame, end_frame = parse_name(vid_name)
    zip_path = os.path.join(ZIPS_DIR, zip_name)
    video = ZipCV2Video(zip_path, avi_name)
    frames = [
        video.get_frame(frame_idx)
        for frame_idx in range(start_frame, end_frame)
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
        fps=50 / 9.0,
        action_labels=action_labels)
    plt.show()
