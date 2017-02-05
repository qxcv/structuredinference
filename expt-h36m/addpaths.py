#!/usr/bin/env python2

"""Initialises paths to data loading scripts."""

import os
import sys

h36m_path = os.path.expanduser('~/repos/pose-prediction/keras')
tm_path = os.path.expanduser('~/repos/theanomodels')
paths = {h36m_path, tm_path}
for path in paths:
    assert os.path.isdir(path), 'code at %s must exist' % path
    if path not in sys.path:
        sys.path.append(path)
