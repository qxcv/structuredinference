#!/bin/bash

extra_args=""
if [ ! -z "$@" ]; then
    if [ $* == "--debug" ]; then
        extra_args="$extra_args -m ipdb"
    fi
fi
exec python2.7 $extra_args train.py -vm LR -infm structured -ds 50 -dh 50 -uid ntu-2d \
     -reload chkpt-ntu-2d/DKF_lr-8_0000e-04-vm-LR-inf-structured-dh-50-ds-50-nl-relu-bs-20-ep-2000-rs-600-ttype-simple_gated-etype-mlp-previnp-False-ar-1_0000e+01-rv-5_0000e-02-nade-False-nt-5000-cond-False-ntu-2d-EP200-params.npz \
     -params chkpt-ntu-2d/DKF_lr-8_0000e-04-vm-LR-inf-structured-dh-50-ds-50-nl-relu-bs-20-ep-2000-rs-600-ttype-simple_gated-etype-mlp-previnp-False-ar-1_0000e+01-rv-5_0000e-02-nade-False-nt-5000-cond-False-ntu-2d-config.pkl
