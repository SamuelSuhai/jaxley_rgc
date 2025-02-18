#!/usr/bin/bash

date="2020-08-29"
exp_num="1"

if [ "$(hostname)"=="ssuhai_GPU1-dj_gpu4" ]; then



else
    echo "Not on specified host! Will not run scripts to avoid Head Node usage!"
fi

