#!/usr/bin/bash

if [ "$(hostname)"=="ssuhai_GPU1-dj_gpu4" ]; then


echo "Running 1 ..."	
# run setup 
python 01_setup.py > logs/log.txt 2>&1


echo "Running 2 ..."    
# lowpass data
python 02_labels_lowpass.py >> logs/log.txt 2>&1


echo "Running 3 ..."
# meta data
python 03_recording_meta.py >> logs/log.txt 2>&1


echo "Running 4 ..."
# get bc output
python 04_bc_output.py >> logs/log.txt 2>&1

else
    echo "Not on specified host! Will not run scripts to avoid Head Node usage!"
fi

