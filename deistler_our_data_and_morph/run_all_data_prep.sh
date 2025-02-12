#!/usr/bin/bash

date="2020-08-29"
exp_num="1"

if [ "$(hostname)"=="ssuhai_GPU1-dj_gpu4" ]; then


echo "Running 1 ..."	
# run setup 
python continue 01_setup.py --date "$date" --exp_num "$exp_num" 


echo "Running 2 ..."    
# lowpass data
python 02_labels_lowpass.py --date "$date" --exp_num "$exp_num" --sampling_method 'subsample'


echo "Running 3 ..."
# meta data
python 03_recording_meta.py --date "$date" --exp_num "$exp_num" 


echo "Running 4 ..."

# get bc output
python 04_bc_output.py --date "$date" --exp_num "$exp_num" 


echo "Running 5 ..."
python 05_stimuli_meta.py --date "$date" --exp_num "$exp_num" --distance_cutoff 10



else
    echo "Not on specified host! Will not run scripts to avoid Head Node usage!"
fi

