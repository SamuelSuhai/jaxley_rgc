
'''
Creates the recording_meta.pkl file which contains 
info on which jaxley branch the recording rois were located 
to the closest. So the output has one row for each ROI in reach 
recording field.

Note: its important that there only be one .swc file per cell in
morpohologies folder.

'''
from jax import config

config.update("jax_enable_x64", True)
config.update("jax_platform_name", "cpu")

import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".8"
import jax.numpy as jnp
from jax import jit, vmap
import numpy as np
import matplotlib.pyplot as plt
import pickle
import h5py
import pandas as pd
import matplotlib as mpl
from scipy.ndimage import rotate

import jaxley as jx
from jaxley.channels import HH

import argparse
import ast




# Parse command line arguments
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--date', type=str, help='Date of recording')
parser.add_argument('--exp_num', type=str, help='The number of the experiment')


args = parser.parse_args()

assert args.date is not None and '-' in args.date, "Please provide valid date eg: 2020-07-08"




# Set these to change what cell you want
date = args.date
stimulus = "noise_1500"
exp_num = args.exp_num
cell_id = date + "_" + exp_num


# Set directory 
home_directory = os.path.expanduser("~")
base_dir = f'{home_directory}/GitRepos/jaxley_rgc/deistler_our_data_and_morph'

def compute_jaxley_branch(roi_pos,cell):
    min_dists = []
    min_comps = []
    for xyzr in cell.xyzr:
        dists = np.sum((roi_pos[:3] - xyzr[:, :3])**2, axis=1)
        min_dist = np.min(dists)
        argmin_dist = np.argmin(dists)
        if len(xyzr) > 1:
            comp_of_min = argmin_dist / (len(xyzr) - 1)
        else:
            comp_of_min = 0.5
        min_dists.append(min_dist)
        min_comps.append(comp_of_min)
        
    return np.argmin(min_dists), min_comps[np.argmin(min_dists)]

def main ():    
    fnames = []
    for (dirpath, dirnames, filenames) in os.walk(f"{base_dir}/morphologies"):
        fnames.extend(filenames)

    setup_df = pd.read_pickle(f"{base_dir}/results/data/setup.pkl")
    write_dfs = []

    for morph_full in fnames:
        print(f'Processing {morph_full}')
        df = setup_df[setup_df["cell_id"] == cell_id]
        cell = jx.read_swc(f"{base_dir}/morphologies/{morph_full}", nseg=4, max_branch_len=300.0, min_radius=1.0)
        
        for index, pos in df[["roi_x", "roi_y", "roi_z", "cell_id", "rec_id", "roi_id"]].iterrows():
            write_df = pd.DataFrame()
            jaxley_branch, jaxley_compartment = compute_jaxley_branch(pos.to_numpy(),cell)
            write_df["cell_id"] = [pos["cell_id"]]
            write_df["rec_id"] = [pos["rec_id"]]
            write_df["roi_id"] = [pos["roi_id"]]
            write_df["roi_x"] = [pos["roi_x"]]
            write_df["roi_y"] = [pos["roi_y"]]
            write_df["roi_z"] = [pos["roi_z"]]
            write_df["branch_ind"] = [int(jaxley_branch)]
            write_df["comp"] = [jaxley_compartment]

            write_dfs.append(write_df)
        
    write_dfs = pd.concat(write_dfs).reset_index(drop=True)
    write_dfs.to_pickle(f"{base_dir}/results/data/recording_meta.pkl")

if __name__ == "__main__":
    print("\nRunning 03_recording_meta.py")

    main()

    print("\nFinished running 03_recording_meta.py")