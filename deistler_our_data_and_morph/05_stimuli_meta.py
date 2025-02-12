
'''
Example usage
python -m pdb 05_stimuli_meta.py --date "2020-08-29" --exp_num 1 --distance_cutoff 10

'''

print("\nRunning 05_stimuli_meta.py")

from jax import config

config.update("jax_enable_x64", True)
config.update("jax_platform_name", "cpu")

import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".8"
import numpy as np
import matplotlib.pyplot as plt
import h5py
import pandas as pd
import matplotlib as mpl

import jaxley as jx
import argparse
import ast



# Parse command line arguments
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--date', type=str, help='Date of recording')
parser.add_argument('--exp_num', type=str, help='The number of the experiment')
parser.add_argument('--distance_cutoff', type=int, help='The number of the experiment')

args = parser.parse_args()

assert args.date is not None and '-' in args.date, "Please provide valid date eg: 2020-07-08"
assert args.distance_cutoff is not None, "Please provide a distance cutoff"

# Set these to change what cell you want
date = args.date
stimulus = "noise_1500"
exp_num = args.exp_num
cell_id = date + "_" + exp_num
field_stim_extract = "d1"  # Assume stimuli are all the same with each ROI


distance_cutoff = args.distance_cutoff


rec_id: int = 1  # Can pick any here.

cell = jx.read_swc(f"morphologies/{cell_id}.swc", nseg=4, max_branch_len=300.0, min_radius=5.0)
try:
    bc_output_df = pd.read_pickle(f"results/data/off_bc_output_{cell_id}.pkl")
    stim = bc_output_df[bc_output_df["cell_id"] == cell_id]
    stim = stim[stim["rec_id"] == rec_id]
except:
    rec_id = 'd1'
    bc_output_df = pd.read_pickle(f"results/data/off_bc_output_{cell_id}.pkl")
    stim = bc_output_df[bc_output_df["cell_id"] == cell_id]
    stim = stim[stim["rec_id"] == rec_id]

def compute_jaxley_stim_locations(x, y, distance_cutoff=20):
    """For a given (x,y) location, return all branch and compartment inds within a specified distance."""
    min_dists = []
    min_comps = []
    branch_inds_in_pixel = []
    comps_in_pixel = []
    min_dist_of_branch_in_pixel = []

    for i, xyzr in enumerate(cell.xyzr):
        dists = np.sqrt((x - xyzr[:, 0])**2 + (y - xyzr[:, 1])**2)
        is_in_reach = np.min(dists) < distance_cutoff  # 20 um

        if is_in_reach:
            branch_inds_in_pixel.append(i)
            min_dist_of_branch_in_pixel.append(np.min(dists))
            
            argmin_dist = np.argmin(dists)
            if len(dists) > 1:
                comp = argmin_dist / (len(dists) - 1)
            else:
                comp = 0.5
            comps_in_pixel.append(comp)
            
    return branch_inds_in_pixel, comps_in_pixel, min_dist_of_branch_in_pixel




bc_loc_x = stim["x_loc"].to_numpy()
bc_loc_y = stim["y_loc"].to_numpy()
bc_ids = stim["bc_id"].to_numpy()
bcs_which_stimulate = 0

branch_inds_for_every_bc = []
comp_inds_for_every_bc = []
mind_dists_of_branches_for_every_bc = []
bc_ids_per_stim = []

for x, y, id in zip(bc_loc_x, bc_loc_y, bc_ids):
    branches, comps, min_dist_of_branch_in_pixel = compute_jaxley_stim_locations(x, y,distance_cutoff)
    branch_inds_for_every_bc += branches
    comp_inds_for_every_bc += comps
    mind_dists_of_branches_for_every_bc += min_dist_of_branch_in_pixel
    bc_ids_per_stim += [id] * len(branches)
cell_id
stim_df = pd.DataFrame().from_dict(
    {
        "cell_id": cell_id, 
        "bc_id": bc_ids_per_stim, 
        "branch_ind": branch_inds_for_every_bc, 
        "comp": comp_inds_for_every_bc, 
        "dist_from_bc": mind_dists_of_branches_for_every_bc
    }
)
stim_df
stim_df["num_synapses_of_bc"] = stim_df.groupby("bc_id").bc_id.transform(len)
stim_df.to_pickle(f"results/data/stimuli_meta_{cell_id}.pkl")

print("Done running 05_stimuli_meta.py")