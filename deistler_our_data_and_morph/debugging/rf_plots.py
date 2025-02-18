# from jax import config

# config.update("jax_enable_x64", True)
# config.update("jax_platform_name", "cpu")

import os

# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".6"



import h5py
import hydra
import pickle
import yaml

from omegaconf import DictConfig
import logging
from functools import partial

import jax.numpy as jnp
from jax import jit, vmap, value_and_grad
import jax.debug
import numpy as np
import matplotlib.pyplot as plt
import pickle
import matplotlib as mpl
import optax
import pandas as pd
import tensorflow as tf
import sys
import jaxley as jx
import glob
from skimage.transform import resize

sys.path.append('/gpfs01/euler/User/ssuhai/GitRepos/jaxley_rgc/deistler_our_data_and_morph')

from utils.rf_utils import (
    compute_all_trained_rfs,
)
from train import create_rf_plots,build_avg_recordings, read_data



# define paths
save_base = '/gpfs01/euler/User/ssuhai/GitRepos/jaxley_rgc/deistler_our_data_and_morph/debugging/figs'
results_base = '/gpfs01/euler/User/ssuhai/GitRepos/jaxley_rgc/deistler_our_data_and_morph/results/train_runs'
config_path = "/gpfs01/euler/User/ssuhai/GitRepos/jaxley_rgc/deistler_our_data_and_morph/debugging/config"
path_prefix = "/gpfs01/euler/User/ssuhai/GitRepos/jaxley_rgc/deistler_our_data_and_morph"

def plot_prediction_label_rf_comparison


@hydra.main(config_path=config_path,config_name="rf_plots.yaml") 
def main (cfg: DictConfig) -> None:

    # read metadata from training data
    meta_data_path = os.path.join(results_base,cfg.results_dir, '0','.hydra','config.yaml')
    with open(meta_data_path, 'r') as file:
        training_config = yaml.safe_load(file)  # Use safe_load to avoid potential security risks

    num_datapoints_per_scanfield = training_config['num_datapoints_per_scanfield']
    cell_id = training_config['cell_id']
    rec_ids = training_config['rec_ids']
    try:
        start_n_scan = training_config['start_n_scan']
    except KeyError:
        print("start_n_scan not found in training config, using default value of 100")
        start_n_scan = 100
    try:
        nseg = training_config['nseg']
    except KeyError:
        print("nseg not found in training config, using default value of 4")
        nseg = 4
    
    # load which indices were used in the train split
    with open(os.path.join(results_base,cfg.results_dir, '0','data',f'{cfg.split}_inds.pkl') ,'rb') as f:
        idx_split = pickle.load(f)

    # load data used in training
    stimuli, recordings, setup, all_images = read_data(
            start_n_scan,
            num_datapoints_per_scanfield,
            cell_id,
            rec_ids,
            "noise",
            path_prefix=path_prefix
        )
    # get images in correct order of split
    all_images = all_images[:, :, idx_split]
    all_images = np.transpose(all_images, (2, 0, 1)) # this is to because later in rf funciton it is transposed again


    # get labels
    labels = pd.read_pickle(f"{path_prefix}/results/data/labels_lowpass_{cell_id}.pkl")
    labels = np.vstack(labels['ca'].values).T

    # import the cell
    cell = jx.read_swc(f"{path_prefix}/morphologies/{cell_id}.swc", nseg=nseg, max_branch_len=300.0, min_radius=1.0, assign_groups=True)


    # load loss weights
    assert cfg['split'] in ['train', 'val', 'test']
    loss_weight_file = glob.glob(os.path.join(results_base,cfg.results_dir, '0','predictions_labels',f'loss_weights_*{cfg.split}*.pkl'))
    assert len(loss_weight_file) == 1, f"Found {len(loss_weight_file)} files for {cfg.split}"
    with open(loss_weight_file[0],'rb') as f:
        all_loss_weights = pickle.load(f)

    # Get all prediction files
    prediction_files = glob.glob(os.path.join(results_base,cfg.results_dir, '0', 'predictions_labels',f'predictions_*{cfg.split}*.pkl'))
    file_epochs = [int(f.split('_')[-3]) for f in prediction_files]
    assert cfg['epoch'] in file_epochs or cfg['epoch'] == 'latest', f"Epoch {cfg['epoch']} not found in {file_epochs}"
    if cfg['epoch'] == 'latest':
        epoch = max(file_epochs)
    else:
        epoch = cfg['epoch']


    #  load predictions
    ca_prediction_files_epoch = glob.glob(os.path.join(results_base,cfg.results_dir, '0', 'predictions_labels',
                                     f'predictions*epoch*{epoch}*split*{cfg.split}.pkl'))
    assert len(ca_prediction_files_epoch) == 1, f"Found {len(ca_prediction_files_epoch)} files for epoch {epoch}"
    with open(ca_prediction_files_epoch[0],'rb') as f:
        all_ca_predictions = pickle.load(f)


    avg_rec_dir = os.path.join(results_base,cfg.results_dir, '0', 'results', 'intermediate')
    if os.path.isdir(avg_rec_dir):
        with open(avg_rec_dir+ '/avg_recordings.pkl', "rb") as handle:
            avg_recordings = pickle.load(handle)
    else: 
        os.makedirs(avg_rec_dir,exist_ok=True)
        avg_recordings = build_avg_recordings(
            recordings, rec_ids, nseg, num_datapoints_per_scanfield
        )
        with open(avg_rec_dir + '/avg_recordings.pkl', "wb") as handle:
            pickle.dump(avg_recordings, handle)

    save_dir = os.path.join(save_base, cfg.results_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


    counters = cfg['counters'] 
    save_path_base = os.path.join(save_dir, f"rf_plots_epoch_{epoch}_recs_{''.join(map(str, cfg.rec_ids))}")

    print(f"Creating RF plots for epoch {epoch} and recs {cfg.rec_ids} and rois {counters}")
    create_rf_plots(counters,
                    cell,
                    all_loss_weights,
                    all_ca_predictions,
                    all_images,
                    avg_recordings,
                    setup,
                    epoch,
                    save_path_base,
                    return_ax_and_not_save=False)


if __name__ == "__main__":

    main()