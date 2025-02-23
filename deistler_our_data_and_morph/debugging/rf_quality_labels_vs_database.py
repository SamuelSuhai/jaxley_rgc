import os
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
    save_path_base = os.path.join(save_dir, f"rf_plots_labels_recs_{''.join(map(str, cfg.rec_ids))}")

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


# ''' 
# Goals: 
# 1) Apply the RF model and quality control to the labels. 
# '''
# from djimaging.tables.receptivefield.rf_utils import normalize_stimulus_for_rf_estimation
# from djimaging.tables.rf_glm.rf_glm_utils import ReceptiveFieldGLM, plot_rf_summary, quality_test
# import numpy as np
# import matplotlib.pyplot as plt
# from copy import deepcopy
# import glob
# import os 
# import pickle
# import h5py
# import pandas as pd

# '''
# a) Load the labels data 
# '''
# path_prefix = '/gpfs01/euler/User/ssuhai/GitRepos/jaxley_rgc/deistler_our_data_and_morph'
# save_dir = f'{path_prefix}/debugging/figs'
# save_dir = os.path.join(save_dir,'labels','rfs')
# noise_path = f'{path_prefix}/data/SMP_C1_d2_Dnoise.h5'


# data_base_dir = f'{path_prefix}/results/data'
# labels_file = glob.glob(os.path.join(data_base_dir, 'labels*.pkl'))[0]

# date = "2020-08-29"
# exp_num = "1"


# def get_data():
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)

#     # get labels 
#     with open(labels_file, 'rb') as f:
#         labels = pickle.load(f)

#     # load setup 
#     setup = pd.read_pickle(f"{path_prefix}/results/data/setup.pkl")

#     # get stimulus
#     file = h5py.File(noise_path, 'r')
#     noise_stimulus = file["NoiseArray3D"][()]

#     return labels, setup, noise_stimulus


# '''
# b) Apply RF model
# '''
# def compute_rf (dt, 
#                 key, 
#                 trace, 
#                 stim_or_idxs, 
#                 suppress_outputs,
#                 clear_outputs):
#     params = {'rf_glm_params_id': 10,
#             'filter_dur_s_past': 1.2,
#             'filter_dur_s_future': 0.2,
#             'df_ts': (10,),
#             'df_ws': (9,),
#             'betas': (0.005,),
#             'kfold': 0,
#             'metric': 'mse',
#             'output_nonlinearity': 'none',
#             'other_params_dict': {
#                 'frac_test': 0,
#                 'min_iters': 100,
#                 'max_iters': 2000,
#                 'step_size': 0.1,
#                 'tolerance': 5,
#                 'alphas': (1.0,),
#                 'verbose': 100,
#                 'n_perm': 20,
#                 'min_cc': 0.2,
#                 'seed': 42,
#                 'fit_R': False,
#                 'fit_intercept': True,
#                 'init_method': 'random',
#                 'atol': 1e-05,
#                 'distr': 'gaussian',
#                 'step_size_finetune': 0.03}}

#     other_params_dict = params.pop('other_params_dict')
#     if suppress_outputs:
#         other_params_dict['verbose'] = 0

#     assert stim_or_idxs.ndim == 3
#     stim = stim_or_idxs

#     breakpoint()
#     stim = normalize_stimulus_for_rf_estimation(stim)

#     stim = stim.transpose(2, 0, 1)

#     if stim.shape[0] != trace.shape[0]:
#         raise ValueError(f"Stimulus and trace have different number of samples: "
#                             f"{stim.shape[0]} vs. {trace.shape[0]}")

#     model = ReceptiveFieldGLM(
#         dt=dt, trace=trace, stim=stim,
#         filter_dur_s_past=params['filter_dur_s_past'],
#         filter_dur_s_future=params['filter_dur_s_future'],
#         df_ts=params['df_ts'], df_ws=params['df_ws'],
#         betas=params['betas'], kfold=params['kfold'], metric=params['metric'],
#         output_nonlinearity=params['output_nonlinearity'], **other_params_dict)

#     rf, quality_dict, model_dict = model.compute_glm_receptive_field()

#     if clear_outputs or suppress_outputs:
#         from IPython.display import clear_output
#         clear_output(wait=True)

#     if not suppress_outputs:
#         plot_rf_summary(rf=rf, quality_dict=quality_dict, model_dict=model_dict,
#                         title=f"{key['date']} {key['exp_num']} {key['field']} {key['roi_id']}")
#         plt.show()

#     drop_keys = [k for k, v in model_dict.items() if
#                     k.startswith('y_') or isinstance(v, np.ndarray)]
#     for k in drop_keys:
#         model_dict.pop(k)
#         rf_entry = deepcopy(key)
#         rf_entry['rf'] = rf
#         rf_entry['dt'] = model_dict.pop('dt')
#         rf_entry['model_dict'] = model_dict
#         rf_entry['quality_dict'] = quality_dict

#     return rf_entry




# def calculate_rfs():
#     labels, setup, noise_stimulus = get_data()

#     roi_num = labels.shape[0]
#     roi_ids = labels['roi_id']
#     rec_ids = labels['rec_id']

#     rfs = []
#     field = 0
#     roi_id = 1
#     key = {'date': date,'exp_num':exp_num,'field':field,'roi_id':roi_id}

#     current_roi = 0 
#     trace = np.array(labels['ca'][current_roi])

#     # truncate noise stimulus
#     noise_stimulus = noise_stimulus[...,:trace.shape[0]]

#     rf_entry = compute_rf(dt=0.2,
#                         key=key,
#                         trace=trace,
#                         stim_or_idxs=noise_stimulus,
#                         suppress_outputs=False,
#                         clear_outputs=False)
#     rfs.append(rf_entry)

#     return rfs

# rfs = calculate_rfs()
# '''
# c) Get quality measures
# '''


