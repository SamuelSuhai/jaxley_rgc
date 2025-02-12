from jax import config

config.update("jax_enable_x64", True)
config.update("jax_platform_name", "cpu")

import os

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".6"



import h5py
import hydra
import pickle

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
from skimage.transform import resize

sys.path.append('/gpfs01/euler/User/ssuhai/GitRepos/jaxley_rgc/deistler_our_data_and_morph')

from utils.rf_utils import (
    compute_all_trained_rfs,
)

import jaxley as jx

path_prefix = "/gpfs01/euler/User/ssuhai/GitRepos/jaxley_rgc/deistler_our_data_and_morph"

cell_id = "2020-08-29_1"
rec_ids = [1,2,3,4]

# get setup
setup = pd.read_pickle(f"{path_prefix}/results/data/setup.pkl")
setup = setup[setup["cell_id"] == cell_id]
setup = setup[setup["rec_id"].isin(rec_ids)]

# get labels
labels = pd.read_pickle(f"{path_prefix}/results/data/labels_lowpass_{cell_id}.pkl")
labels = np.vstack(labels['ca'].values).T

# get noise stimulus
file = h5py.File(f"{path_prefix}/data/noise.h5", 'r+')
noise_full = file["NoiseArray3D"][()]

# import the cell
cell = jx.read_swc(f"{path_prefix}/morphologies/{cell_id}.swc", nseg=4, max_branch_len=300.0, min_radius=1.0, assign_groups=True)


# some more settings
# # Receptive field setup.
# avg_recordings["roi_id"] = avg_recordings["roi_id"].astype(int)
# rec_and_roi = avg_recordings[["rec_id", "roi_id"]].to_numpy()
# rec_ids_of_all_rois = rec_and_roi[:, 0]
# roi_ids_of_all_rois = rec_and_roi[:, 1]
noise_mag = 1e-1
num_iter = 20
# avg_recordings = avg_recordings.reset_index()
center = np.asarray([170, 150])
pixel_size = 30
levels = [0.5]



all_ca_predictions = labels
all_loss_weights = np.ones_like(labels)
all_images = noise_full
epoch = 0 

# Compute receptive fields.
counters = [0] #np.arange(0, 147, 5) # which rois to visualize
rfs_trained = compute_all_trained_rfs(
    counters,
    all_loss_weights,
    all_ca_predictions,
    np.transpose(all_images, (1, 2, 0)),
    noise_mag,
    num_iter,
)

fig, ax = plt.subplots(1, 1, figsize=(4.9, 6.5))

for i, counter in enumerate(counters):
    # changed
    ax = cell.vis(ax=ax) # ,color="k",morph_plot_kwargs={"zorder": 1000, "linewidth": 0.3})

    rec_id = rec_ids_of_all_rois[counter]
    roi_id = roi_ids_of_all_rois[counter]
    rf_pred = rfs_trained[i]
    setup_rec = setup[setup["rec_id"] == rec_id]
    offset_x = setup_rec["image_center_x"].to_numpy()[0]
    offset_y = setup_rec["image_center_y"].to_numpy()[0]

    upsample_factor = 5
    im_pos_x = (
        np.linspace(
            -7.0 * pixel_size, 7.0 * pixel_size, 15 * upsample_factor
        )
        + offset_x
    )
    im_pos_y = (
        -np.linspace(
            -9.5 * pixel_size, 9.5 * pixel_size, 20 * upsample_factor
        )
        + offset_y
    )
    image_xs, image_ys = np.meshgrid(im_pos_x, im_pos_y)

    rec = avg_recordings.loc[counter]
    dist = np.sqrt(
        np.sum(
            (
                center
                - np.asarray([rec["roi_x"].item(), rec["roi_y"].item()])
            )
            ** 2
        )
    )
    cmap = mpl.colormaps["viridis"]
    col = cmap((dist + 20) / 150)

    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    _ = ax.scatter(
        rec["roi_x"].item(),
        rec["roi_y"].item(),
        color=col,
        s=20.0,
        edgecolors="k",
        zorder=10000,
    )

    # Contours
    output_shape = (np.array([20, 15]) * upsample_factor).astype(int)
    upsampled_rf = resize(
        rf_pred, output_shape=output_shape, mode="constant"
    )
    _ = ax.contour(
        image_xs,
        image_ys,
        upsampled_rf,
        levels=levels,
        colors=[col],
        linestyles="solid",
        linewidths=0.5,
    )
    _ = ax.set_xticks([])
    _ = ax.set_yticks([])

plt.savefig(
    f"{os.getcwd()}/figs/rf_{epoch}.png", dpi=150, bbox_inches="tight"
)

# visualize a heatmap of the receptive fields
fig, ax = plt.subplots(1, 1, figsize=(4.9, 6.5))
for i, counter in enumerate(counters):
    # changed
    ax = cell.vis(ax=ax) # ,color="k",morph_plot_kwargs={"zorder": 1000, "linewidth": 0.3})

    rec_id = rec_ids_of_all_rois[counter]
    roi_id = roi_ids_of_all_rois[counter]
    rf_pred = rfs_trained[i]
    setup_rec = setup[setup["rec_id"] == rec_id]
    offset_x = setup_rec["image_center_x"].to_numpy()[0]
    offset_y = setup_rec["image_center_y"].to_numpy()[0]

    ax.imshow(rf_pred, cmap="viridis")

plt.savefig(
    f"{os.getcwd()}/figs/rf_heatmap_{epoch}.png")
