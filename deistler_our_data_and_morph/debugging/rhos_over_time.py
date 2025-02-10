import pickle
import matplotlib.pyplot as plt

import pickle
import matplotlib.pyplot as plt
import os
import hydra
from omegaconf import DictConfig
import os
import numpy as np
import sys

results_base = '/gpfs01/euler/User/ssuhai/GitRepos/jaxley_rgc/deistler_our_data_and_morph/results/train_runs'
results_dir = 'bc_dist_15_labels_64_pts'
rho_dir = os.path.join(results_base, results_dir, '0','rhos')
with open(os.path.join(rho_dir, 'train_rho.pkl'), 'rb') as f:
    rhos = pickle.load(f)

breakpoint()
