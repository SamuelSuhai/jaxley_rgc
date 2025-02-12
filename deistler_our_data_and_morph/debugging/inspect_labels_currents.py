import pickle
import matplotlib.pyplot as plt

import pickle
import matplotlib.pyplot as plt
import os
import hydra
from omegaconf import DictConfig
import os
import numpy as np
import glob


save_dir = '/gpfs01/euler/User/ssuhai/GitRepos/jaxley_rgc/deistler_our_data_and_morph/debugging/figs'
save_dir = os.path.join(save_dir,'labels')

results_base = '/gpfs01/euler/User/ssuhai/GitRepos/jaxley_rgc/deistler_our_data_and_morph/results/train_runs'
data_base_dir = '/gpfs01/euler/User/ssuhai/GitRepos/jaxley_rgc/deistler_our_data_and_morph/results/data'

labels_file = glob.glob(os.path.join(data_base_dir, 'labels*.pkl'))[0]
bc_activity_file = 'off_bc_output_2020-07-08_1.pkl'

compute_correlation = False

# create results directory 

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# get labels 
with open(labels_file, 'rb') as f:
    labels = pickle.load(f)

rec_ids = labels['rec_id'].unique()


# compute the correlation between labels and bc activity for each BC and each recording 
if compute_correlation:

    # get bc activtiy
    with open(os.path.join(data_base_dir, 'off_bc_output_2020-07-08_1.pkl'), 'rb') as f:
        bc_activity = pickle.load(f)

    for rec_id in rec_ids:
        labels_in_rec = labels.loc[labels['rec_id'] == rec_id]['ca'].to_numpy()
        
        for roi in range(labels_in_rec.shape[0]):
            max_corr = -1
            max_corr_bc = -1

            for bc in range(bc_activity.shape[0]):
                corr = np.corrcoef(labels_in_rec[roi], bc_activity.iloc[bc]['activity'][:1498])
                if corr[0,1] > max_corr:
                    max_corr = corr[0,1]
                    max_corr_bc = bc
            print(f'Maximal correlation between labels rec_id {rec_id} and roi_id {roi} and BC activity for recording {rec_id} is {max_corr} with BC row {max_corr_bc}')            


for rec_id in rec_ids:
    labels_in_rec = labels.loc[labels['rec_id'] == rec_id]

    fig, ax = plt.subplots(labels_in_rec.shape[0], 1, figsize=(20, 10))
    ax = ax.ravel()

    for roi in range(labels_in_rec.shape[0]):
        ax[roi].plot(labels_in_rec.iloc[roi]['ca'])

    print(f"Figure saved to {os.path.join(save_dir, f'labels_{rec_id}.pdf')}")
    fig.savefig(os.path.join(save_dir, f'labels_{rec_id}.pdf'))

