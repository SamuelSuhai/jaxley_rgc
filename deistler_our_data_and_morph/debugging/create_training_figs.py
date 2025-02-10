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
sys.path.append('/gpfs01/euler/User/ssuhai/GitRepos/jaxley_rgc/deistler_our_data_and_morph')
from utils.data_utils import read_data,build_avg_recordings,build_training_data

# Define default save directory. Can be overwritten by command-line argument
save_dir = '/gpfs01/euler/User/ssuhai/GitRepos/jaxley_rgc/deistler_our_data_and_morph/debugging/figs'
results_base = '/gpfs01/euler/User/ssuhai/GitRepos/jaxley_rgc/deistler_our_data_and_morph/results/train_runs'
cwd = os.getcwd()


def plot_loss_over_epochs(loss_path_pkl,save_dir):

    save_path = os.path.join(save_dir, 'loss_over_epochs.pdf')

    with open(loss_path_pkl, 'rb') as f:
        loss = pickle.load(f)

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(loss)
    ax.set_title('Loss over epochs')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')

    fig.savefig(save_path)
    print(f"Figure saved to {save_path}")

    plt.close(fig)  # Close the figure to free memory


def plot_single_prediction_and_labels_on_ax(pred_path_pkl,labels_path_pkl,ax,title=''):
    with open(pred_path_pkl, 'rb') as f:
        pred = pickle.load(f)
    with open(labels_path_pkl, 'rb') as f:
        labels = pickle.load(f)

    colors = plt.cm.viridis(np.linspace(0, 1, pred.shape[0]))
    for datapoint in range(pred.shape[0]):
        color = colors[datapoint]
        ax.plot(pred[datapoint, :], color=color, label=f'Prediction {datapoint}', linestyle='--')
        ax.plot(labels[datapoint, :], color=color, label=f'Labels {datapoint}')

    ax.set_title('Prediction and labels')
    ax.set_xlabel('Time')
    ax.set_ylabel('Activity')
    ax.set_title(title)
    ax.legend()

  


def plot_all_prediction_and_labels(predictions_pkl_dir,save_dir,cfg):

    selected_type = ['train','val','test'][0]
    

    
    start_n_scan = 0
    num_datapoints_per_scanfield = 64
    cell_id = "2020-07-08_1" #"20161028_1"  # "20170610_1", "20161028_1"
    rec_ids = [1]
    nseg = 4

    # Get all prediction files
    prediction_files = [f for f in os.listdir(predictions_pkl_dir) if f.startswith('predictions') and f.endswith(f'{selected_type}.pkl')]


    # get the labels files
    labels_files = [f for f in os.listdir(predictions_pkl_dir) if f.startswith('labels') and f.endswith(f'{selected_type}.pkl')]
    labels_files.sort()

    with open(os.path.join(results_base, cfg.results_dir, '0','data',f'{selected_type}_inds.pkl'), 'rb') as f:
        inds = pickle.load(f)


    stimuli, recordings, setup, noise_full = read_data(
        start_n_scan,
        num_datapoints_per_scanfield,
        cell_id,
        rec_ids,
        "noise",
        '/gpfs01/euler/User/ssuhai/GitRepos/jaxley_rgc/deistler_our_data_and_morph'
    )

    stim_branch_inds = stimuli["branch_ind"].to_numpy()
    stim_comps = stimuli["comp"].to_numpy()

    avg_recordings = build_avg_recordings(
                recordings, rec_ids, nseg, num_datapoints_per_scanfield
            )
    comp_inputs_all = np.zeros((len(avg_recordings), num_datapoints_per_scanfield))
    
    for i, rec in avg_recordings.iterrows():

        
        input_to_comp = stimuli.loc[(stimuli["branch_ind"] == rec["branch_ind"])] 
        idx_of_min = (input_to_comp["comp"] - rec["comp"]).abs().idxmin()
        input_to_comp = np.array(input_to_comp.loc[idx_of_min]['activity'])
        
        comp_inputs_all[i, :] = input_to_comp

    with open(os.path.join(results_base, cfg.results_dir, '0',f'predictions_labels/currents_split_{selected_type}.pkl'), 'rb') as f:
        currents = pickle.load(f)

    # for j in range(comp_inputs_all.shape[0]):
    #     for i in range(currents.shape[1]):
    #         if (currents[:,i] == comp_inputs_all[j,inds[0]]).any():
    #             print(f"Found the comp_inputs_all row {j} in the currents array at index {i}")
                
    
    # breakpoint()



    all_preds = []
    all_labels = []
        
    # get the labels files: there is only one
    full_label_path_pkl = os.path.join(predictions_pkl_dir,labels_files[0])

    for idx in range(len(prediction_files)):


        file = [f for f in prediction_files if f'epoch_{idx}_' in f]
        assert len(file) == 1, f"No unique file found for epoch {idx}"
        
        full_pred_path_pkl = os.path.join(predictions_pkl_dir,file[0])

        with open(full_pred_path_pkl, 'rb') as f:
            pred = pickle.load(f)
        with open(full_label_path_pkl, 'rb') as f:
            labels = pickle.load(f)

        all_preds.append(pred)
        all_labels.append(labels)


    colors = plt.cm.viridis(np.linspace(0, 1, pred.shape[0]))

    for datapoint in range(np.minimum(pred.shape[0],10)):    
        fig, ax = plt.subplots(int(np.ceil(np.sqrt(len(prediction_files)))), int(np.ceil(np.sqrt(len(prediction_files)))), figsize=(50, 50))
        ax = ax.flatten()


        # calculate the current on the compartment
        current_comp_input = comp_inputs_all[:,inds[datapoint]]

        for epoch in range(len(prediction_files)):
            ax[epoch].plot([i for i in range(pred.shape[1])],all_preds[epoch][datapoint, :], color='blue', label=f'Prediction at {datapoint}', linestyle='--')
            ax[epoch].plot([i for i in range(pred.shape[1])],all_labels[epoch][datapoint, :], color='red', label=f'Labels at {datapoint}',linewidth=2)
            ax[epoch].plot([i for i in range(pred.shape[1])],current_comp_input, color='green', label=f'Input to compartments at {datapoint}',linewidth=4)

        
            ax[epoch].set_title('Prediction and labels')
            ax[epoch].set_xlabel('Compartment')
            ax[epoch].set_ylabel('Activity')
            ax[epoch].set_title(f'Epoch {epoch}')
            ax[epoch].legend()

            #

        saving_at = os.path.join(save_dir,f'{selected_type}_pred_and_labels_datapoint_{datapoint}.pdf')
        plt.tight_layout()
        
        fig.savefig(saving_at)
        print(f"Figure of datapont {datapoint} saved to {saving_at}")
        plt.close(fig)  


    # Plot labels and predictions accross time for single compartment across epochs
    for comp_idx in range(pred.shape[1]):
        fig, ax = plt.subplots(int(np.ceil(np.sqrt(len(prediction_files)))), int(np.ceil(np.sqrt(len(prediction_files)))), figsize=(50, 50))
        ax = ax.flatten()

        for epoch in range(len(prediction_files)):
            ax[epoch].plot([i for i in range(pred.shape[0])],all_preds[epoch][:,comp_idx], color='blue', label=f'Prediction at epoch {epoch}', linestyle='--')
            ax[epoch].plot([i for i in range(pred.shape[0])],all_labels[epoch][:,comp_idx], color='red', label=f'Labels at {epoch}',linewidth=2)
            ax[epoch].plot([i for i in range(pred.shape[0])],comp_inputs_all[comp_idx,inds], color='green', label=f'Input to compartment at {epoch}',linewidth=4)

        
            ax[epoch].set_title('Prediction, labels, input current')
            ax[epoch].set_xlabel('Time step idx')
            ax[epoch].set_ylabel('Activity')
            ax[epoch].set_title(f'Epoch {epoch}')
            ax[epoch].legend()

        saving_at = os.path.join(save_dir,f'{selected_type}_pred_and_labels_over_time_compartment_{comp_idx}.pdf')
        plt.tight_layout()
        
        fig.savefig(saving_at)
        print(f"Figure of compartment {comp_idx} saved to {saving_at}")
        plt.close(fig)  


config_path = os.path.join(cwd, 'debugging/config')
assert os.path.exists(config_path), f"Config path {config_path} does not exist"

@hydra.main(config_path=config_path,config_name="create_training_figs") 
def run(cfg: DictConfig) -> None:
    global save_dir 

    assert cfg.plot_type is not None, "plot_type must be specified in config file"
    
    assert cfg.results_dir is not None, "results_dir must be specified in config file"
    save_dir = os.path.join(save_dir, cfg.results_dir.split('/')[-1])
    os.makedirs(save_dir, exist_ok=True)

    
    if cfg['plot_type'] ==  0 or cfg['plot_type'] == 'all':    
    
        loss_path_pkl = os.path.join(results_base,cfg.results_dir, '0', 'loss.pkl')
        plot_loss_over_epochs(loss_path_pkl,save_dir)

    if cfg['plot_type'] == 1 or cfg['plot_type'] == 'all':
        predictions_path_pkl = os.path.join(results_base,cfg.results_dir, '0', 'predictions_labels')
        plot_all_prediction_and_labels(predictions_path_pkl,save_dir,cfg)

    if cfg['plot_type'] == 2 or cfg['plot_type'] == 'all':
        pass


if __name__== '__main__':
    run()

