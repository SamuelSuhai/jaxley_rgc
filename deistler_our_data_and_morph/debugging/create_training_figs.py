import pickle
import matplotlib.pyplot as plt
import matplotlib
import pickle
import matplotlib.pyplot as plt
import os
import hydra
from omegaconf import DictConfig
import os
import numpy as np
import sys
import glob
sys.path.append('/gpfs01/euler/User/ssuhai/GitRepos/jaxley_rgc/deistler_our_data_and_morph')
from utils.data_utils import read_data,build_avg_recordings,build_training_data



# Define default save directory. Can be overwritten by command-line argument
save_dir = '/gpfs01/euler/User/ssuhai/GitRepos/jaxley_rgc/deistler_our_data_and_morph/debugging/figs'
results_base = '/gpfs01/euler/User/ssuhai/GitRepos/jaxley_rgc/deistler_our_data_and_morph/results/train_runs'
cwd = os.getcwd()
matplotlibrc_dir = '/gpfs01/euler/User/ssuhai/GitRepos/jaxley_rgc'

matplotlib.rc_file(f"{matplotlibrc_dir}/.matplotlibrc")

'''
Example usage:

python debugging/create_training_figs.py results_dir="2020-08-29_64_pts_bs_1" cell_id=2020-08-29_1 rec_ids=[2] start_n_scan=100 num_datapoints_per_scanfield=64
python debugging/create_training_figs.py results_dir="2020-08-29_1024_pts_4_epochs" cell_id=2020-08-29_1 rec_ids=[2] start_n_scan=100 num_datapoints_per_scanfield=1024
python debugging/create_training_figs.py results_dir="2020-08-29_512_pts_rec_ids_2" cell_id=2020-08-29_1 rec_ids=[2] start_n_scan=100 num_datapoints_per_scanfield=512 


'''





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



def retrieve_rhos_meas_accorss_epochs(rhos_dir,selected_type):

    all_rhos_files = glob.glob(os.path.join(rhos_dir, f"{selected_type}_rho_all_epoch*.pkl"))
    assert len(all_rhos_files) > 0, f"No files found for {selected_type} in {rhos_dir}"

    all_meas_files = glob.glob(os.path.join(rhos_dir, f"{selected_type}_mae_all_epoch*.pkl"))
    assert len(all_meas_files) > 0, f"No files found for {selected_type} in {rhos_dir}"
    
    epochs = []
    for f in all_rhos_files:
        epoch = int(f.split("_")[-1].split(".")[0])
        epochs.append(epoch)
    epochs = sorted(list(set(epochs))) 

    all_rhos = []
    all_meas = []
    for epoch in epochs:
        rhos_file = [f for f in all_rhos_files if f"epoch_{epoch}" in f][0]
        meas_file = [f for f in all_meas_files if f"epoch_{epoch}" in f][0]

        with open(os.path.join(rhos_dir, rhos_file), "rb") as f:
            rhos = pickle.load(f)
        with open(os.path.join(rhos_dir, meas_file), "rb") as f:
            meas = pickle.load(f)
        assert len(rhos) == len(meas), f"Length of rhos and meas not equal for epoch {epoch}"

        
        all_rhos.append(rhos)
        all_meas.append(meas)
    
    assert len(all_rhos) == max(epochs) + 1 == len(all_meas), "Number of epochs not equal to number of rhos or meas files"

    return np.array(all_meas), np.array(all_rhos)



def plot_rhos_meas_accorss_epochs(rhos_dir,save_dir,cfg):
    save_base = os.path.join(save_dir, 'rhos_meas_over_epochs')
    os.makedirs(save_base, exist_ok=True)

    
    selected_type = cfg['selected_type']

    all_meas, all_rhos = retrieve_rhos_meas_accorss_epochs(rhos_dir,selected_type)
    assert all_meas.shape ==all_rhos.shape, "Shape of rhos and meas not equal"
    epochs = np.arange(all_meas.shape[0])
    
    # check if rec id specified
    if 'rec_ids_of_roi' not in globals():
        rec_ids_of_roi = None
    else:
        rec_ids_of_roi = globals()['rec_ids_of_roi']


    # MEA
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    markers = ['o','s','^','x','D','p','h','*','v','<','>']
    colors = plt.cm.tab10.colors
    meas_mean = np.mean(all_meas,axis=1)
    
    unique_rec_ids = np.unique(rec_ids_of_roi)
    
    for comp in range(all_meas.shape[1]):
        rec_id = rec_ids_of_roi[comp]
        marker_idx = np.where(unique_rec_ids == rec_id)[0][0]
        ax.plot(epochs,all_meas[:,comp], label=f'Compartment {comp}',marker=markers[marker_idx],color=colors[rec_id])
    # add mean
    ax.plot(epochs,meas_mean, label=f'Mean',marker='o',color='black',linewidth=3)
    

    ax.legend()
    ax.set_title('Mean absolute error over epochs')
    ax.set_ylabel('Mean absolute error')
    ax.set_xlabel('Epoch')
    ax.set_xticks(epochs)
    ax.set_xticklabels(epochs)
    print(f"Saving MEA figure to {os.path.join(save_base, f'{selected_type}_mean_absolute_error_over_epochs.pdf')}")
    fig.savefig(os.path.join(save_base, f'{selected_type}_mean_absolute_error_over_epochs.pdf'))
    plt.close(fig)


    # RHO
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    markers = ['o','s','^','x','D','p','h','*','v','<','>']
    colors = plt.cm.tab10.colors
    rhos_mean = np.mean(all_rhos,axis=1)
    unique_rec_ids = np.unique(rec_ids_of_roi)
    
    for comp in range(all_rhos.shape[1]):
        rec_id = rec_ids_of_roi[comp]
        ax.plot(epochs,all_rhos[:,comp], label=f'Compartment {comp}',marker=markers[np.where(unique_rec_ids == rec_id)[0][0]],color=colors[rec_id])
   
    ax.plot(epochs,rhos_mean, label=f'Mean',marker='o',color='black')
    
    ax.legend()
    ax.set_ylabel('Correlation coefficient')
    ax.set_xlabel('Epoch')
    ax.set_title('Correlation coefficient over epochs')
    ax.set_xticks(epochs)
    ax.set_xticklabels(epochs)
    print(f"Saving RHO figure to {os.path.join(save_base, f'{selected_type}_correlation_coefficient_over_epochs.pdf')}")
    fig.savefig(os.path.join(save_base, f'{selected_type}_correlation_coefficient_over_epochs.pdf'))
    plt.close(fig)

  

def get_stimuli_labels_predictions(predictions_pkl_dir,cfg):
    selected_type = cfg['selected_type']
    

    # Get all prediction files
    prediction_files = [f for f in os.listdir(predictions_pkl_dir) if f.startswith('predictions') and f.endswith(f'{selected_type}.pkl')]


    # get the labels files
    labels_files = [f for f in os.listdir(predictions_pkl_dir) if f.startswith('labels') and f.endswith(f'{selected_type}.pkl')]
    labels_files.sort()

    with open(os.path.join(results_base, cfg.results_dir, '0','data',f'{selected_type}_inds.pkl'), 'rb') as f:
        inds = pickle.load(f)


    stimuli, recordings, setup, noise_full = read_data(
        cfg['start_n_scan'],
        cfg['num_datapoints_per_scanfield'],
        cfg['cell_id'],
        cfg['rec_ids'],
        "noise",
        '/gpfs01/euler/User/ssuhai/GitRepos/jaxley_rgc/deistler_our_data_and_morph'
    )



    stim_branch_inds = stimuli["branch_ind"].to_numpy()
    stim_comps = stimuli["comp"].to_numpy()

    avg_recordings = build_avg_recordings(
        recordings, cfg['rec_ids'], cfg['nseg'], cfg['num_datapoints_per_scanfield']
            )
    
    
    number_of_recordings_each_scanfield = list(avg_recordings.groupby("rec_id").size())
    global rec_ids_of_roi 
    rec_ids_of_roi = avg_recordings["rec_id"].to_numpy()
    unique_rec_ids = np.unique(rec_ids_of_roi)
    comp_inputs_all = np.zeros((len(avg_recordings),cfg['num_datapoints_per_scanfield'] * len(unique_rec_ids)))
    

    for idx,(i_df, rec) in enumerate(avg_recordings.iterrows()):
        
        # which recording is this compartment from?
        rec_idx = np.where(unique_rec_ids == rec_ids_of_roi[idx])[0]
        input_to_comp = stimuli.loc[(stimuli["branch_ind"] == rec["branch_ind"])] 
        idx_of_min = (input_to_comp["comp"] - rec["comp"]).abs().idxmin()
        input_to_comp = np.array(input_to_comp.loc[idx_of_min]['activity'])
        
        comp_inputs_all[idx, :] = input_to_comp

    with open(os.path.join(results_base, cfg.results_dir, '0',f'predictions_labels/currents_split_{selected_type}.pkl'), 'rb') as f:
        currents = pickle.load(f)


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

    return all_preds, all_labels, comp_inputs_all, rec_ids_of_roi,inds, prediction_files,avg_recordings
    




def plot_prediction_labels_currents_over_epochs(predictions_pkl_dir,save_dir,cfg,max_datapoints=10):

    save_base = os.path.join(save_dir, 'prediction_labels_currents_over_epochs')
    os.makedirs(save_base, exist_ok=True)

    all_preds, all_labels, comp_inputs_all, rec_ids_of_roi,inds, prediction_files,avg_recordings = get_stimuli_labels_predictions(predictions_pkl_dir,cfg)

    for datapoint in range(np.minimum(all_preds[0].shape[0],max_datapoints)):    
        fig, ax = plt.subplots(int(np.ceil(np.sqrt(len(prediction_files)))), int(np.ceil(np.sqrt(len(prediction_files)))), figsize=(50, 50))
        ax = ax.flatten() if type(ax) is np.ndarray else [ax]

        # get the idx of the time series. remember the data is stacked across rec_ids so we need to 
        # take floor division to get the
        inferred_rec_id: int = inds[datapoint]  // cfg['num_datapoints_per_scanfield']
        comp_was_measured = rec_ids_of_roi == inferred_rec_id
        current_comp_input = comp_inputs_all[comp_was_measured,inds[datapoint]]
        nr_of_comps = np.sum(comp_was_measured)
        assert nr_of_comps > 0, "No compartments were measured in this scanfield"
        for epoch in range(len(prediction_files)):
            ax[epoch].plot([i for i in range(nr_of_comps)],all_preds[epoch][datapoint, comp_was_measured], color='blue', label=f'Prediction at {datapoint}', linestyle='--')
            ax[epoch].plot([i for i in range(nr_of_comps)],all_labels[epoch][datapoint, comp_was_measured], color='red', label=f'Labels at {datapoint}',linewidth=2)
      
            ax[epoch].plot([i for i in range(nr_of_comps)],current_comp_input, color='green', label=f'Input to compartments at {datapoint}',linewidth=4)

        
            ax[epoch].set_title(f'Prediction and labels rec field {inferred_rec_id}')
            ax[epoch].set_xlabel('Compartment')
            ax[epoch].set_ylabel('Activity')
            ax[epoch].set_title(f'Epoch {epoch}')
            ax[epoch].legend()

            #

        saving_at = os.path.join(save_base,f'{cfg.selected_type}_pred_and_labels_datapoint_{datapoint}.pdf')
        plt.tight_layout()
        
        fig.savefig(saving_at)
        print(f"Figure of datapont {datapoint} saved to {saving_at}")
        plt.close(fig)  


    # Plot labels and predictions accross time for single compartment across epochs
    
    assert all_preds[0].shape[0] == len(inds), "Number of datapoints per scanfield not equal to number of indices of split"

    for comp_idx in range(all_preds[0].shape[1]):
        fig, ax = plt.subplots(int(np.ceil(np.sqrt(len(prediction_files)))), int(np.ceil(np.sqrt(len(prediction_files)))), figsize=(50, 50))
        ax = ax.flatten() if type(ax) is np.ndarray else [ax]

        inferred_rec_id: int = rec_ids_of_roi[comp_idx]
        rec_id_index = np.where(np.unique(rec_ids_of_roi) == inferred_rec_id)[0]
        indx_of_split_where_comp_was_measured = np.where((rec_id_index * cfg['num_datapoints_per_scanfield'] <= inds) & (inds < (rec_id_index + 1) * cfg['num_datapoints_per_scanfield']))[0]
        assert len(indx_of_split_where_comp_was_measured) > 0, f"No compartments were measured in this scanfield for rec_id {inferred_rec_id}"
        

        for epoch in range(len(prediction_files)):

            predicted = all_preds[epoch][indx_of_split_where_comp_was_measured,comp_idx]
            labels = all_labels[epoch][indx_of_split_where_comp_was_measured,comp_idx]
            input_current = comp_inputs_all[comp_idx,inds[indx_of_split_where_comp_was_measured]]
            nr_datapoints = len(indx_of_split_where_comp_was_measured)
            ax[epoch].plot([i for i in range(nr_datapoints)],predicted, color='blue', label=f'Prediction at epoch {epoch}', linestyle='--')
            ax[epoch].plot([i for i in range(nr_datapoints)],labels, color='red', label=f'Labels at {epoch}',linewidth=2)
            ax[epoch].plot([i for i in range(nr_datapoints)],input_current, color='green', label=f'Input to compartment at {epoch}',linewidth=4)
            
            label_pred_corr = np.corrcoef(predicted,labels)[0,1]
            current_label_corr = np.corrcoef(input_current,labels)[0,1]
            pred_current_corr = np.corrcoef(predicted,input_current)[0,1]
            
            # Create the text box with correlations
            textstr = '\n'.join((
                r'$\rho(label, pred)=%.2f$' % (label_pred_corr, ),
                r'$\rho(current, label)=%.2f$' % (current_label_corr, ),
                r'$\rho(pred, current)=%.2f$' % (pred_current_corr, )))
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax[epoch].text(0.05, 0.95, textstr, transform=ax[epoch].transAxes, fontsize=14,
                verticalalignment='top', bbox=props) 

            ax[epoch].set_title('Prediction, labels, input current')
            ax[epoch].set_xlabel('Time step idx')
            ax[epoch].set_ylabel('Activity')
            ax[epoch].set_title(f'Epoch {epoch}')
            ax[epoch].legend()

        saving_at = os.path.join(save_base,f'{cfg.selected_type}_pred_and_labels_over_time_rec_id_{inferred_rec_id}_compartment_{comp_idx}.pdf')
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

    
    if 'all' in  cfg['plot_type'] or 'loss_over_epochs' in cfg['plot_type']:    
    
        loss_path_pkl = os.path.join(results_base,cfg.results_dir, '0', 'loss.pkl')
        plot_loss_over_epochs(loss_path_pkl,save_dir)

    if 'prediction_labels' in cfg['plot_type'] or 'all' in cfg['plot_type']:
        predictions_path_pkl = os.path.join(results_base,cfg.results_dir, '0', 'predictions_labels')
        plot_prediction_labels_currents_over_epochs(predictions_path_pkl,save_dir,cfg)

    if 'rhos_meas_over_epochs' in cfg['plot_type'] or 'all' in cfg['plot_type']:
        rhos_path = os.path.join(results_base,cfg.results_dir, '0', 'rhos')
    
        plot_rhos_meas_accorss_epochs(rhos_path,save_dir,cfg)


if __name__== '__main__':
    run()

