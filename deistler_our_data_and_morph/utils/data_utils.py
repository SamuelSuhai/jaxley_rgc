from itertools import chain
import numpy as np
import h5py
import pandas as pd


def read_data(
    start_n_scan,
    num_datapoints_per_scanfield,
    cell_id,
    rec_ids,
    noise_name,
    path_prefix=".",
):
    
    '''
    Takes in:
    - start_n_scan: int - the starting stimulus number
    - num_datapoints_per_scanfield: int - the number of datapoints per scanfield (i.e. length of time series)
    

    Outputs: 
    Stimuli: pd.DataFrame - contains the stimulation of BCs over time for each compartment! 
            Compartments in rows. It gets some columns from the bc_output DF. x_/y_loc are the coordinates of the BCs.
    Recordings: pd.DataFrame - combines the labels with meta info on rois.    

    '''


    # Only loaded for visualization.
    file = h5py.File(f"{path_prefix}/data/{noise_name}.h5", 'r+')
    # changed this field from k
    noise_stimulus = file["NoiseArray3D"][()]
    noise_stimulus = noise_stimulus[:, :, start_n_scan:start_n_scan+num_datapoints_per_scanfield]
    noise_full = np.concatenate([noise_stimulus for _ in range(len(rec_ids))], axis=2)
    
    setup = pd.read_pickle(f"{path_prefix}/results/data/setup.pkl")
    recording_meta = pd.read_pickle(f"{path_prefix}/results/data/recording_meta.pkl")
    stimuli_meta = pd.read_pickle(f"{path_prefix}/results/data/stimuli_meta_{cell_id}.pkl")
    labels_df = pd.read_pickle(f"{path_prefix}/results/data/labels_lowpass_{cell_id}.pkl")
    
    # TODO Change to file that contains all outputs. -> isnt this done??
    bc_output = pd.read_pickle(f"{path_prefix}/results/data/off_bc_output_{cell_id}.pkl")
    
    setup = setup[setup["cell_id"] == cell_id]
    # Setup contains one row per roi for each recording field
    setup = setup[setup["rec_id"].isin(rec_ids)]
    
    # this has one row per compartment
    stimuli_meta = stimuli_meta[stimuli_meta["cell_id"] == cell_id]
    
    bc_output = bc_output[bc_output["cell_id"] == cell_id]
    
    # select only the bc output for the current rec_id
    bc_output = bc_output[bc_output["rec_id"].isin(rec_ids)] 

    # has one row per roi in each rec id 
    recording_meta = recording_meta[recording_meta["cell_id"] == cell_id]
    recording_meta = recording_meta[recording_meta["rec_id"].isin(rec_ids)]
    
    labels_df = labels_df[labels_df["cell_id"] == cell_id]
    labels_df = labels_df[labels_df["rec_id"].isin(rec_ids)]
    
    # Contrain the number of labels.
    constrained_ca_activities = np.stack(labels_df["ca"].to_numpy())[:, start_n_scan:start_n_scan+num_datapoints_per_scanfield].tolist()
    labels_df["ca"] = constrained_ca_activities

    ### If we use multiple labels per image, we have to interpolate the "images" (i.e. the BC acitvities) here
    constrained_activities = np.stack(bc_output["activity"].to_numpy())[:, start_n_scan:start_n_scan+num_datapoints_per_scanfield].tolist()
    bc_output["activity"] = constrained_activities

    ### Concatenate the BC activity along the recording ids
    # Contrain the number of stimulus images.
    bc_output_concatenated = bc_output.groupby("bc_id", sort=False)["activity"].apply(lambda x: list(chain(*list(x))))
    
    # Constrain to a single rec_id because, apart from the activity (which is dealt with above) the bc_outputs have the same info for every scanfield.
    bc_output = bc_output[bc_output["rec_id"] == rec_ids[0]]
    bc_output["activity"] = list(bc_output_concatenated.to_numpy())

    # Join stimulus dfs.
    # Stimuli has compartments in rows and for each the bc input over time
    stimuli = stimuli_meta.join(bc_output.set_index("bc_id"), on="bc_id", how="left", rsuffix="_bc")
    stimuli = stimuli.drop(columns="cell_id_bc")
    
    
    assert labels_df.shape[0] == recording_meta.shape[0]

    # Join recording dfs.
    # create a unique identifier for reach roi
    labels_df["unique_id"] = labels_df["rec_id"] * 100 + labels_df["roi_id"]
    # same for recording meta: contains info on branch and compoartment closest 
    # to recording site. COMP HAS FLOAT VALUES. IS THIS CORRECT?1111
    recording_meta["unique_id"] = recording_meta["rec_id"] * 100 + recording_meta["roi_id"]
    recordings = recording_meta.join(labels_df.set_index("unique_id"), on="unique_id", how="left", rsuffix="_ca")
    recordings = recordings.drop(columns=["cell_id_ca", "rec_id_ca"])


    return stimuli, recordings, setup, noise_full


def _average_calcium_in_identical_comps(rec_df, num_datapoints_per_scanfield):
    '''
    
    Inputs:
    - rec_df: pd.DataFrame - For one scanfield contains rois in rows and the corresponding calcium trace
                in the column oir_id_ca. 
            
    - num_datapoints_per_scanfield: int - the number of datapoints per scanfield (i.e. length of time series

    Outputs:

    mean_df: pd.DataFrame - contains the mean calcium trace across all ROIs in the same compartment.
            has nr of rows equal to the number of compartments.
    
    '''

    num_datapoints = num_datapoints_per_scanfield

    # expand the calcium trace into num_datapoints columns: one column for each time point (stimulus presentation)
    rec_df[[f"ca{i}" for i in range(num_datapoints)]] = pd.DataFrame(rec_df.ca.tolist(), index=rec_df.index)
    rec_df = rec_df.drop(columns="ca")
    
    # Take the mean calcium trace across all ROIs in the same compartment!!!
    mean_df = rec_df.groupby(["branch_ind", "comp_discrete"]).mean()

    # Merge columns into a list of a single column.
    mean_df["ca"]= mean_df[[f"ca{i}" for i in range(num_datapoints)]].values.tolist()
    for i in range(num_datapoints):
        mean_df = mean_df.drop(columns=f"ca{i}")
    return mean_df


def build_avg_recordings(recordings, rec_ids, nseg, num_datapoints_per_scanfield):
    '''
    taks in:
    recordings: pd.DataFrame - contains the labels with meta info on rois.
                It has one row per roi in each rec id and has the calcium trace of this roi.


    Outputs:
    avg_recordings: pd.DataFrame - contains the average calcium trace across all ROIs in the same compartment.
            This is concatenated for each scan field
            has nr of rows equal to the number of compartments in all scan fields.


    
    '''

    avg_recordings = []
    for rec_id in rec_ids:
        rec_in_scanfield = recordings[recordings["rec_id"] == rec_id].copy()

        rec_in_scanfield["comp_discrete"] = np.clip(np.floor(rec_in_scanfield["comp"] * nseg).tolist(), a_min=0, a_max=nseg-1)
        rec_in_scanfield = rec_in_scanfield.drop(columns="cell_id")
        
        # the average ca in each compartment (if mulitple rois in one comp they are averaged)
        avg_recordings_in_scanfield = _average_calcium_in_identical_comps(
            rec_in_scanfield, num_datapoints_per_scanfield
        )
        
        # Make `branch_ind` and `discrete_comp` columns again. Before they are set as index?
        avg_recordings_in_scanfield = avg_recordings_in_scanfield.reset_index()
    
        avg_recordings.append(avg_recordings_in_scanfield)
    
    avg_recordings = pd.concat(avg_recordings)
    avg_recordings["rec_id"] = avg_recordings["rec_id"].astype(int)
    return avg_recordings


def build_training_data(
    i_amp,
    stimuli,
    avg_recordings,
    rec_ids,
    num_datapoints_per_scanfield,
    number_of_recordings_each_scanfield,
    scale_by_bc_number=False,
):
    '''
    inputs:
    stimuli: pd.DataFrame - contains the stimulation of BCs over time for each compartment!
            Compartments (of entire cell) in rows.

    outputs:
    currents: np.array - the currents that will be used as step currents.
                        Is of shape (num_datapoints_per_scanfield, num_compartments in entire cell)
    
    labels: np.array - the averaged calcium traces of the compartments.
                        In a special form: a matrix with shape 
                        (num_datapoints_per_scanfield * nr scan fields (= nr of images per scanfield aka time points), num_compartments in which we record the intracellular calcium )                        

    loss_weights: np.array - the loss weights for each compartment. Again with special form as labels

    '''


    number_of_recordings = len(avg_recordings)
    
    # The currents that will be used as step currents.
    bc_activity = np.stack(stimuli["activity"].to_numpy()).T
    
    # scale stimulation by number of synapses to have every BC have the same influence over RGC in expectation 
    # (individual BC to RGC sysnapse streghts are drawn randomly)
    # TODO: check what happoens if not scaled by nur of synapses
    currents = i_amp * np.asarray(bc_activity) 
    
    if scale_by_bc_number:
        currents /= stimuli["num_synapses_of_bc"].to_numpy()

    # Labels will also have to go to a dataloader.
    loss_weights = np.zeros((number_of_recordings, len(rec_ids) * num_datapoints_per_scanfield))
    labels = np.zeros((number_of_recordings, len(rec_ids) * num_datapoints_per_scanfield))
    
    cumsum_rec = np.cumsum([0] + number_of_recordings_each_scanfield)
    for i in range(len(rec_ids)):
        rec_id = rec_ids[i]

        # indices for the current scan field.
        start = cumsum_rec[i]
        end = cumsum_rec[i+1]
    
        # Masks for loss.
        loss_weights[start:end, i*num_datapoints_per_scanfield: (i+1)*num_datapoints_per_scanfield] = 1.0
    
        # Labels.
        recordings_in_this_scanfield = avg_recordings[avg_recordings["rec_id"] == rec_id]
        labels_in_this_scanfield = np.asarray(np.stack(recordings_in_this_scanfield["ca"].to_numpy()).T)
        

        labels[start:end, i*num_datapoints_per_scanfield: (i+1)*num_datapoints_per_scanfield] = labels_in_this_scanfield.T
        i += 1
    
    loss_weights = np.asarray(loss_weights)
    loss_weights = loss_weights.T  # shape (num_images_per_scanfield, 4).
    
    labels = labels.T  # shape (num_images_per_scanfield, 4).
    labels = np.asarray(labels)

    return currents, labels, loss_weights