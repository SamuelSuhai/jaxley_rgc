import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import numpy as np
import pickle
from matplotlib.cm import ScalarMappable



save_dir = '/gpfs01/euler/User/ssuhai/GitRepos/jaxley_rgc/deistler_our_data_and_morph/debugging/figs'


def plot_roi_labels_and_cell(labels, cell, x_coords, y_coords):
    """
    Plots:
      - Time series for each ROI in the left subplot
      - Cell morphology in the right subplot,
        with ROI indices labeled at (x, y) positions.
    Saves the figure in the directory: 
      current_working_directory / debugging / figs
    
    Parameters
    ----------
     : np.ndarray
        Time-series data of shape (num_time_points, num_rois).
    cell : object
        An object with a `vis(ax=..., type="morph", dims=[...])` method
        for plotting morphology.
    x_coords : array-like
        x-coordinates of each ROI on the morphological plot.
    y_coords : array-like
        y-coordinates of each ROI on the morphological plot.
    """

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    # --- Subplot 1: Time series for each ROI ---
    num_time_points, num_rois = labels.shape
    time_steps = np.array([i for i in range(num_time_points)]) * 0.2     
    for comp_idx in range(num_rois):
        ax[0].plot(time_steps,labels[:, comp_idx], label=f'Compartment {comp_idx+1}')
    
    ax[0].set_title('Label for each Compartment')
    ax[0].set_xlabel('Time (s)')
    ax[0].set_ylabel('Ca label')
    ax[0].legend(loc='best')

    # --- Subplot 2: Cell morphology ---
    cell.vis(ax=ax[1])
    ax[1].set_title('Cell morphology')

    # Plot ROI indices at (x, y) positions
    for i in range(num_rois):
        ax[1].text(
            x_coords[i],
            y_coords[i],
            str(i+1),
            color='red',
            fontsize=12,
            ha='center',  # center horizontally
            va='center'   # center vertically
        )

    plt.tight_layout()

    save_path = os.path.join(save_dir, 'labels_comps_on_cell.pdf')
    fig.savefig(save_path)
    print(f"Figure saved to {save_path}")

    plt.close(fig)  # Close the figure to free memory



def plot_recording_compartments_and_rois_on_cell(cell,avg_recordings,recordings_raw,over_write_save_path=False):

    
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    
    # Plot the cell morphology
    cell.vis(ax=ax)

    # define colorwheel with 10 colors
    colors = plt.cm.tab10.colors
    
    # plor recorded compartments
    for i, rec in avg_recordings.iterrows():
        cell.branch(rec["branch_ind"]).loc(rec["comp"]).vis(ax=ax, color=colors[i%10])

    # Store labels and handles for the legend
    legend_labels = []
    legend_handles = []

    # Plot the experimental ROIs
    for i, rec in recordings_raw.iterrows():
        rec_id = int(rec["rec_id"])
        
        # Check if this rec_id has already been added to the legend
        if f"Rec field {rec_id}" not in legend_labels:
            scatter = ax.scatter(rec["roi_x"], rec["roi_y"], s=5.0, alpha=0.7,
                            color=colors[rec_id % 10], zorder=10000,
                            label=f"Rec field {rec_id}")
            legend_labels.append(f"Rec field {rec_id}")
            legend_handles.append(scatter)
        else:
            # If it's already in the legend, plot without a label to prevent duplicates
            ax.scatter(rec["roi_x"], rec["roi_y"], s=5.0, 
                            color=colors[rec_id % 10], alpha=0.7,zorder=10000) # no label


    
    save_path = over_write_save_path if over_write_save_path else os.path.join(save_dir, 'cell_recording_sites.pdf')
    
    ax.set_title('Recording compartments and experimental ROIs')

    # Use the stored handles and labels
    ax.legend(handles=legend_handles, labels=legend_labels, loc='best')

    fig.savefig(save_path)
    print(f"Figure saved to {save_path}")

    plt.close(fig) 


def upsample_image(image, pixel_size=30):
    """Returns a new image where every value is one micro meter in size."""
    image = np.repeat(image, (pixel_size,), axis=0)
    image = np.repeat(image, (pixel_size,), axis=1)
    return image

def get_image_locs(df,pixel_size=30):
    im_pos_x = np.linspace(-7.5*pixel_size + 0.5, 7.5*pixel_size - 0.5, 15*pixel_size) + df["image_center_x"].item()
    im_pos_y = -np.linspace(-10.0*pixel_size + 0.5, 10.0*pixel_size - 0.5, 20*pixel_size) + df["image_center_y"].item()
    return im_pos_x, im_pos_y


def plot_bc_output_on_cell_each_image(cell,
                                      stimuli,
                                      noise_full,
                                      setup,
                                      avg_recordings):
    '''
    Takes the bc_activity from stimuli  frame and plots it on the cell morphology.
    '''

    # get stimulus info
    num_time_points = len(stimuli.loc[0,'activity'])
    times = np.array([i for i in range(num_time_points)]) * 0.2

    # Normalize bc_activity to the range [0, 1] for colormap scaling
    bc_activity = np.stack(stimuli["activity"].to_numpy())
    norm = mpl.colors.Normalize(vmin=np.min(bc_activity), vmax=np.max(bc_activity))
    cmap = mpl.colormaps['viridis']

    # loop over stimuli (i.e. images)
    for idx_time, t in enumerate(times):
        
        fig, ax = plt.subplots(1, 1, figsize=(20, 20))

        cell.vis(ax=ax, linewidth=2)  # Increase the line width
        ax.set_title('Cell')

        colors = plt.cm.tab10.colors # for ROI

        # show recording branches and compartments in red
        for i, rec in avg_recordings.iterrows():
            cell.branch(rec["branch_ind"]).loc(rec["comp"]).vis(ax=ax, color=colors[i % 10], linewidth=2)  # Increase the line width


        for i,bc in stimuli.iterrows():
            ax.scatter(bc['x_loc'], bc['y_loc'], s=60.0, color=cmap(norm(bc['activity'][idx_time])), zorder=10000)

        # upsample and plot image presented at time t
        upsampled_image = upsample_image(noise_full[idx_time].T)
        
        # show image in correct position
        roi_idx = 0 # does not matter because within a rec field the images are at the same place
        im_pos_x, im_pos_y = get_image_locs(setup.iloc[roi_idx], pixel_size=30)
        _ = ax.imshow(upsampled_image, extent=[im_pos_x[0], im_pos_x[-1], im_pos_y[-1], im_pos_y[0]], clim=[0, 1], alpha=0.4, cmap="viridis")

        # Scatter plot with normalized bc_activity
        scatter = ax.scatter([], [], s=20.0, c=[], cmap=cmap, norm=norm, zorder=10000)  # Dummy scatter for colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('BC Activity')

        plt.tight_layout()

        save_path = os.path.join(save_dir, f'BC_output_on_cell_morph_with_noise_frame_{idx_time}.pdf')
        fig.savefig(save_path)
        print(f"Figure saved to {save_path}")

        plt.close(fig)  # Close the figure to free memory


def plot_currents_on_cell_each_image(cell,
                                    currents,
                                    stimuli,
                                    noise_full,
                                    setup,
                                    avg_recordings):

    # get stimulus info
    num_time_points, num_comps = currents.shape
    times = np.array([i for i in range(num_time_points)]) * 0.2

    # Plot the locations at which the current is set
    # this is the averaged position of the rois combined in one compartment
    
    # loop over stimuli (i.e. images)
    for idx_time, t in enumerate(times):
        
        fig, ax = plt.subplots(1, 1, figsize=(20, 20))

        cell.vis(ax=ax)
        ax.set_title('Cell')

        colors = plt.cm.tab10.colors # for ROI

        # Normalize currents to the range [0, 1] for colormap scaling
        norm = mpl.colors.Normalize(vmin=np.min(currents), vmax=np.max(currents))
        cmap = mpl.colormaps['viridis']
 


        # show recording branches and compartments in red
        for i, rec in avg_recordings.iterrows():
            cell.branch(rec["branch_ind"]).loc(rec["comp"]).vis(ax=ax, color=colors[i % 10])

        stim_branch_inds = stimuli["branch_ind"].to_numpy()
        stim_comps = stimuli["comp"].to_numpy()

        for i, (branch, comp) in enumerate(zip(stim_branch_inds, stim_comps)):
            cell.branch(branch).loc(comp).vis(ax=ax, color=cmap(norm(currents[idx_time, i])),linewidth=10)

        # upsample and plot image presented at time t
        upsampled_image = upsample_image(noise_full[idx_time].T)
        
        # show image in correct position
        roi_idx = 0 
        im_pos_x, im_pos_y = get_image_locs(setup.iloc[roi_idx], pixel_size=30)
        _ = ax.imshow(upsampled_image, extent=[im_pos_x[0], im_pos_x[-1], im_pos_y[-1], im_pos_y[0]], clim=[0, 1], alpha=0.4, cmap="viridis")

        # Scatter plot with normalized currents
        scatter = ax.scatter([], [], s=20.0, c=[], cmap=cmap, norm=norm, zorder=10000)  # Dummy scatter for colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Current Strength')

        plt.tight_layout()

        save_path = os.path.join(save_dir, f'BC_currents_on_cell_morph_with_noise_frame_{idx_time}.pdf')
        fig.savefig(save_path)
        print(f"Figure saved to {save_path}")

        plt.close(fig)  # Close the figure to free memory


def plot_calcium_on_cell_each_image(cell,
                                    labels,
                                    stimuli,
                                    noise_full,
                                    setup,
                                    avg_recordings,
                                    max_data_points=10):
    """
    Plots the labels on the cell morphology for each time point in a separate figure.
    """
    num_time_points, num_rois = labels.shape
    times = np.array([i for i in range(num_time_points)]) * 0.2

    # Normalize labels to the range [0, 1] for colormap scaling
    norm = mpl.colors.Normalize(vmin=np.min(labels), vmax=np.max(labels))
    cmap = mpl.colormaps['viridis']
    bc_activity = np.stack(stimuli["activity"].to_numpy())
    norm_bc = mpl.colors.Normalize(vmin=np.min(bc_activity), vmax=np.max(bc_activity))

    # loop over stimuli (i.e. images)
    for idx_time, t in enumerate(times):

        if idx_time >= max_data_points:
            break

        
        fig, ax = plt.subplots(1, 1, figsize=(20, 20))

        cell.vis(ax=ax)
        ax.set_title('Label at '+f' at time {t:.1f}s')

        colors = plt.cm.tab10.colors # for ROI

        # show recording branches and compartments in red
        for i, rec in avg_recordings.iterrows():
            cell.branch(rec["branch_ind"]).loc(rec["comp"]).vis(ax=ax, color=cmap(norm(labels[idx_time,i])), linewidth=2)

        # upsample and plot image presented at time t
        upsampled_image = upsample_image(noise_full[idx_time].T)
        
        # show image in correct position
        roi_idx = 0 
        im_pos_x, im_pos_y = get_image_locs(setup.iloc[roi_idx], pixel_size=30)
        _ = ax.imshow(upsampled_image, extent=[im_pos_x[0], im_pos_x[-1], im_pos_y[-1], im_pos_y[0]], clim=[0, 1], alpha=0.4, cmap="viridis")

        # Plot the BC acpivity and currents
        for i, bc in stimuli.iterrows():
            ax.scatter(bc['x_loc'], bc['y_loc'], s=40.0, color=cmap(norm_bc(bc['activity'][idx_time])), zorder=10000)
        
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # or np.array(labels[idx_time,:]) if you want it scaled
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label("Calcium Activity")
        print(f"Saving figure at path: {os.path.join(save_dir, f'Labels_and_BC_activity_on_cell_morph_with_noise_frame_{idx_time}.pdf')}")
        fig.savefig(os.path.join(save_dir, f'Labels_and_BC_activity_on_cell_morph_with_noise_frame_{idx_time}.pdf'))
        plt.close(fig)  # Close the figure to free memory


        
def plot_currents_bc_activity_calcium_all_times(labels,
                                                cell,
                                                currents,
                                                stimuli,
                                                noise_full,
                                                setup,
                                                avg_recordings):
    
    '''
    Creates a plot with two rows: in the top you have one axis per time step which shows the cell morphology, bc_activity, and currents on branches.
    In the bottom row you have the calcium activity for each compartment. Both are heatmaps.
    '''
    num_time_points, num_comps = currents.shape
    times = np.array([i for i in range(num_time_points)]) * 0.2

    # Normalize bc_activity and currents to the range [0, 1] for colormap scaling
    bc_activity = np.stack(stimuli["activity"].to_numpy())
    norm_bc = mpl.colors.Normalize(vmin=np.min(bc_activity), vmax=np.max(bc_activity))
    norm_currents = mpl.colors.Normalize(vmin=np.min(currents), vmax=np.max(currents))
    calc_norm = mpl.colors.Normalize(vmin=np.min(labels), vmax=np.max(labels))
    cmap = mpl.colormaps['viridis']

    fig, axs = plt.subplots(2, num_time_points, figsize=(100, 50))

    for idx_time, t in enumerate(times):
        ax_top = axs[0, idx_time]
        ax_bottom = axs[1, idx_time]

        # Top row: Cell morphology with bc_activity and currents
        cell.vis(ax=ax_top, linewidth=2)
        ax_top.set_title(f'Time {t:.1f}s')

        # Plot the BC acpivity and currents
        for i, bc in stimuli.iterrows():
            ax_top.scatter(bc['x_loc'], bc['y_loc'], s=40.0, color=cmap(norm_bc(bc['activity'][idx_time])), zorder=10000)

        stim_branch_inds = stimuli["branch_ind"].to_numpy()
        stim_comps = stimuli["comp"].to_numpy()

        for i, (branch, comp) in enumerate(zip(stim_branch_inds, stim_comps)):
            cell.branch(branch).loc(comp).vis(ax=ax_top, color=cmap(norm_currents(currents[idx_time, i])), linewidth=2)




        # Bottom row: Calcium activity 
        cell.vis(ax=ax_bottom, linewidth=2)

        # Show recording branches and compartments calcium activity in axes below
        for i, rec in avg_recordings.iterrows():
            cell.branch(rec["branch_ind"]).loc(rec["comp"]).vis(ax=ax_bottom, color=cmap(calc_norm(labels[idx_time,i])), linewidth=2)

        ax_bottom.set_title('Calcium Activity')

    plt.tight_layout()

    save_path = os.path.join(save_dir, 'currents_bc_activity_calcium_all_times.pdf')
    fig.savefig(save_path)
    print(f"Figure saved to {save_path}")

    plt.close(fig)  # Close the figure to free memory
    


def change_labels_to_inputs(labels,
                             stimuli,
                             avg_recordings):
    

    labels_out = np.zeros_like(labels)

    for idx,recorded_compartments in avg_recordings.iterrows():
        
        # get the closest branch
        closest_stimulus = stimuli.loc[stimuli['branch_ind'] == recorded_compartments['branch_ind']]

        # get the closest compartment
        closest_stimulus = closest_stimulus.loc[np.argmin(np.abs(closest_stimulus['comp'] - recorded_compartments['comp']))]

        # make sure we selected only one compartment on one branch
        assert closest_stimulus.shape[0] == 1

        # retrieve the stimuluation with which we want to replace the calcium activity
        stimulus = closest_stimulus['activity'].to_numpy()

        # replace the calcium activity with the stimulus
        labels_out[:,idx] = stimulus

    return labels_out




if __name__ == '__main__':
    pass
    # plot_bc_outputs_activity_stimulus()
