from jax import config

config.update("jax_enable_x64", False)
config.update("jax_platform_name", "gpu")

import os

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".2" # '0.6'





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

import jaxley as jx
from jaxley.optimize import TypeOptimizer
from jaxley.optimize.utils import l2_norm
from utils.data_utils import ( # changed from nex.fig4_rgc.utils.data_utils 
    read_data,
    build_avg_recordings,
    build_training_data,
)
from utils.transforms import (
    transform_params,
    transform_basal,
    transform_somatic,
)
from utils.dataloader import build_dataloaders
from utils.utils import (
    build_cell,
    build_kernel,
)
from utils.rf_utils import (
    compute_all_trained_rfs,
)
from simulate import loss_fn, predict
from jaxley.optimize import TypeOptimizer
from jax import tree_map
import pandas as pd
from warnings import simplefilter
from hydra.utils import get_original_cwd
from skimage.transform import resize

# for debugging
import debugging.debug_utils as dbg



log = logging.getLogger("rgc")

coinfig_fullpath = "/gpfs01/euler/User/ssuhai/GitRepos/jaxley_rgc/deistler_our_data_and_morph/config"

def plot_prediction_histogram(vmapped_predict,
                              opt_params,
                              opt_basal_params,
                              opt_somatic_params,
                              currents,
                              init_states,
                              epoch):
    # Evaluate after every epoch.
    predictions = vmapped_predict(
        transform_params.forward(opt_params),
        transform_basal.forward(opt_basal_params),
        transform_somatic.forward(opt_somatic_params),
        currents[:64],
        init_states,
    )
    fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    ax.hist(predictions.flatten(), bins=40, density=True)
    plt.savefig(
        f"{os.getcwd()}/figs/hist_{epoch}.png", dpi=150, bbox_inches="tight"
    )
    plt.show()



def create_rf_plots(counters,
                    cell,
                    all_loss_weights,
                    all_ca_predictions,
                    all_images,
                    avg_recordings,
                    setup,
                    epoch,
                    save_path_base = None,
                    return_ax_and_not_save = False):
    '''
    counters- list of indices of the recordings for which we want to plot the receptive fields
    save_path is a list or None
    '''
    # Receptive field setup.
    avg_recordings["roi_id"] = avg_recordings["roi_id"].astype(int)
    rec_and_roi = avg_recordings[["rec_id", "roi_id"]].to_numpy()
    rec_ids_of_all_rois = rec_and_roi[:, 0]
    roi_ids_of_all_rois = rec_and_roi[:, 1]
    noise_mag = 1e-1
    num_iter = 20
    avg_recordings = avg_recordings.reset_index()
    center = np.asarray([170, 150])
    pixel_size = 30
    levels = [0.5]

    # Compute receptive fields.
    rfs_trained = compute_all_trained_rfs(
        counters,
        all_loss_weights,
        all_ca_predictions,
        np.transpose(all_images, (1, 2, 0)),
        noise_mag,
        num_iter,
    )

    # contour RF plot
    for i, counter in enumerate(counters):


        fig, ax = plt.subplots(1, 1, figsize=(4.9, 6.5))

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

        # paths for saving figs
        if save_path_base is None:
            save_path_contour = f"{os.getcwd()}/figs/rf_{epoch}_rec_id_{rec_id}_roi_id_{roi_id}.png"
        else:
            save_path_contour = save_path_base + f"_roi_id_{roi_id}_contour.png"
        
        if not return_ax_and_not_save:
            print(f"Saving contour to {save_path_contour}")
            plt.savefig(
                save_path_contour, dpi=150, bbox_inches="tight"
            )
        else:
            ax_list = [ax]

    
    # visualize a heatmap of the receptive fields
    for i, counter in enumerate(counters):
        fig, ax = plt.subplots(1, 1, figsize=(4.9, 6.5))

        # changed
        ax = cell.vis(ax=ax) # ,color="k",morph_plot_kwargs={"zorder": 1000, "linewidth": 0.3})

        rec_id = rec_ids_of_all_rois[counter]
        roi_id = roi_ids_of_all_rois[counter]
        rf_pred = rfs_trained[i]
        setup_rec = setup[setup["rec_id"] == rec_id]
        offset_x = setup_rec["image_center_x"].to_numpy()[0]
        offset_y = setup_rec["image_center_y"].to_numpy()[0]

        im_pos_x = np.linspace(-7.5*pixel_size + 0.5, 7.5* pixel_size - 0.5, 15*pixel_size) + offset_x
        im_pos_y = -np.linspace(-10.0*pixel_size + 0.5, 10.0*pixel_size - 0.5, 20*pixel_size) + offset_y

        _ = ax.imshow(rf_pred, 
                            extent=[im_pos_x[0], im_pos_x[-1], im_pos_y[-1], im_pos_y[0]], 
                            clim=[0, 1], 
                            alpha=0.4, cmap="viridis")
        
        rec = avg_recordings.loc[counter]
        _ = ax.scatter(
            rec["roi_x"].item(),
            rec["roi_y"].item(),
            color=col,
            s=20.0,
            edgecolors="k",
            zorder=10000,
        )

        # paths for saving figs
        if save_path_base is None:
            save_path_heatmap = f"{os.getcwd()}/figs/rf_heatmap_{epoch}_rec_id_{rec_id}_roi_id_{roi_id}.png"
        else:
            save_path_heatmap = save_path_base + f"_roi_id_{roi_id}_heatmap.png"

        if not return_ax_and_not_save:
            print(f"Saving heatmap to {save_path_heatmap}")
            plt.savefig(
                save_path_heatmap, dpi=150, bbox_inches="tight")
        else:
            ax_list.append(ax)
            return(ax_list)



@hydra.main(config_path=coinfig_fullpath, config_name="train")
def run(cfg: DictConfig) -> None:
    simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
    os.mkdir("figs")
    os.mkdir("opt_params")
    os.mkdir("transforms")
    os.mkdir("data")
    os.mkdir("rhos")
    os.mkdir("predictions_labels")

    dt = 0.025
    t_max = 200.0
    num_truncations = 4
    nseg = cfg['nseg']#4
    time_vec = np.arange(0, t_max + 2 * dt, dt)

    warmup = 5.0
    i_amp = 0.1

    start_n_scan = cfg["start_n_scan"] # Start from time point 100 there were some weird things in the labels before
    num_datapoints_per_scanfield = cfg["num_datapoints_per_scanfield"]
    cell_id = cfg["cell_id"] #"2020-07-08_1" #"20161028_1"  # "20170610_1", "20161028_1"
    rec_ids = cfg["rec_ids"]
    log.info(f"Recording ids {rec_ids}")

    
    stimuli, recordings, setup, noise_full = read_data(
        start_n_scan,
        num_datapoints_per_scanfield,
        cell_id,
        rec_ids,
        "noise",
        get_original_cwd(),
    )
    noise_full = np.transpose(noise_full, (2, 0, 1))

    cell = build_cell(cell_id, nseg, cfg["soma_radius"], get_original_cwd())
    kernel = build_kernel(time_vec, dt)

    # Build average recordings.
    avg_rec_path = f"{get_original_cwd()}/results/intermediate/avg_recordings.pkl"
    avg_rec_dir = os.path.dirname(avg_rec_path)

    # Ensure the directory exists
    os.makedirs(avg_rec_dir, exist_ok=True)

    if cfg['reuse_avg_recordings']:
        log.info(f"Loading avg_recordings from intermediate file with path {avg_rec_path}")
        
        with open(avg_rec_path, "rb") as handle:
            avg_recordings = pickle.load(handle)
    else:
        log.info("Recomputing and saving avg_recordings - no intermediate file found")

        avg_recordings = build_avg_recordings(
            recordings, rec_ids, nseg, num_datapoints_per_scanfield
        )
        with open(avg_rec_path, "wb") as handle:
            pickle.dump(avg_recordings, handle)
    
    # Build recordings.
    number_of_recordings_each_scanfield = list(avg_recordings.groupby("rec_id").size())
    log.info(
        f"number_of_recordings_each_scanfield {number_of_recordings_each_scanfield}"
    )

    cell.delete_recordings()
    cell.delete_stimuli()

    # insert recordings in cell 
    for idx,(i, rec) in enumerate(avg_recordings.iterrows()):
        # here we set that the internal calcium is recorded???
        cell.branch(rec["branch_ind"]).loc(rec["comp"]).record("Cai", verbose=False)
    

    # debugging: visualize compartments in which we record
    if 'recording_comps_and_rois' in cfg['debugging_figs']:
        dbg.plot_recording_compartments_and_rois_on_cell(cell, 
                                                         avg_recordings,
                                                         recordings,
                                                         over_write_save_path= False) #os.path.join(get_original_cwd(), 'report_figs/2020-08-29_experimental_and_model_recordings.pdf'))


    log.info(f"Inserted {len(cell.recordings)} recordings")
    log.info(
        f"number_of_recordings_each_scanfield {number_of_recordings_each_scanfield}"
    )
    number_of_recordings = np.sum(number_of_recordings_each_scanfield)
    assert number_of_recordings == len(cell.recordings)
    assert len(number_of_recordings_each_scanfield) == len(rec_ids)

    # Build training data.
    currents, labels, loss_weights = build_training_data(
        i_amp,
        stimuli,
        avg_recordings,
        rec_ids,
        num_datapoints_per_scanfield,
        number_of_recordings_each_scanfield,
    )

    # # debug
    # log.info(f"Setting labels and current to constant value and remove all except first compartment for debugging")    

    # labels = np.ones_like(labels) * 0.5
    # currents = np.ones_like(currents) * 0.5




    # # debug visualization 
    # dbg.plot_roi_labels_and_cell(labels, cell, 
    #                          avg_recordings['roi_x'],
    #                          avg_recordings['roi_y'])


    # dbg.plot_bc_output_on_cell_each_image(cell,
    #                                   stimuli,
    #                                   noise_full,
    #                                   setup,
    #                                   avg_recordings)  
    
    # # dbg.plot_currents_bc_activity_calcium_all_times(labels,
    # #                                             cell,
    # #                                             currents,
    # #                                             stimuli,
    # #                                             noise_full,
    # #                                             setup,
    # #                                             avg_recordings)

    # dbg.plot_currents_on_cell_each_image(cell,
    #                                 currents,
    #                                 stimuli,
    #                                 noise_full,
    #                                 setup,
    #                                 avg_recordings)

    # dbg.plot_calcium_on_cell_each_image(cell,
    #                                 labels,
    #                                 stimuli,
    #                                 noise_full,
    #                                 setup,
    #                                 avg_recordings)
    
    # breakpoint()
    
    log.info(f"currents.shape {currents.shape}")
    log.info(f"labels.shape {labels.shape}")
    log.info(f"loss_weights.shape {loss_weights.shape}")
    

    stim_branch_inds = stimuli["branch_ind"].to_numpy()
    stim_comps = stimuli["comp"].to_numpy()

    # Somehow my cell.group_nodes is None. 
    # basal_inds = list(np.unique(cell.group_nodes["basal"]["branch_index"].to_numpy()))
    # somatic_inds = list(np.unique(cell.group_nodes["soma"]["branch_index"].to_numpy()))
    # changed to: hope it works lol

    basal_inds = list(np.unique(cell.basal.nodes["global_branch_index"].to_numpy()))
    somatic_inds = list(np.unique(cell.soma.nodes["global_branch_index"].to_numpy()))

    # Get initial state.
    _, init_states = jx.integrate(cell, t_max=warmup, return_states=True)

    # Define trainables.
    cell.delete_trainables()
    cell.basal.branch("all").make_trainable("axial_resistivity")
    cell.basal.branch("all").make_trainable("radius")
    parameters = cell.get_parameters()

    _ = np.random.seed(cfg["seed_weights"]+cfg["seed_ruler"])
    if cfg["weight_init"] == "random":
        parameters = [
            # changed the mean of the random weight to mid-value of the sigmoid. Scaled by 0.2 not 0.1
            {"w_bc_to_rgc": 0.2 * jnp.asarray(np.random.rand(currents.shape[1]))} # uniform
            # {"w_bc_to_rgc": 0.03 * jnp.asarray(np.random.randn(currents.shape[1]) + 0.1 )}
        ] + parameters
    else:
        assert isinstance(cfg["weight_init"], float)
        parameters = [
            {"w_bc_to_rgc": cfg["weight_init"] * jnp.ones(currents.shape[1])}
        ] + parameters
    log.info(f"Weight mean of w_bc_to_rgc: {parameters[0]['w_bc_to_rgc'].mean()}")
    
    # # debug: investigate parameters. Why are they initialized such that the transformed
    # # are moslty negative?


    output_scale = jnp.asarray(cfg["output_scale"])
    output_offset = jnp.asarray(cfg["output_offset"])

    with open(f"{os.getcwd()}/transforms/transform_params.pkl", "wb") as handle:
        pickle.dump(transform_params, handle)
    with open(f"{os.getcwd()}/transforms/transform_basal.pkl", "wb") as handle:
        pickle.dump(transform_basal, handle)
    with open(f"{os.getcwd()}/transforms/transform_somatic.pkl", "wb") as handle:
        pickle.dump(transform_somatic, handle)

    _ = np.random.seed(cfg["seed_membrane_conds"]+cfg["seed_ruler"])
    basal_neuron_params = [
        {"Na_gNa": jnp.asarray(0.05)},
        {"K_gK": jnp.asarray(0.05)},
        {"Leak_gLeak": jnp.asarray(1e-4)},
        {"KA_gKA": jnp.asarray(36e-3)},
        {"Ca_gCa": jnp.asarray(2.2e-3)},
        {"KCa_gKCa": jnp.asarray(0.05e-3)},
    ]
    # basal_neuron_params = [{key: sample_basal[key]} for key in transform_basal.uppers.keys()]

    somatic_neuron_params = [
        {"Na_gNa": jnp.asarray(0.05)},  # 0.2
        {"K_gK": jnp.asarray(0.05)},
        {"Leak_gLeak": jnp.asarray(1e-4)},
        {"KA_gKA": jnp.asarray(36e-3)},
        {"Ca_gCa": jnp.asarray(2.2e-3)},
        {"KCa_gKCa": jnp.asarray(0.05e-3)},
    ]
    # somatic_neuron_params = [{key: sample_somatic[key]} for key in transform_somatic.uppers.keys()]

    # Training.
    opt_params = transform_params.inverse(parameters)
    opt_basal_params = transform_basal.inverse(basal_neuron_params)
    opt_somatic_params = transform_somatic.inverse(somatic_neuron_params)

    # Dataloaders.
    tf.random.set_seed(cfg["seed_tf_dataloader"]+cfg["seed_ruler"])
    dataloader, eval_dataloaders = build_dataloaders(
        noise_full,
        currents,
        labels,
        loss_weights,
        cfg["val_frac"],
        cfg["test_num"],
        cfg["batchsize"],
    )

    log.info(f"noise_full {noise_full.shape}")

    num_batches = int(dataloader.cardinality())
    log.info(f"number of training batches {num_batches}")

    epoch_losses = []

    ###### Optimizers. ######
    # {lr: 0.01, momentum: 0.5, epochs: 10}, {lr: 0.001, momentum: 0.9, epochs: 10}
    lr_scaler = cfg["lr"]
    momentum = cfg["momentum"]
    bounary_dict = {}

    for i in range(1, int(cfg["iterations"] / cfg["reduce_lr_every"]) + 1):
        bounary_dict[i * cfg["reduce_lr_every"]] = 1 / cfg["reduce_lr_by"]
    log.info(f"lr scheduling dict: {bounary_dict}")

    scheduler = optax.piecewise_constant_schedule(
        init_value=lr_scaler,
        boundaries_and_scales=bounary_dict,
    )

    optimizer_params = TypeOptimizer(
        lambda lr: optax.chain(
            optax.clip(10.0),
            optax.sgd(scheduler, momentum=momentum),
        ),
        {
            "w_bc_to_rgc": lr_scaler,
            "axial_resistivity": lr_scaler,
            "radius": lr_scaler,
        },
        opt_params=opt_params,
    )
    opt_state_params = optimizer_params.init(opt_params)

    ###### BASAL PARAMETERS ######
    optimizer_basal_params = TypeOptimizer(
        lambda lr: optax.chain(
            optax.clip(10.0), optax.sgd(scheduler, momentum=momentum)
        ),
        {
            "Na_gNa": lr_scaler,
            "K_gK": lr_scaler,
            "Leak_gLeak": lr_scaler,
            "KA_gKA": lr_scaler,
            "Ca_gCa": lr_scaler,
            "KCa_gKCa": lr_scaler,
        },
        opt_basal_params,
    )
    opt_state_basal_params = optimizer_basal_params.init(opt_basal_params)

    ###### SOMATIC PARAMETERS ######
    optimizer_somatic_params = optax.sgd(scheduler, momentum=momentum)
    opt_state_somatic_params = optimizer_somatic_params.init(opt_somatic_params)


    static = {
        "cell": cell,
        "dt": dt,
        "t_max": t_max,
        "time_vec": time_vec,
        "num_truncations": num_truncations,
        "output_scale": output_scale,
        "output_offset": output_offset,
        "kernel": kernel,
        "transform_params": transform_params,
        "transform_somatic": transform_somatic,
        "transform_basal": transform_basal,
        "somatic_inds": somatic_inds,
        "basal_inds": basal_inds,
        "stim_branch_inds": stim_branch_inds,
        "stim_comps": stim_comps,
    }
    with open(f"{os.getcwd()}/cell.pkl", "wb") as handle:
        pickle.dump(cell, handle)

    # Build all functions.
    # sim_split = jit(partial(simulate_split, static=static))
    # vmapped_sim_split = jit(vmap(partial(simulate_split, static=static), in_axes=(None, None, None, None, 0, None)))
    # sim = jit(partial(simulate, static=static))
    # vmapped_sim = jit(vmap(partial(simulate, static=static), in_axes=(None, None, None, 0, None)))
    vmapped_predict = jit(
        vmap(partial(predict, static=static), in_axes=(None, None, None, 0, None))
    )
    vmapped_loss_fn = jit(
        vmap(partial(loss_fn, static=static), in_axes=(None, None, None, 0, 0, 0, None))
    )

    def batch_loss_fn(
        opt_params,
        opt_basal_neuron_params,
        opt_somatic_neuron_params,
        all_currents,
        all_labels,
        all_loss_weights,
        all_states,
    ):
        """Return average loss across a batch given transformed parameters."""
        params = transform_params.forward(opt_params)
        basal_neuron_params = transform_basal.forward(opt_basal_neuron_params)
        somatic_neuron_params = transform_somatic.forward(opt_somatic_neuron_params)
        losses = vmapped_loss_fn(
            params,
            basal_neuron_params,
            somatic_neuron_params,
            all_currents,
            all_labels,
            all_loss_weights,
            all_states,
        )
        return jnp.mean(losses)

    batch_grad_fn = jit(value_and_grad(batch_loss_fn, argnums=(0, 1, 2)))

    log.info(f"Starting to train")
    tf.random.set_seed(cfg["seed_tf_train_loop"]+cfg["seed_ruler"])
    best_validation_rho = -1.0
    best_test_rho = -1.0
    best_train_rho = -1.0

    num_epochs = int(np.ceil(cfg["iterations"] / num_batches))
    log.info(f"Number of epochs {num_epochs}")

    ######################################################## Training epochs ###############################################
    for epoch in range(num_epochs):
        with open(f"{os.getcwd()}/opt_params/params_{epoch}.pkl", "wb") as handle:
            pickle.dump([opt_params, opt_basal_params, opt_somatic_params], handle)

        with open(f"{os.getcwd()}/loss.pkl", "wb") as handle:
            pickle.dump(epoch_losses, handle)

        epoch_loss = 0.0

        # 1) Loop over training data 
        for batch_ind, batch in enumerate(dataloader):
            current_batch = batch[1].numpy()
            label_batch = batch[2].numpy()
            loss_weight_batch = batch[3].numpy()


            # 1.1) Compute gradient and update
            log.info(f"\tApplying batch grad function of epoch {epoch} and batch {batch_ind}")
            loss, gradient = batch_grad_fn(
                opt_params,
                opt_basal_params,
                opt_somatic_params,
                current_batch,
                label_batch,
                loss_weight_batch,
                init_states,
            )
            grad_params, grad_basal_params, grad_somatic_params = gradient
            
        
            
            #log.info(f"\tUpdating weights of batch {batch_ind}")
            # Update for weights.
            beta = cfg.beta
            for i in range(3):
                weight_norm = l2_norm(grad_params[i])
                key = list(grad_params[i].keys())[0]
                num_params = len(grad_params[i][key])
                grad_params[i] = tree_map(
                    lambda x: x / weight_norm**beta * num_params, grad_params[i]
                )

            # Update for basal parameters.
            num_params = 6 # !!!??????? change this??
            grad_norm_basal = l2_norm(grad_basal_params)
            grad_basal_params = tree_map(
                lambda x: x / grad_norm_basal**beta * num_params, grad_basal_params
            )

            # Update for basal parameters.
            num_params = 6
            grad_norm_somatic = l2_norm(grad_somatic_params)
            grad_somatic_params = tree_map(
                lambda x: x / grad_norm_somatic**beta * num_params, grad_somatic_params
            )

   

            epoch_loss += loss


            log.info(f"\tUpdating weights of batch {batch_ind}")
            # Update all parameters with optimizer
            updates, opt_state_params = optimizer_params.update(
                grad_params, opt_state_params
            )



            opt_params = optax.apply_updates(opt_params, updates)


            updates, opt_state_basal_params = optimizer_basal_params.update(
                grad_basal_params, opt_state_basal_params
            )
            opt_basal_params = optax.apply_updates(opt_basal_params, updates)

            updates, opt_state_somatic_params = optimizer_somatic_params.update(
                grad_somatic_params, opt_state_somatic_params
            )
            opt_somatic_params = optax.apply_updates(opt_somatic_params, updates)

            log.info(f"Batch {batch_ind}, avg loss per batch: {loss}")
            

            # 1.2) Evaluate after every nth batch
            if (batch_ind % cfg["eval_every_nth_batch"]) == (cfg["eval_every_nth_batch"] - 1):
                
                print("ENTERING EVALUATION FOR BATCH IND", batch_ind)

                record_as_best = False
                for split, dl in eval_dataloaders.items():
                    
                    # Compute correlation.
                    all_ca_predictions = []
                    all_ca_recordings = []
                    all_images = []
                    all_loss_weights = []

                    for batch_ind, batch in enumerate(dl):
                        image_batch = batch[0].numpy()
                        current_batch = batch[1].numpy()
                        label_batch = batch[2].numpy()
                        loss_weight_batch = batch[3].numpy()

                        all_images.append(image_batch)
                        all_ca_recordings.append(label_batch)
                        all_loss_weights.append(loss_weight_batch)

                        # Trained.
                        ca_predictions = vmapped_predict(
                            transform_params.forward(opt_params),
                            transform_basal.forward(opt_basal_params),
                            transform_somatic.forward(opt_somatic_params),
                            current_batch,
                            init_states,
                        )
                        all_ca_predictions.append(ca_predictions)

                    all_images = np.concatenate(all_images, axis=0)
                    all_ca_recordings = np.concatenate(all_ca_recordings, axis=0)
                    all_loss_weights = np.concatenate(all_loss_weights, axis=0)
                    all_ca_predictions = np.concatenate(all_ca_predictions, axis=0)

            
                    # Compute correlation and mean absolute error for every ROI.
                    trained_rhos = []
                    mae_trained = []
                    for roi_id in range(len(avg_recordings)):
                        roi_was_measured = all_loss_weights[:, roi_id].astype(bool)

                        # correlation
                        rho_trained = np.corrcoef(
                            all_ca_recordings[roi_was_measured, roi_id],
                            all_ca_predictions[roi_was_measured, roi_id],
                        )[0, 1]
                        trained_rhos.append(rho_trained)
                          
                        # absolute error
                        mae_trained.append(np.mean(np.abs(all_ca_recordings[roi_was_measured, roi_id] - all_ca_predictions[roi_was_measured, roi_id])))


                    avg_rho = np.mean(trained_rhos)
                    avg_mae = np.mean(mae_trained)
                    log.info(f"AVG rho on {split} data: {avg_rho}")
                    log.info(f"AVG mae on {split} data: {avg_mae}")

                    #save all trained maes for this epoch and split
                    with open(f"{os.getcwd()}/rhos/{split}_mae_all_epoch_{epoch}.pkl", "wb") as handle:
                        pickle.dump(mae_trained, handle)
                    
                    if split == "val" and avg_rho > best_validation_rho:
                        best_validation_rho = avg_rho
                        record_as_best = True
                    if split == "train" and record_as_best:
                        best_train_rho = avg_rho
                    if split == "test" and record_as_best:
                        best_test_rho = avg_rho



                log.info(f"Current best rhos: train {best_train_rho}, val {best_validation_rho}, test {best_test_rho}")
                with open(f"{os.getcwd()}/rhos/train_rho.pkl", "wb") as handle:
                    pickle.dump(best_train_rho, handle)
                with open(f"{os.getcwd()}/rhos/val_rho.pkl", "wb") as handle:
                    pickle.dump(best_validation_rho, handle)
                with open(f"{os.getcwd()}/rhos/test_rho.pkl", "wb") as handle:
                    pickle.dump(best_test_rho, handle)

                # save all trained rhos
                with open(f"{os.getcwd()}/rhos/train_rho_all_epoch_{epoch}.pkl", "wb") as handle:
                    pickle.dump(trained_rhos, handle)

                

                    
                

        log.info(f"================= Epoch {epoch}, loss: {epoch_loss} ===============")
        epoch_losses.append(epoch_loss)

        # Visuaisation of predictions
        if cfg["vis"]:
            log.info(f"Visualizing histograms")
            plot_prediction_histogram(vmapped_predict,
                              opt_params,
                              opt_basal_params,
                              opt_somatic_params,
                              currents,
                              init_states,
                              epoch)
            



        # 2) Evaluate after every epoch
        record_as_best = False
        for split, dl in eval_dataloaders.items():

            # Compute correlation.
            all_ca_predictions = []
            all_ca_recordings = []
            all_images = []
            all_loss_weights = []

            for batch_ind, batch in enumerate(dl):
                image_batch = batch[0].numpy()
                current_batch = batch[1].numpy()
                label_batch = batch[2].numpy()
                loss_weight_batch = batch[3].numpy()

                all_images.append(image_batch)
                all_ca_recordings.append(label_batch)
                all_loss_weights.append(loss_weight_batch)

                # Trained.
                ca_predictions = vmapped_predict(
                    transform_params.forward(opt_params),
                    transform_basal.forward(opt_basal_params),
                    transform_somatic.forward(opt_somatic_params),
                    current_batch,
                    init_states,
                )
                all_ca_predictions.append(ca_predictions)

            all_images = np.concatenate(all_images, axis=0)
            all_ca_recordings = np.concatenate(all_ca_recordings, axis=0)
            all_loss_weights = np.concatenate(all_loss_weights, axis=0)
            all_ca_predictions = np.concatenate(all_ca_predictions, axis=0)
            

            # save predictions and labels and currents
            with open(f"{os.getcwd()}/predictions_labels/predictions_epoch_{epoch}_split_{split}.pkl", "wb") as handle:
                pickle.dump(all_ca_predictions, handle)
            
            # save things that do not change seperately 
            if epoch == 0:
                with open(f"{os.getcwd()}/predictions_labels/labels_split_{split}.pkl", "wb") as handle:
                    pickle.dump(all_ca_recordings, handle)
                with open(f"{os.getcwd()}/predictions_labels/loss_weights_split_{split}.pkl", "wb") as handle:
                    pickle.dump(all_loss_weights, handle)
                with open(f"{os.getcwd()}/predictions_labels/currents_split_{split}.pkl", "wb") as handle:
                    pickle.dump(current_batch, handle)

            

            # Compute correlation and mean absolute error for every ROI.
            trained_rhos = []
            mae_trained = []

            for roi_id in range(len(avg_recordings)):
                # compute the correlation for predicted vs lael across masurements
                # for each roi seperately
                roi_was_measured = all_loss_weights[:, roi_id].astype(bool)
                
                # we can only do the correlation across time ponts if enought data
                # is in the validation set. Otherwise we do it over rois 
                if len(roi_was_measured)  > 1:    
                    rho_trained = np.corrcoef(
                        all_ca_recordings[roi_was_measured, roi_id],
                        all_ca_predictions[roi_was_measured, roi_id],
                    )[0, 1]

                else:
                    corr_roi = np.corrcoef(all_ca_recordings,all_ca_predictions)[0,1]
                    log.info(f"ROI {roi_id} has not enough data points to compute correlation. Setting it to zero. But correlation acorss rois is {corr_roi}")
                    rho_trained = 0.0
                
                # absolute error
                mae_trained.append(np.mean(np.abs(all_ca_recordings[roi_was_measured, roi_id] - all_ca_predictions[roi_was_measured, roi_id])))
                trained_rhos.append(rho_trained)

            avg_rho = np.mean(trained_rhos)
            avg_mae = np.mean(mae_trained)
            log.info(f"AVG rho on {split} data: {avg_rho}")
            log.info(f"AVG Mean Absolute Error on {split} data: {avg_mae}")
            
            # save all trained maes for this epoch and split
            with open(f"{os.getcwd()}/rhos/{split}_mae_all_epoch_{epoch}.pkl", "wb") as handle:
                pickle.dump(mae_trained, handle)

            # Save all trained rhos to see how they develop 
            with open(f"{os.getcwd()}/rhos/{split}_rho_all_epoch_{epoch}.pkl", "wb") as handle:
                pickle.dump(trained_rhos, handle)
                    

            if split == "val" and avg_rho > best_validation_rho:
                best_validation_rho = avg_rho
                record_as_best = True
            if split == "train" and record_as_best:
                best_train_rho = avg_rho
            if split == "test" and record_as_best:
                best_test_rho = avg_rho

        log.info(f"Current best rhos: train {best_train_rho}, val {best_validation_rho}, test {best_test_rho}")
        with open(f"{os.getcwd()}/rhos/train_rho.pkl", "wb") as handle:
            pickle.dump(best_train_rho, handle)
        with open(f"{os.getcwd()}/rhos/val_rho.pkl", "wb") as handle:
            pickle.dump(best_validation_rho, handle)
        with open(f"{os.getcwd()}/rhos/test_rho.pkl", "wb") as handle:
            pickle.dump(best_test_rho, handle)



    # ################################################ RF Figures ########################################################    
    log.info("Creating Receptive Field Figures ... ")

    if cfg["vis"]:
        
        try:
            create_rf_plots(counters=[0, 1, 2, 3, 4],
                            cell=cell,
                            all_loss_weights=all_loss_weights,
                            all_ca_predictions=all_ca_predictions,
                            all_images=all_images,
                            avg_recordings=avg_recordings,
                            setup=setup,
                            epoch=epoch)  
            plt.show()
        except Exception as e:
            log.info(f"Could not create RF plots. Error: {e}")



    with open(f"{os.getcwd()}/opt_params/params_{epoch+1}.pkl", "wb") as handle:
        pickle.dump([opt_params, opt_basal_params, opt_somatic_params], handle)

    log.info(f"Finished")


if __name__ == "__main__":
    run()
