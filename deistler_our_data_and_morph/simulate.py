import jax.numpy as jnp
import numpy as np
from jax.lax import stop_gradient
import jaxley as jx
import jax.debug



def simulate_split(
    params,
    basal_neuron_params,
    somatic_neuron_params,
    currents,
    all_states,
    static,
):
    """
    
    Run simulation and return recorded voltages.
    
    Input:
    - params: list of parameters. List of dictionaries. Each dictionary contains the parameter name as key and an array as values.
             For example w_bc_to_rgc is first element of list. Contains one value for each compartment making contact with bc

    Output:
    - v: np.array - the recorded voltages: shape (nr comp were recording from, truncated time window length/dt)
    - all_states - dict with parameters as keys. vale is an array of lenght TOTAL number of compartments in cell.

    """
    all_states = stop_gradient(all_states)
    syn_weights = params[0]["w_bc_to_rgc"]
    cell_params = params[1:]

    
    input_currents = syn_weights * currents

    # Define stimuli.
    # step_currents has shape (nr comp, time window length/dt)
    step_currents = jx.datapoint_to_step_currents(
        0, static["t_max"] / static["num_truncations"], input_currents, static["dt"], static["t_max"] / static["num_truncations"]
    )
    data_stimuli = None
    for branch, comp, step_current in zip(static["stim_branch_inds"], static["stim_comps"], step_currents):
        data_stimuli = (
            static["cell"].branch(branch)
            .loc(comp)
            .data_stimulate(step_current, data_stimuli=data_stimuli)
        )

    # Define parameters.
    pstate = None
    for param in basal_neuron_params:
        name = list(param.keys())[0]
        parameter_values = param[name]
        value = parameter_values
        pstate = static["cell"][static["basal_inds"]].data_set(name, value, param_state=pstate)

    for param in somatic_neuron_params:
        name = list(param.keys())[0]
        parameter_values = param[name]
        value = parameter_values
        pstate = static["cell"][static["somatic_inds"]].data_set(name, value, param_state=pstate)

    # Run simulation.
    steps = len(static["time_vec"]) / static["num_truncations"]
    v, all_states = jx.integrate(
        static["cell"],
        params=cell_params,
        param_state=pstate,
        data_stimuli=data_stimuli,
        checkpoint_lengths=[int(np.ceil(np.sqrt(steps))), int(np.ceil(np.sqrt(steps)))],
        all_states=all_states,
        return_states=True,
    )

    return v, all_states


def simulate(params, basal_neuron_params, somatic_neuron_params, currents, all_states, static):
    '''
    Divides the simulation time window into static["num_truncations"] parts and calls the
    simulate_split function for each part. Returns the concatenated voltages.
    '''
    
    vs = []
    for _ in range(static["num_truncations"]):
        v, all_states = simulate_split(
            params, basal_neuron_params, somatic_neuron_params, currents, all_states, static
        )
        # reminder: each v has shape (nr of comp we are recording from, truncated time window length/dt)
        vs.append(v[:, 1:])

    # concatenate along the time dimension
    return jnp.concatenate(vs, axis=1)


def predict(params, basal_neuron_params, somatic_neuron_params, currents, all_states, static):
    """Return calcium activity in each of the recordings."""

    v = simulate(
        params, basal_neuron_params, somatic_neuron_params, currents, all_states, static
    )
    v = v[:, : len(static["kernel"])]
    # v as shape (nr of comp we are recording from, TOTAL time window length/dt)
    # mean over time = mean ca activity over time
    convolved_at_last_time = jnp.mean(jnp.flip(static["kernel"]) * v, axis=1)
    return (convolved_at_last_time * static["output_scale"]) + static["output_offset"]


def loss_fn(
    params,
    basal_neuron_params,
    somatic_neuron_params,
    currents,
    labels,
    loss_weights,
    all_states,
    static,
):
    """Return loss for a single (data,label) pair."""
    prediction = predict(
        params, basal_neuron_params, somatic_neuron_params, currents, all_states, static
    )
    loss_of_each_recording = jnp.abs(prediction - labels)
    loss = jnp.sum(loss_weights * loss_of_each_recording)
    


    return loss


# def sample_basal(key):
#     diff = transform_basal.uppers[key] - transform_basal.lowers[key]
#     low = transform_basal.lowers[key]
#     return jnp.asarray(low + diff * np.random.rand())


# def sample_somatic(key):
#     diff = transform_somatic.uppers[key] - transform_somatic.lowers[key]
#     low = transform_somatic.lowers[key]
#     return jnp.asarray(low + diff * np.random.rand())