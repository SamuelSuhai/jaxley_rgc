import jax.numpy as jnp
import numpy as np
from jax.lax import stop_gradient
import jaxley as jx
import jax.debug



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