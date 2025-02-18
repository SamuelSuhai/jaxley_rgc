import jax.numpy as jnp
import numpy as np
from jax.lax import stop_gradient
import jaxley as jx
import jax.debug

import sys
sys.path.append('/gpfs01/euler/User/ssuhai/GitRepos/jaxley_rgc/deistler_our_data_and_morph')
from simulate import simulate



def plot_within_time_window():

    v = simulate(params, basal_neuron_params, somatic_neuron_params, currents, all_states, static)

