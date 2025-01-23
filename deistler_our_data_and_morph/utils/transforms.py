import jaxley as jx

from jaxley.optimize.transforms import ParamTransform, SigmoidTransform
# Transform building was changed in jaxley 0.5.0 see change logs
#https://jaxley.readthedocs.io/en/latest/changelog.html

transform_params_in = [
    # Note: the order is important here and should match what you have in train.py
    {"w_bc_to_rgc": SigmoidTransform(lower=0.0, upper=0.2)},
    {"axial_resistivity": SigmoidTransform(lower=100.0, upper=10_000.0)}, # wtf is 10_000
    {"radius": SigmoidTransform(lower=0.1, upper=1.0)},
]

transform_params = ParamTransform(transform_params_in)

transform_basal_in = [
    {"Na_gNa": SigmoidTransform(lower=0.0, upper=0.5)},
    {"K_gK": SigmoidTransform(lower=0.01, upper=0.1)},
    {"Leak_gLeak": SigmoidTransform(lower=1e-5, upper=1e-3)},
    {"KA_gKA": SigmoidTransform(lower=10e-3, upper=100e-3)},
    {"Ca_gCa": SigmoidTransform(lower=2e-3, upper=3e-3)},
    {"KCa_gKCa": SigmoidTransform(lower=0.02e-3, upper=0.2e-3)},
]

transform_basal = ParamTransform(transform_basal_in)

transform_somatic_in = [
    {"Na_gNa": SigmoidTransform(lower=0.05, upper=0.5)},
    {"K_gK": SigmoidTransform(lower=0.01, upper=0.1)},
    {"Leak_gLeak": SigmoidTransform(lower=1e-5, upper=1e-3)},
    {"KA_gKA": SigmoidTransform(lower=10e-3, upper=100e-3)},
    {"Ca_gCa": SigmoidTransform(lower=2e-3, upper=3e-3)},
    {"KCa_gKCa": SigmoidTransform(lower=0.02e-3, upper=0.2e-3)},
]

transform_somatic = ParamTransform(transform_somatic_in)
