'''
Utilities for extracting the data for my project from the data base from Jonathan.
'''
import os
import numpy as np
from matplotlib import pyplot as plt
import datajoint as dj
import h5py
import sys



def get_stimulus_array(path):
    '''Extracts stim array from .h5 '''

    assert os.path.isfile(path)
    with h5py.File(path, "r") as f:
        noise_stimulus = f['NoiseArray3D'][:].T.astype(int)
    return noise_stimulus


def extract_simulus_trigg_path(
    pres_query = database.Presentation(),
    date = "2020-07-08",
    stim_name = "noise_1500",
    field = "d1",
    exp_num = "1"
):
    
    #database.Presentation() & 'date="2020-07-08"' & 'stim_name="noise_1500"' & 'field="D1"'
    processed_query = pres_query & f'date="{date}"' & f'exp_num="{exp_num}"' & f'stim_name="{stim_name}"' & f'field="{field}"'
    data_dict = processed_query.fetch1()
    
    return (data_dict['triggertimes'],data_dict['h5_header'])
