



'''
Example usage:
python 01_setup.py --date 2020-08-29 --exp_num 1 --quality_filter True

'''




import os
import numpy as np
from matplotlib import pyplot as plt
import datajoint as dj
import h5py
import sys
import pandas as pd
import shutil
import pickle
import argparse
import ast

from djimaging.user.alpha.utils import database



# Parse command line arguments
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--date', type=str, help='Date of recording')
parser.add_argument('--exp_num', type=str, help='The number of the experiment')
parser.add_argument('--quality_filter', type=bool, help='Use RF quality filter for data',default=False)

args = parser.parse_args()

assert args.date is not None and '-' in args.date, "Please provide valid date eg: 2020-07-08"


# Set these to change what cell you want
date = args.date
stimulus = "noise_1500"
exp_num = args.exp_num
cell_id = date + "_" + exp_num
field_stim_extract = "d1"  # Assume stimuli are all the same with each ROI


# Get the current working directory
cwd = os.getcwd()

# Get the username
username = os.popen('whoami').read().strip()

# Get the home directory
home_directory = os.path.expanduser("~")
print(username, home_directory)

# Set the path to the data directory where results will be stored
base_dir = f'{home_directory}/GitRepos/jaxley_rgc/deistler_our_data_and_morph'
assert os.path.exists(base_dir), f'{base_dir} does not exist.'
  





# Set config file
config_file = f'{home_directory}/datajoint/dj_{username}_conf.json'
assert os.path.isfile(config_file), f'Set the path to your config file: {config_file}'

# Path to djimaging
path_to_djimaging = f'{home_directory}/GitRepos/s-on-alpha-regional/code/AlphaDjimaging/djimaging'

# Set schema name
schema_name = 'ageuler_joesterle_alpha_ca'
indicator = 'calcium'
database.connect_dj(indicator=indicator)

# Fetch SWC path and copy morphology data
swc_path = (database.SWC() & f'date="{date}"').fetch1()['swc_path']
shutil.copy(swc_path, f'{base_dir}/morphologies/{cell_id}.swc')


# Fetch stimulus path and copy data
stim_path = (database.Presentation() & f'date="{date}"' & f'stim_name="{stimulus}"' & f'field="{field_stim_extract}"').fetch1()['h5_header']
shutil.copy(stim_path, f'{base_dir}/data/noise.h5')

# Extract all Field and Stimulus presetnation info 
stimulus_query = (database.Presentation() & f'date="{date}"' & f'stim_name="{stimulus}"' & 'field NOT LIKE "%ROI"')
recording_field_query = (database.FieldStackPos & f'date="{date}"')
presentation_query = stimulus_query * recording_field_query
field_presentation_all = presentation_query.fetch()
# pixel size for the cell
pixel_size = (database.Field() & "field='stack'" & f'date="{date}"').fetch('pixel_size_um')

# Fetch ROI position query
ROI_pos_query = (database.FieldStackPos.RoiStackPos() & f'date="{date}"') 


# Fetch calcium trace query
ca_trace_query = (database.PreprocessTraces() & f'date="{date}"' & f'stim_name="{stimulus}"' & 'field NOT LIKE "%ROI"')

# Combine data
all_roi_data = (ROI_pos_query * ca_trace_query).fetch(as_dict=True)

# Retrive morphology file name for identification
fnames = []
for (dirpath, dirnames, filenames) in os.walk("morphologies"):
    fnames.extend(filenames)



# retrieve RF data to accont for offset
rf_tab = database.get_rf_tab(quality_filter=True, inc_nl=False, rf_quality_filter=True)
field_rf_tab = database.get_rf_tab(roi_kind='field', inc_nl=False, rf_quality_filter=True, quality_filter=False)
roi_pos_tab = database.get_roi_pos_tab(stim_restriction=["stim_name='noise_2500'", "stim_name='noise_1500'"])
morph_tab = database.get_morph_tab()
df_field_rfs, field_avg_dx, field_avg_dy = database.get_field_avg_offset(rf_quality_filter=True)
df_rf_dxy = df_field_rfs.loc[date,['field_rf_dx_um','field_rf_dy_um']]

# drop the rois not in the roi_pos_tab: they did not meet the quality filter
if args.quality_filter:
    rois_fields = (rf_tab & f'date="{date}"' & f'stim_name="{stimulus}"').fetch('roi_id','field',as_dict=True)
    rois_set = {(r['field'], r['roi_id']) for r in rois_fields}
    all_roi_data = [i for i in all_roi_data 
                    if (i['field'], i['roi_id']) in rois_set]

# Prepare DataFrame
cell_df = pd.DataFrame()
for idx, roi_data in enumerate(all_roi_data):
    field_idx = int(roi_data['field'][-1]) - 1

    # Field RF offset
    delta_x = df_rf_dxy.iloc[field_idx]['field_rf_dx_um']
    delta_y = df_rf_dxy.iloc[field_idx]['field_rf_dy_um']

    # Field presentation-specific attributes
    noise_trigger_ts = field_presentation_all[field_idx]['triggertimes']

    # pixel_size = field_presentation_all[field_idx]['pixel_size_um']
    rec_pos = field_presentation_all[field_idx]['rec_cpos_stack_xyz'] * pixel_size

    roi_dict = {
        
        # Noise stimulus
        "Triggertimes_noise": noise_trigger_ts,

        # Calcium traces
        "Tracetimes0_noise": roi_data['preprocess_trace_times'],
        "Traces0_raw_noise": roi_data['preprocess_trace'],

        # ROI information
        "roi_id": roi_data['roi_id'],
        "roi_x": roi_data['roi_pos_stack_xyz'][0] * pixel_size,
        "roi_y": roi_data['roi_pos_stack_xyz'][1] * pixel_size,
        "roi_z": roi_data['roi_pos_stack_xyz'][2] * pixel_size,
        
        # Field information
        "image_center_x_uncorrected": rec_pos[0],
        "image_center_y_uncorrected": rec_pos[1],
        "image_center_x": rec_pos[0] - delta_x,
        "image_center_y": rec_pos[1] - delta_y,
        "pixel_size": pixel_size,
        "rec_id": field_idx, # we turn this into an int to avoid problems with jaxley experiment code later

        "expdate": roi_data['date'],
        #"cell_id": fnames[0], # unsure whether cell_id is morph file or date_expnumber
        "cell_id": cell_id,
        "recording_id": roi_data['date'],
        "date_expnumber": date + "_" + exp_num 
    }

    # Add to DataFrame
    new_row_df = pd.DataFrame([roi_dict])
    cell_df = pd.concat([cell_df, new_row_df], ignore_index=True)

# Save data 
with open(f'{base_dir}/results/data/setup.pkl', 'wb') as file:
    pickle.dump(cell_df, file)

print(f"Data saved successfully to {base_dir}/results/data/setup.pkl")
