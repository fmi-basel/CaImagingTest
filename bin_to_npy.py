#%%
import os
from utilities import convert_bin_to_npy, find_bin_files
import numpy as np

#%%
main_dir = "/Users/guerbura/Desktop/current/data/2p"



# Update these
bin_directory = '260205_anabg_fly1'  # Update this path accordingly
target_dir = os.path.join(main_dir, bin_directory)
height = 400
width = 750

#%%
bin_files = find_bin_files(target_dir)

for bin_file in bin_files:
    print(f"Processing {bin_file}...")
    npy_path = convert_bin_to_npy(bin_file, width=width, height=height, dtype='uint16')
    print(f"Saved {npy_path}")

# %%
