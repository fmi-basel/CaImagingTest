# CaImaging

## Installation
Some important points:
- We'll use conda to create an environment, but first:

- Make sure you're only using conda forge channel for FMI computers, if you're not sure do this:
    - conda config --add channels conda-forge
    - conda config --set channel_priority strict
    - conda config --remove channels defaults
    - conda config --show channels (only the conda-forge should appear)

- Create the environment:
Use python3.11 for the suitable Suite2p version. Otherwise the script will have problems due to the differences in Suite2p code.
    - conda create -n CaImaging python=3.11
    - conda activate CaImaging

- Install the following packages
    - suite2p: > pip install suite2p
    - For VS Code interactive kernel, ipykernel: conda install ipykernel -c conda-forge 
    - For interactive ROI selection: pyqt: conda install pyqt=5 -c conda-forge
    - other packages as necessary (once you run the code you'll see it)
    - For server computer Windows to fix 
        - Fixing the shm.dll error: conda install mkl intel-openmp --force-reinstall -c conda-forge

## Running

### Folder organization
- Folders can be organized using folder_processor.py

### Motion correction
- Motion correction can be done using the script: batch_pre_process.py