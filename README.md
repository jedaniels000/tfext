# TFExt
This project contains extended functionality for TensorFlow including new layers, loss functions, and generators. It can be used directly, or just viewed for examples of implementing custom functionality. Also available as a (very WIP) anaconda package: conda install -c jedaniels000 tfext.

# Code Structure
## base-tfext
Contains the 'setup.py' file used in creating a conda package as well as the **main code folder, 'tfext'**. 
### tfext
- tf_ext.py: Extended functionality for TensorFlow including new layers, loss functions, and generators. Can be used directly, or just viewed for examples of implementing custom functionality.

## tfext-conda-build
Contains other files used in creating a conda package. No functionality code here.

# Usage
## Creating a local build of the Anaconda package
- Navigate to the directory: tfext/tfext-conda-build
- Run the following command: `conda-build .`
- Alternatively, from the root directory, the package can be built by running `conda-build tfext-conda-build/`