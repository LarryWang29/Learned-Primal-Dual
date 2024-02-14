"""A toy file for testing the h5py library."""

import h5py
import numpy as np
filename = "./data/truth_test_pseudodataset/ground_truth_test_003.hdf5"
sinogram = "./data/observation_test_pseudodataset/observation_test_003.hdf5"


with h5py.File(filename, "r") as f:
    # Print all root level object names (aka keys)
    # these can be group or dataset names
    print(len(f))
    print("Keys: %s" % f.keys())
    # get first object name/key; may or may NOT be a group
    a_group_key = list(f.keys())[0]
    print(a_group_key)

    # get the object type for a_group_key: usually group or dataset
    # print(type(f[a_group_key]))

    # If a_group_key is a dataset name,
    # this gets the dataset values and returns as a list
    # data = list(f[a_group_key])
    data = f[a_group_key]
    # preferred methods to get dataset values:
    ds_obj = f[a_group_key]      # returns as a h5py dataset object
    ds_arr = f[a_group_key][()]  # returns as a numpy array
    print(len(data))
    print(data[0].shape)

    print(ds_obj.dtype)
    print(ds_obj[0])

    # Export the first image to a text file
    np.savetxt("first_image.txt", ds_arr[0])

    # Export the first sinogram to a text file
with h5py.File(sinogram, "r") as f:
    data = list(f[a_group_key])

    a_group_key = list(f.keys())[0]
    ds_arr = f[a_group_key][()]

    np.savetxt("first_sino.txt", ds_arr[0])
