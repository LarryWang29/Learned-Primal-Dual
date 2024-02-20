from torch.utils.data import Dataset
import h5py
import tomosipo as ts
import torch

"""
This file contains the dataloader helper functions for the project.
"""


class DataSetHelper(Dataset):

    """Dataset class for CT images."""

    def __init__(self, data_path, target_path, index=8):
        # Get the directory containing the training data
        self.input_file = str(data_path) + \
                          "ground_truth_test_%03d.hdf5" % index
        self.target_file = str(target_path) + \
            "observation_test_%03d.hdf5" % index

        # Open the files
        self.input_data = h5py.File(self.input_file, "r")["data"][()]
        self.target_data = h5py.File(self.target_file, "r")["data"][()]

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        # Get the data
        train_image = self.input_data[idx]
        target_image = self.target_data[idx]

        return train_image, target_image
