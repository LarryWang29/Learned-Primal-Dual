from torch.utils.data import Dataset
import h5py
import tomosipo as ts
import torch
import glob
import os

"""
This file contains the dataloader helper functions for the project.
"""


class TrainingDataset(Dataset):

    """Dataset class for CT images."""

    def __init__(self, ground_truth_path, observation_path):
        # Get the number of hdf5 files at the data path
        ground_truth_path_pattern = os.path.join(ground_truth_path + "*.hdf5")
        observation_path_pattern = os.path.join(observation_path + "*.hdf5")
        
        # List of hdf5 files in the directory
        self.ground_truth_files = glob.glob(ground_truth_path_pattern)
        self.observation_files = glob.glob(observation_path_pattern)
    
        # Count of hdf5 files
        self.num_of_files = len(self.ground_truth_files)

    def __len__(self):
        # There are only 108 data samples in the last file
        return (self.num_of_files - 1) * 128 + 108

    def __getitem__(self, index):
        # Get the file which the index is in
        file_index, data_index = divmod(index, 128)
        ground_truth_file = self.ground_truth_files[file_index]
        observation_file = self.observation_files[file_index]

        # Open the files
        observation = h5py.File(observation_file, "r")["data"][()][data_index]
        ground_truth = h5py.File(ground_truth_file, "r")["data"][()][data_index]

        return observation, ground_truth
    
class TestDataset(Dataset):

    """Dataset class for CT images."""

    def __init__(self, ground_truth_path, observation_path):
        # Get the number of hdf5 files at the data path
        ground_truth_path_pattern = os.path.join(ground_truth_path + "*.hdf5")
        observation_path_pattern = os.path.join(observation_path + "*.hdf5")
        
        # List of hdf5 files in the directory
        self.ground_truth_files = glob.glob(ground_truth_path_pattern)
        self.observation_files = glob.glob(observation_path_pattern)
    
        # Count of hdf5 files
        self.num_of_files = len(self.ground_truth_files)


    def __len__(self):
        # There are only 97 data samples in the last file
        return (self.num_of_files - 1) * 128 + 97

    def __getitem__(self, index):
        # Get the file which the index is in
        file_index, data_index = divmod(index, 128)
        ground_truth_file = self.ground_truth_files[file_index]
        observation_file = self.observation_files[file_index]

        # Open the files
        observation = h5py.File(observation_file, "r")["data"][()][data_index]
        ground_truth = h5py.File(ground_truth_file, "r")["data"][()][data_index]

        return observation, ground_truth
    
class ValidationDataset(Dataset):

    """Dataset class for CT images."""

    def __init__(self, ground_truth_path, observation_path):
        # Get the number of hdf5 files at the data path
        ground_truth_path_pattern = os.path.join(ground_truth_path + "*.hdf5")
        observation_path_pattern = os.path.join(observation_path + "*.hdf5")
        
        # List of hdf5 files in the directory
        self.ground_truth_files = glob.glob(ground_truth_path_pattern)
        self.observation_files = glob.glob(observation_path_pattern)
    
        # Count of hdf5 files
        self.num_of_files = len(self.ground_truth_files)


    def __len__(self):
        # There are only 66 data samples in the last file
        return (self.num_of_files - 1) * 128 + 66

    def __getitem__(self, index):
        # Get the file which the index is in
        file_index, data_index = divmod(index, 128)
        ground_truth_file = self.ground_truth_files[file_index]
        observation_file = self.observation_files[file_index]

        # Open the files
        observation = h5py.File(observation_file, "r")["data"][()][data_index]
        ground_truth = h5py.File(ground_truth_file, "r")["data"][()][data_index]

        return observation, ground_truth
