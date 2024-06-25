"""
dataloader.py - This module contains the dataset class for the CT images.
In this module, three classes are defined: TrainingDataset, TestDataset, 
and ValidationDataset. Each class inherits from the Dataset class in the 
torch.utils.data module. Specifically, the TrainingDataset class is used
to load the training data, the TestDataset class is used to load the test
data, and the ValidationDataset class is used to load the validation data.
The external package, h5py, is used to read the hdf5 files containing the
CT images.
"""

from torch.utils.data import Dataset
import h5py
import glob
import os


class TrainingDataset(Dataset):

    """
    This class is used to load the training data, using the Dataset class
    from the torch.utils.data module as well as the h5py package to read
    the hdf5 files containing the CT images.
    """

    def __init__(self, ground_truth_path, observation_path):
        """
        Initialize the TrainingDataset class.

        Parameters
        ----------
        ground_truth_path : str
            The path to the ground truth hdf5 files.
        observation_path : str
            The path to the observation hdf5 files.
        """
        # Get the number of hdf5 files at the data path
        ground_truth_path_pattern = os.path.join(ground_truth_path + "*.hdf5")
        observation_path_pattern = os.path.join(observation_path + "*.hdf5")
        
        # List of hdf5 files in the directory
        self.ground_truth_files = glob.glob(ground_truth_path_pattern)

        # Sort the files with ascending indices
        self.ground_truth_files.sort()

        self.observation_files = glob.glob(observation_path_pattern)

        # Sort the files with descending indices
        self.observation_files.sort()
    
        # Count of hdf5 files
        self.num_of_files = len(self.ground_truth_files)

    def __len__(self):
        """
        Return the number of data samples in the dataset. The number of data
        samples is the number of hdf5 files minus one times 128 plus the number
        of data samples in the last file (108).

        Returns
        -------
        int
            The number of data samples in the training dataset.
        """
        # There are only 108 data samples in the last file
        return (self.num_of_files - 1) * 128 + 108

    def __getitem__(self, index):
        """
        Get the data sample at the given index.
        
        Parameters
        ----------
        index : int
            The index of the data sample to get.
        
        Returns
        -------
        tuple
            A tuple containing the observation and ground truth data samples.
        """
        # Get the file which the index is in
        file_index, data_index = divmod(index, 128)
        ground_truth_file = self.ground_truth_files[file_index]
        observation_file = self.observation_files[file_index]

        # Open the files
        observation = h5py.File(observation_file, "r")["data"][()][data_index]
        ground_truth = h5py.File(ground_truth_file, "r")["data"][()][data_index]

        return observation, ground_truth
    
class TestDataset(Dataset):

    """
    This class is used to load the test data, using the Dataset class
    from the torch.utils.data module as well as the h5py package to read
    the hdf5 files containing the CT images.
    """

    def __init__(self, ground_truth_path, observation_path):#
        """
        Initialize the TestDataset class.

        Parameters
        ----------
        ground_truth_path : str
            The path to the ground truth hdf5 files.
        observation_path : str
            The path to the observation hdf5 files.
        """
        # Get the number of hdf5 files at the data path
        ground_truth_path_pattern = os.path.join(ground_truth_path + "*.hdf5")
        observation_path_pattern = os.path.join(observation_path + "*.hdf5")
        
        # List of hdf5 files in the directory
        self.ground_truth_files = glob.glob(ground_truth_path_pattern)

        # Sort the files with ascending indices
        self.ground_truth_files.sort()

        self.observation_files = glob.glob(observation_path_pattern)

        # Sort the files with descending indices
        self.observation_files.sort()

        # Count of hdf5 files
        self.num_of_files = len(self.ground_truth_files)


    def __len__(self):
        """
        Return the number of data samples in the dataset. The number of data
        samples is the number of hdf5 files minus one times 128 plus the number
        of data samples in the last file (97).
        
        Returns
        -------
        int
            The number of data samples in the test dataset.
        """
        # There are only 97 data samples in the last file
        return (self.num_of_files - 1) * 128 + 97

    def __getitem__(self, index):
        """
        Get the data sample at the given index.

        Parameters
        ----------
        index : int
            The index of the data sample to get.

        Returns
        -------
        tuple
            A tuple containing the observation and ground truth data samples.
        """
        # Get the file which the index is in
        file_index, data_index = divmod(index, 128)
        ground_truth_file = self.ground_truth_files[file_index]
        observation_file = self.observation_files[file_index]

        # Open the files
        observation = h5py.File(observation_file, "r")["data"][()][data_index]
        ground_truth = h5py.File(ground_truth_file, "r")["data"][()][data_index]

        return observation, ground_truth
    
class ValidationDataset(Dataset):

    """
    This class is used to load the validation data, using the Dataset class
    from the torch.utils.data module as well as the h5py package to read
    the hdf5 files containing the CT images.
    """

    def __init__(self, ground_truth_path, observation_path):
        """
        Initialize the ValidationDataset class.

        Parameters
        ----------
        ground_truth_path : str
            The path to the ground truth hdf5 files.
        observation_path : str
            The path to the observation hdf5 files.
        """
        # Get the number of hdf5 files at the data path
        ground_truth_path_pattern = os.path.join(ground_truth_path + "*.hdf5")
        observation_path_pattern = os.path.join(observation_path + "*.hdf5")
        
        # List of hdf5 files in the directory
        self.ground_truth_files = glob.glob(ground_truth_path_pattern)

        # Sort the files with ascending indices
        self.ground_truth_files.sort()

        self.observation_files = glob.glob(observation_path_pattern)

        # Sort the files with descending indices
        self.observation_files.sort()
    
        # Count of hdf5 files
        self.num_of_files = len(self.ground_truth_files)


    def __len__(self):
        """
        Return the number of data samples in the dataset. The number of data
        samples is the number of hdf5 files minus one times 128 plus the number
        of data samples in the last file (66).
        
        Returns
        -------
        int
            The number of data samples in the validation dataset.
        """
        # There are only 66 data samples in the last file
        return (self.num_of_files - 1) * 128 + 66

    def __getitem__(self, index):
        """
        Get the data sample at the given index.

        Parameters
        ----------
        index : int
            The index of the data sample to get.
        
        Returns
        -------
        tuple
            A tuple containing the observation and ground truth data samples.
        """
        # Get the file which the index is in
        file_index, data_index = divmod(index, 128)
        ground_truth_file = self.ground_truth_files[file_index]
        observation_file = self.observation_files[file_index]

        # Open the files
        observation = h5py.File(observation_file, "r")["data"][()][data_index]
        ground_truth = h5py.File(ground_truth_file, "r")["data"][()][data_index]

        return observation, ground_truth