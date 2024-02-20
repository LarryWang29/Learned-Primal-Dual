from dataloader import DataSetHelper
from torch.utils.data import DataLoader
import torch

# Try to import the first 10 images from the first hdf5 file

# Specify the paths
data_path = "./data/truth_test_pseudodataset/"
target_path = "./data/observation_test_pseudodataset/"

# Create a dataset object
dataset = DataSetHelper(data_path, target_path, index=0)

# Get the first 10 images
train_dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# for i, data in enumerate(train_dataloader):
#     print(data[0].shape)
#     print(data[1].shape)
#     print(data[0].dtype, data[0].shape)

# Get the first image
training_data = next(iter(train_dataloader))

ground_truth = training_data[0]
observation = training_data[1]

print(ground_truth.shape, ground_truth.dtype)
print(torch.max(ground_truth))
print(torch.max(observation))
