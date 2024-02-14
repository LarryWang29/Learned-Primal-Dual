# Import necessary libraries
import torch
import tomosipo as ts
from ts_algorithms import fbp
from torch.utils.data import DataLoader
from dataloader import DataSetHelper
import numpy as np
import matplotlib.pyplot as plt

# Specify the paths
data_path = "./data/truth_test_pseudodataset/"
target_path = "./data/observation_test_pseudodataset/"

# Create a dataset object
dataset = DataSetHelper(data_path, target_path, index=3)

# Obtain the first image
train_dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
training_data = next(iter(train_dataloader))

ground_truth = training_data[0]
observation = training_data[1]

# Setup up volume and parallel projection geometry
vg = ts.volume(shape=(1, 362, 362))
pg = ts.parallel(angles=1000, shape=(1, 513))
A = ts.operator(vg, pg)

# Forward project
projected_sinogram = A(ground_truth)

# Visualise the sinogram
plt.imshow(projected_sinogram[0, :, :])
plt.colorbar()
plt.title("Projected Sinogram")
plt.show()