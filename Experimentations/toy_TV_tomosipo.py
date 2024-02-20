import sys
import matplotlib.pyplot as plt
import torch
import tomosipo as ts
sys.path.append('../dw661')
from ts_algorithms import tv_min2d
from torch.utils.data import DataLoader
from dataloader import DataSetHelper
import utils

# Specify the paths
data_path = "./data/truth_training_pseudodataset/"
target_path = "./data/observation_training_pseudodataset/"

# Create a dataset object
dataset = DataSetHelper(data_path, target_path, index=8)

# Obtain the first image
train_dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
training_data = next(iter(train_dataloader))

ground_truth = training_data[0]

# Add some Poisson noise to the projection
torch.manual_seed(1029)
observation = utils.add_noise(ground_truth)

# Setup up volume and parallel projection geometry
vg = ts.volume(shape=(1, 362, 362))
pg = ts.parallel(angles=1000, shape=(1, 513))
A = ts.operator(vg, pg)

# Use the TV minimisation algorithm to reconstruct the image
reconstructed_image = tv_min2d(A, observation, lam=0.0001, num_iterations=1000)

# Visualiase the reconstruction along with original image
plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(reconstructed_image[0, :, :])
plt.colorbar()
plt.title("Reconstructed Image")
plt.subplot(1, 2, 2)
plt.imshow(ground_truth[0, :, :])
plt.colorbar()
plt.title("Original Image")
plt.show()
