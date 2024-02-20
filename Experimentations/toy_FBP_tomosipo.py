# Import necessary libraries
import sys
sys.path.append('../dw661')
import tomosipo as ts
from ts_algorithms import fbp
from torch.utils.data import DataLoader
from dataloader import DataSetHelper
import matplotlib.pyplot as plt
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
observation = utils.add_noise(ground_truth)

# Setup up volume and parallel projection geometry
vg = ts.volume(shape=(1, 362, 362))
pg = ts.parallel(angles=1000, shape=(1, 513))
A = ts.operator(vg, pg)

# Now use the forward projection algorithm proposed by the paper to construct
# the sinogram
reconstructed_image = fbp(A, observation)

# Visualiase the reconstruction with colour bar
plt.imshow(reconstructed_image[0, :, :])
plt.colorbar()
plt.title("Reconstructed Image")
plt.show()
