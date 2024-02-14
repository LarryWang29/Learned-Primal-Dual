import astra
from torch.utils.data import DataLoader
from dataloader import DataSetHelper
import matplotlib.pyplot as plt
import numpy as np

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

# Do a simple FBP reconstruction; create a projection geometry
proj_geom = astra.create_proj_geom("parallel", 1.0, 513,
                                   np.linspace(0, np.pi, 1000, endpoint=False))
vol_geom = astra.create_vol_geom(362, 362)
projection = astra.create_projector("linear", proj_geom, vol_geom)

# Create a sinogram from forward projection
sinogram_id, sinogram = astra.create_sino(ground_truth[0, :, :].numpy(),
                                          projection)
# Convert the absorption sinogram to an X-ray transform sinogram
x_ray_transform = np.exp(-sinogram) * 4096

# Visualise the sinogram and the actual sinogram
plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(sinogram)
plt.colorbar()
plt.title("Sinogram")
plt.subplot(1, 2, 2)
plt.imshow(observation[0, :, :])
plt.title("Observation")
plt.colorbar()
plt.show()

# Now do a simple FBP reconstruction on both the observation and
# forward projection
reconstruction = astra.data2d.create("-vol", vol_geom, 0)
cfg = astra.astra_dict('FBP')
cfg['ProjectorId'] = projection
cfg['ProjectionDataId'] = sinogram_id
cfg['ReconstructionDataId'] = reconstruction

# Create the algorithm object from the configuration structure
alg_id = astra.algorithm.create(cfg)

# Run the algorithm
astra.algorithm.run(alg_id)
V = astra.data2d.get(reconstruction)

# Visualise the reconstruction and the ground truth
plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(V)
plt.title("Reconstruction")
plt.subplot(1, 2, 2)
plt.imshow(ground_truth[0, :, :])
plt.title("Ground Truth")
plt.show()

# Clean up
astra.data2d.delete([sinogram_id, reconstruction])
astra.projector.delete(projection)
astra.algorithm.delete(alg_id)

observation = observation[0, :, :].numpy()

# Apply transforms to rescale the observation
# observation = 4096 * observation

# Now do a simple FBP reconstruction on the observation
proj_geom = astra.create_proj_geom("parallel", 1.0, 513,
                                   np.linspace(-0.5*np.pi, 0.5*np.pi, 1000,
                                               endpoint=False))
vol_geom = astra.create_vol_geom(362, 362)
sinogram_id = astra.data2d.create("-sino", proj_geom, observation)

# Make the projection geometry

projection = astra.create_projector("linear", proj_geom, vol_geom)

# Create the algorithm object from the configuration structure
reconstruction = astra.data2d.create("-vol", vol_geom, 0)
cfg = astra.astra_dict('FBP')
cfg['ProjectorId'] = projection
cfg['ProjectionDataId'] = sinogram_id
cfg['ReconstructionDataId'] = reconstruction

# Create the algorithm object from the configuration structure
alg_id = astra.algorithm.create(cfg)

# Run the algorithm
astra.algorithm.run(alg_id)
V = astra.data2d.get(reconstruction)

# Clip the reconstruction to range 0, 1
# V = np.clip(V, 0, 1)

# Visualise the reconstruction and the ground truth
plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(V)
plt.title("Reconstruction")
plt.colorbar()
plt.subplot(1, 2, 2)
plt.imshow(ground_truth[0, :, :])
plt.title("Ground Truth")
plt.colorbar()
plt.show()

# Clean up
astra.data2d.delete([sinogram_id, reconstruction])
astra.projector.delete(projection)
astra.algorithm.delete(alg_id)
