import torch.nn as nn
import torch
from primal_dual_nets import PrimalDualNet
from dataloader import DataSetHelper
import utils
from torch.utils.data import DataLoader

# Specify the paths
data_path = "./data/truth_training_pseudodataset/"
target_path = "./data/observation_training_pseudodataset/"

# Create a dataset object
dataset = DataSetHelper(data_path, target_path, index=8)

# Obtain the first image
train_dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# Set a global seed for reproducibility
torch.manual_seed(1029)


# Define a function that trains the network
def train_network(input_dimension=362, n_detectors=513,
                  n_angles=1000, n_primal=5, n_dual=5, n_iterations=10,
                  epochs=10, learning_rate=0.001, beta=0.99):
    loss_function = nn.MSELoss()

    # Obtain the ground truth
    training_data = next(iter(train_dataloader))
    ground_truth = training_data[0]

    # Add noise to the ground truth
    observation = utils.add_noise(ground_truth)
    model = PrimalDualNet(observation,
                          input_dimension=input_dimension,
                          n_detectors=n_detectors, n_angles=n_angles,
                          n_primal=n_primal, n_dual=n_dual,
                          n_iterations=n_iterations)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=learning_rate, betas=(beta, 0.999))

    for i in range(epochs):
        # Zero the gradients
        optimizer.zero_grad()

        # Pass the observation through the network
        output = model.forward()
        loss = loss_function(output, ground_truth)

        # Backward pass
        loss.backward()
        optimizer.step()

        print(f"Epoch {i + 1}/{epochs}, Loss: {loss.item()}")

    return model
