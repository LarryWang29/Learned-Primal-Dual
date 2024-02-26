import torch.nn as nn
import torch
from primal_dual_nets import PrimalDualNet
from dataloader import DataSetHelper
import utils
from torch.utils.data import DataLoader
import tqdm


# Define a function that trains the network
def train_network(input_dimension=362, n_detectors=513,
                  n_angles=1000, n_primal=5, n_dual=5, n_iterations=10,
                  epochs=25, learning_rate=0.001, beta=0.99):
    loss_function = nn.MSELoss()

    # Specify the paths
    data_path = "./data/truth_training_pseudodataset/"
    target_path = "./data/observation_training_pseudodataset/"

    # Create a dataset object
    dataset = DataSetHelper(data_path, target_path, index=8)

    # Obtain the first image
    train_dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Set a global seed for reproducibility
    torch.manual_seed(1029)

    model = PrimalDualNet(input_dimension=input_dimension,
                          n_detectors=n_detectors, n_angles=n_angles,
                          n_primal=n_primal, n_dual=n_dual,
                          n_iterations=n_iterations)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=learning_rate, betas=(beta, 0.999))

    # Set up a scheduler to set up cosine annealing
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    for i in range(epochs):
        # for batch in train_dataloader:
            # Obtain the ground truth
            # training_data = batch
            # print(len(train_dataloader))
        training_data = next(iter(train_dataloader))
        ground_truth = training_data[0]

        # # Print the model parameters
        # for i in model.parameters():
        #     print(i)

        # Add noise to the ground truth
        # observation = utils.add_noise(ground_truth)

        observation = training_data[1]
        print("Training...")
        # Zero the gradients
        optimizer.zero_grad()

        # Pass the observation through the network
        output = model.forward(observation).squeeze(1)
        # for i in model.parameters():
        #     print(i)
        # print(output)
        loss = loss_function(output, ground_truth)

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0,
                                    norm_type=2)

        optimizer.step()
        scheduler.step()

        # Print out the loss, weights and biases in the model
        print(f"Epoch {i + 1}/{epochs}, Loss: {loss.item()}")

    return model


model = train_network(n_primal=5, n_dual=5)

# Predict the output
model.eval()

# Specify the paths
data_path = "./data/truth_test_pseudodataset/"
target_path = "./data/observation_test_pseudodataset/"

# Create a dataset object
dataset = DataSetHelper(data_path, target_path, index=0)

# Obtain the first image
train_dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
training_data = next(iter(train_dataloader))

observation = training_data[1]
observation = utils.add_noise(training_data[0])
output = model.forward(observation).squeeze(1)

# Visualiase the reconstruction with colour bar as well as the original image
import matplotlib.pyplot as plt
plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(training_data[0].squeeze(0))
plt.colorbar()
plt.title("Ground Truth")
plt.subplot(1, 2, 2)
plt.imshow(output.detach().numpy().squeeze(0))
plt.colorbar()
plt.title("Reconstructed Image")
plt.show()

print("Done!")
