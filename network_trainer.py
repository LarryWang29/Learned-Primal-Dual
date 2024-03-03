import torch.nn as nn
import torch
from primal_dual_nets import PrimalDualNet
from dataloader import TrainingDataset
import utils
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import tomosipo as ts

# Set a global seed for reproducibility
torch.manual_seed(1029)

# Define a function that trains the network
def train_network(input_dimension=362, n_detectors=543,
                  n_angles=1000, n_primal=5, n_dual=5, n_iterations=10,
                  epochs=100, learning_rate=0.001, beta=0.99):
    loss_function = nn.MSELoss()

    # Set a global seed for reproducibility
    torch.manual_seed(1029)

    # Specify the paths
    target_path = "./data/ground_truth_train/"
    input_path = "./data/observation_train/"

    # Create a dataset object
    dataset = TrainingDataset(target_path, input_path)

    # Obtain the first image
    train_dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    vg = ts.volume(size=(1/input_dimension, 1, 1), shape=(1, input_dimension, input_dimension))
    pg = ts.parallel(angles=n_angles, shape=(1, n_detectors), 
                     size=(1/input_dimension, n_detectors/input_dimension))

    model = PrimalDualNet(input_dimension=input_dimension,
                          vg=vg, pg=pg,
                          n_primal=n_primal, n_dual=n_dual,
                          n_iterations=n_iterations).cuda()

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=learning_rate, betas=(beta, 0.999))
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Set up a scheduler to set up cosine annealing
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)


    # TODO: Remove for later - Just training on one sample only
    # to check if it works for now
    training_data = next(iter(train_dataloader))

    ground_truth = training_data[1].cuda()

    observation = utils.add_noise(ground_truth, n_detectors=n_detectors,
                                  n_angles=n_angles, input_dimension=input_dimension)
    
    observation.cuda()

    for i in range(epochs):

        output = model.forward(observation).squeeze(1)
        loss = loss_function(output, ground_truth)

        # Zero the gradients
        optimizer.zero_grad()

        # Backward pass
        loss.backward()

        # TODO: Add gradient clipping later
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0,
                                       norm_type=2)

        optimizer.step()

        # TODO: Add scheduler update later
        # scheduler.step()

        # Print out the loss, weights and biases in the model
        print(f"Epoch {i + 1}/{epochs}, Loss: {loss.item()}")

    return model


model = train_network(n_primal=5, n_dual=5)
# Predict the output

def evaluate_model(model, input_path, target_path):
    model.eval()

    # Specify the paths
    target_path = "./data/ground_truth_train/"
    input_path = "./data/observation_train/"

    # Set a global seed for reproducibility
    torch.manual_seed(1029)

    # Create a dataset object
    dataset = TrainingDataset(target_path, input_path)

    # Obtain the first image
    train_dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    training_data = next(iter(train_dataloader))

    ground_truth = training_data[1].cuda()

    observation = utils.add_noise(ground_truth, n_detectors=543,
                                n_angles=1000, input_dimension=362).cuda()
    output = model.forward(observation).squeeze(1)

    # Visualiase the reconstruction with colour bar as well as the original image

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(training_data[1].squeeze(0))
    plt.colorbar()
    plt.title("Ground Truth")
    plt.subplot(1, 2, 2)
    plt.imshow(output.detach().cpu().numpy().squeeze(0))
    plt.colorbar()
    plt.title("Reconstructed Image")
    plt.show()

    print("Done!")

target_path = "./data/ground_truth_train/"
input_path = "./data/observation_train/"
evaluate_model(model, target_path, input_path)
