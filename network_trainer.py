import torch.nn as nn
import torch
from primal_dual_nets import PrimalDualNet
from dataloader import TrainingDataset
import utils
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import tomosipo as ts
from tqdm import tqdm

# Set a global seed for reproducibility
torch.manual_seed(1029)

# Define a function that trains the network
def train_network(input_dimension=362, n_detectors=543,
                  n_angles=1000, n_primal=5, n_dual=5, n_iterations=10,
                  epochs=5, learning_rate=0.001, beta=0.99, resume=False):
    loss_function = nn.MSELoss()

    # Set a global seed for reproducibility
    torch.manual_seed(1029)

    # Specify the paths
    target_path = "/home/larrywang/Thesis project/dw661/data/ground_truth_train/"
    input_path = "/home/larrywang/Thesis project/dw661/data/observation_train/"

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

    # Set up a scheduler to set up cosine annealing
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    if resume:
        checkpoint = torch.load("/home/larrywang/Thesis project/dw661/checkpoint.pt")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        epoch = checkpoint["epoch"]
        loss = checkpoint["loss"]



    for epoch in range(epochs):

        print(len(train_dataloader))
            
        for batch in tqdm(train_dataloader):

            ground_truth = batch[1].cuda()

            observation = utils.add_noise(ground_truth, n_detectors=n_detectors,
                                        n_angles=n_angles, input_dimension=input_dimension)

            observation.cuda()

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
            scheduler.step()

        # Print out the loss, weights and biases in the model
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

        if epoch % 10 == 0:
           save_checkpoint(epoch, model, optimizer, scheduler, loss, f"/home/larrywang/Thesis project/dw661/checkpoint.pt")


    return model

def save_checkpoint(epoch, model, optimizer, scheduler, loss, file):
    torch.save( {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "loss": loss}, file
    )


model = train_network(n_primal=5, n_dual=5, resume=True)
