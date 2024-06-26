"""
This script is used to train the leanred Primal  network, which was one of the alternative
models suggested by Adler et al. in their paper "Learned Primal-Dual Reconstruction" (https://arxiv.org/abs/1707.06474).
During training, there is option to resume training from a checkpoint, and the model is saved at the end of each epoch
to a checkpoint file. The script also calculates the image metrics (MSE, PSNR, SSIM) on the validation set at the end of each epoch,
and writes the metrics to a csv file.

Example usage:

``python src/training_scripts/learned_primal_network_trainer.py``

"""

import torch.nn as nn
import torch
import sys
sys.path.append("./src")
from models.learned_primal import LearnedPrimal
from dataloader import TrainingDataset, ValidationDataset
import utils as utils
from torch.utils.data import DataLoader
import numpy as np
import tomosipo as ts
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# Set a global seed for reproducibility
torch.manual_seed(1029)

# Define a function that trains the network
def train_network(input_dimension=362, n_detectors=543,
                  n_angles=1000, n_primal=5, n_iterations=10,
                  epochs=50, learning_rate=0.001, beta=0.99, resume=False,
                  checkpoint_path=None):
    """
    This function trains the continuous Learned Primal network on the training dataset.
    The function uses the Mean Squared Error loss function and the Adam optimizer.
    The function also calculates the image metrics (MSE, PSNR, SSIM) on the validation set at the end of each epoch,
    and writes the metrics to a csv file.

    Parameters
    ----------
    input_dimension : int
        The size of the input image.
    n_detectors : int
        The number of detectors in the physical geometry.
    n_angles : int or torch.Tensor
        The number of projection angles in the physical geometry.
    n_primal : int
        The number of primal channels in "history".
    n_iterations : int
        The number of unrolled iterations to run the algorithm.
    epochs : int
        The number of epochs to train the network.
    learning_rate : float
        The learning rate of the Adam optimizer.
    beta : float
        The beta parameter of the Adam optimizer.
    resume : bool
        Boolean value to determine whether to resume training from a checkpoint or train from scratch.
    checkpoint_path : str
        The path to the checkpoint file to resume training from.
    """

    loss_function = nn.MSELoss()

    # Set a global seed for reproducibility
    torch.manual_seed(1029)

    # Specify the paths
    target_path = "/home/larrywang/Thesis project/dw661/data/ground_truth_train/"
    input_path = "/home/larrywang/Thesis project/dw661/data/observation_train/"

    validation_target_path = "/home/larrywang/Thesis project/dw661/data/ground_truth_validation/"
    validation_input_path = "/home/larrywang/Thesis project/dw661/data/observation_validation/"

    # Create a dataset object
    dataset = TrainingDataset(target_path, input_path)

    # Obtain the first image
    train_dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    validation_dataset = ValidationDataset(validation_target_path, validation_input_path)

    validation_dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=False)

    # Open csv file to store validation metrics
    if not resume:
        f = open("/home/larrywang/Thesis project/dw661/learned_primal_checkpoints/validation_metrics.csv", "w")
        f.write("Epoch, MSE_avg, MSE_std, PSNR_avg, PSNR_std, SSIM_avg, SSIM_std\n")
    else:
        f = open("/home/larrywang/Thesis project/dw661/learned_primal_checkpoints/validation_metrics.csv", "a")
    vg = ts.volume(size=(1/input_dimension, 1, 1), shape=(1, input_dimension, input_dimension))
    pg = ts.parallel(angles=n_angles, shape=(1, n_detectors), 
                     size=(1/input_dimension, n_detectors/input_dimension))
    

    model = LearnedPrimal(input_dimension=input_dimension,
                          vg=vg, pg=pg,
                          n_primal=n_primal,
                          n_iterations=n_iterations).cuda()

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=learning_rate, betas=(beta, 0.999))

    # Set up a scheduler to set up cosine annealing
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 130001)
    
    if resume:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        epoch = checkpoint["epoch"]
        loss = checkpoint["loss"]
        epochs = epochs - epoch - 1

    for epoch in range(epochs):

        for iteration, training_data in enumerate(tqdm(train_dataloader), start=1):
            
            model.train()

            ground_truth = training_data[1].cuda()

            observation = utils.add_noise(ground_truth, n_detectors=n_detectors,
                                        n_angles=n_angles, input_dimension=input_dimension)

            observation.cuda()

            output = model.forward(observation).squeeze(1)
            loss = loss_function(output, ground_truth)

            # Zero the gradients
            optimizer.zero_grad()

            # Backward pass
            loss.backward()

            # Clip the gradients according to 2-norm
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0,
                                           norm_type=2)

            optimizer.step()

            # Update the scheduler
            scheduler.step()

        # Print out the loss in the model
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

        utils.save_checkpoint(epoch, model, optimizer, scheduler, loss, 
                        f"/home/larrywang/Thesis project/dw661/learned_primal_checkpoints/checkpoint_epoch{epoch+1}.pt")
        
        # Calculate the image metrics on validation set at the end of each epoch
        model.eval()

        print("Calculating image metrics on validation set")

        model_mses = []
        model_psnrs = []
        model_ssims = []

        for validation_data in tqdm(validation_dataloader):
            data_range = np.max(validation_data[1].cpu().numpy()) - np.min(validation_data[1].cpu().numpy())

            ground_truth = validation_data[1].cuda()

            observation = utils.add_noise(ground_truth, n_detectors=n_detectors,
                                        n_angles=n_angles, input_dimension=input_dimension).cuda()

            with torch.no_grad():
                output = model.forward(observation).squeeze(1)

            # Calculate the MSE loss, PSNR and SSIM of the outputs
            model_mse = torch.mean((output - ground_truth) ** 2)
            model_mse = model_mse.detach().cpu().numpy()

            model_psnr = psnr(output.detach().cpu().numpy(), ground_truth.detach().cpu().numpy(),
                                    data_range=data_range)

            model_ssim = ssim(output.detach().cpu().numpy().squeeze(0),
                                    ground_truth.detach().cpu().numpy().squeeze(0), data_range=data_range)

            model_mses.append(model_mse)
            model_psnrs.append(model_psnr)
            model_ssims.append(model_ssim)

        # Return the averages and standard deviations of the metrics
        model_mse_avg = sum(model_mses) / len(model_mses)
        model_psnr_avg = sum(model_psnrs) / len(model_psnrs)
        model_ssim_avg = sum(model_ssims) / len(model_ssims)

        model_mse_std = np.std(np.array(model_mses))
        model_psnr_std = np.std(np.array(model_psnrs))
        model_ssim_std = np.std(np.array(model_ssims))

        # Write the metrics to the csv file
        print("Writing metrics to csv file")

        f.write(f"{epoch+1}, {model_mse_avg}, {model_mse_std}, {model_psnr_avg}, {model_psnr_std}, {model_ssim_avg}, {model_ssim_std}\n")

    return model

if __name__ == "__main__":
    model = train_network(n_primal=5, resume=False)
