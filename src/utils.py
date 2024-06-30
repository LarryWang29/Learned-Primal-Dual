"""
This module contains utility functions that are used in the training scripts.
They include functions to add noise to the ground truth data, make boxplots and 
violin plots of the image metrics, and save model checkpoints
"""

import torch
import tomosipo as ts
import matplotlib.pyplot as plt

def add_noise(ground_truth, n_detectors, n_angles, input_dimension=362,
              photons_per_pixel=4096.0):
    """
    This function adds noise to the ground truth data by simulating a sinogram
    using the tomosipo library. The sinogram is then transformed by multiplying
    by the mu coefficient and taking the exponential. Poisson noise is added to 
    the transformed sinogram and the noisy sinogram is returned as the input data.

    Parameters
    ----------
    ground_truth : torch.Tensor
        The ground truth data to which noise is added
    n_detectors : int
        The number of detectors in the physical geometry
    n_angles : int or torch.Tensor
        The number of projection angles in the physical geometry
    input_dimension : int
        The size of the input image
    photons_per_pixel : float
        The number of photons per pixel in the simulated sinogram

    Returns
    -------
    torch.Tensor
        The noisy sinogram with added Poisson noise
    """
    # Fix seed for reproduction
    torch.manual_seed(1029)

    # Function to add custom noise instead of using readily simulated noisy data
    vg = ts.volume(shape=(1, input_dimension, input_dimension),
                   size=(1/input_dimension, 1, 1))
    pg = ts.parallel(angles=n_angles, shape=(1, n_detectors),
                     size=(1/input_dimension, n_detectors/input_dimension))
    A = ts.operator(vg, pg)

    # Forward project input data
    projected_sinogram = A(ground_truth)

    max_pixel_value = torch.max(projected_sinogram)

    # Multiply by the mu coefficient then take the exponential
    transformed_sinogram = torch.exp(-projected_sinogram / max_pixel_value) * photons_per_pixel

    # Clamp the transformed sinogram
    transformed_sinogram = torch.clamp(transformed_sinogram, min=0.001)

    # Add Poisson noise to the sinogram
    noisy_sinogram = torch.poisson(transformed_sinogram) / photons_per_pixel
    noisy_sinogram = -torch.log(noisy_sinogram)
    noisy_sinogram *= max_pixel_value

    # Assign target data to the noisy sinogram
    return noisy_sinogram


def make_boxplot_and_violinplot(mses, psnrs, ssims, filename):
    """
    This function creates boxplots and violin plots of the image metrics
    (MSE, PSNR, SSIM) and saves them as png files.

    Parameters
    ----------
    mses : list
        A list of the mean squared errors
    psnrs : list
        A list of PSNR values
    ssims : list
        A list of SSIM values
    filename : str
        The name of the file to save the plots
    """
    plt.boxplot(mses)
    plt.title("Box plot of MSEs")
    plt.savefig("figures/" + filename + "_mses_boxplot.png")
    plt.close()

    plt.boxplot(psnrs)
    plt.title("Box plot of PSNRs")
    plt.savefig("figures/" + filename + "_psnrs_boxplot.png")
    plt.close()

    plt.boxplot(ssims)
    plt.title("Box plot of SSIMs")
    plt.savefig("figures/" + filename + "_ssims_boxplot.png")
    plt.close()

    plt.violinplot(mses)
    plt.title("Violin plot of MSEs")
    plt.savefig("figures/" + filename + "_mses_violinplot.png")
    plt.close()

    plt.violinplot(psnrs)
    plt.title("Violin plot of PSNRs")
    plt.savefig("figures/" + filename + "_psnrs_violinplot.png")
    plt.close()

    plt.violinplot(ssims)
    plt.title("Violin plot of SSIMs")
    plt.savefig("figures/" + filename + "_ssims_violinplot.png")
    plt.close()

def save_checkpoint(epoch, model, optimizer, scheduler, loss, file):
    """
    This function saves the model, optimizer, scheduler, loss and epoch to a file
    during training of Pytorch networks.

    Parameters
    ----------
    epoch : int
        The current epoch of training
    model : torch.nn.Module
        The model being trained
    optimizer : torch.optim.Optimizer
        The current optimizer state
    scheduler : torch.optim.lr_scheduler
        The current scheduler state
    loss : float
        The current loss value
    file : str
        The path to the file to save the checkpoint
    """
    torch.save( {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "loss": loss}, file
    )
