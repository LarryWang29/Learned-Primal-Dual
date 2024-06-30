"""
This script is used to generate figures to compare the performances of LPD and TV-LPD under sparse
view scenarios by visualising their reconstructions zoomed in on particular subsections, as well as 
the difference images. Specifically, figures 4.5 and 4.6 in the thesis document are generated using
this script.
"""

import os
import torch
import sys

sys.path.append("./src")
from dataloader import TestDataset
from torch.utils.data import DataLoader
import utils as utils
import tomosipo as ts
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from models.tv_primal_dual_nets import PrimalDualNet as TVLPD
from models.primal_dual_nets import PrimalDualNet as LPD
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset


def add_zoomed_inset(
    ax, image, center, zoom_factor, zoom_width, ground_truth, loc="upper left"
):
    """
    This function adds a zoomed inset to the image plot.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes object to add the zoomed inset to.
    image : numpy.ndarray
        The image to add the zoomed inset to.
    center : list
        The center of the zoomed inset.
    zoom_factor : int
        The zoom factor of the inset.
    zoom_width : int
        The width of the zoomed inset.
    ground_truth : numpy.ndarray
        The ground truth image.
    loc : str
        The location of the inset.
    """
    center_y, center_x = center
    x1, x2 = center_x - zoom_width, center_x + zoom_width
    y1, y2 = center_y - zoom_width, center_y + zoom_width
    inset_ax = zoomed_inset_axes(ax, zoom_factor, loc=loc)
    vmin, vmax = np.min(ground_truth[y1:y2, x1:x2]), np.max(ground_truth[y1:y2, x1:x2])
    inset_ax.imshow(image, vmin=vmin, vmax=vmax, cmap="gray")
    inset_ax.set_xlim(x1, x2)
    inset_ax.set_ylim(y2, y1)
    inset_ax.set_xticks([])
    inset_ax.set_yticks([])
    mark_inset(ax, inset_ax, loc1=3, loc2=4, fc="none", ec="red")


if __name__ == "__main__":
    # Create a directory to store the figures if it does not exist
    os.makedirs("figures/TV_LPD_LPD_comparisons", exist_ok=True)

    # Load the model from the checkpoint
    input_dimension = 362
    n_detectors = 543
    n_angles = 60
    n_primal = 5
    n_dual = 5
    n_iterations = 10

    vg = ts.volume(
        size=(1 / input_dimension, 1, 1), shape=(1, input_dimension, input_dimension)
    )
    pg = ts.parallel(
        angles=n_angles,
        shape=(1, n_detectors),
        size=(1 / input_dimension, n_detectors / input_dimension),
    )

    tv_model = TVLPD(
        input_dimension=input_dimension,
        vg=vg,
        pg=pg,
        n_primal=n_primal,
        n_dual=n_dual,
        n_iterations=n_iterations,
    ).cuda()
    dicts = torch.load("./checkpoints/tv_checkpoints_sparse/checkpoint_epoch50.pt")
    tv_model.load_state_dict(dicts["model_state_dict"])

    model = LPD(
        input_dimension=input_dimension,
        vg=vg,
        pg=pg,
        n_primal=n_primal,
        n_dual=n_dual,
        n_iterations=n_iterations,
    ).cuda()
    dicts = torch.load("./checkpoints/LPD_checkpoints_sparse/checkpoint_epoch50.pt")
    model.load_state_dict(dicts["model_state_dict"])

    # Specify the paths
    target_path = "./data/ground_truth_test/"
    input_path = "./data/observation_test/"

    # Set a global seed for reproducibility
    torch.manual_seed(1029)

    # Create a dataset object
    dataset = TestDataset(target_path, input_path)

    # Evaluate the image metrics for all the images in the test set
    test_dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    for test_data in tqdm(test_dataloader):
        ground_truth = test_data[1].cuda()

        observation = utils.add_noise(
            ground_truth,
            n_detectors=543,
            n_angles=60,
            input_dimension=362,
            photons_per_pixel=1000.0,
        ).cuda()

        with torch.no_grad():
            output = model.forward(observation).squeeze(1)
            tv_output = tv_model.forward(observation).squeeze(1)

        # Calculate the MSE loss, PSNR and SSIM of the outputs
        model_mse = torch.mean((output - ground_truth) ** 2)
        data_range = np.max(ground_truth.cpu().numpy()) - np.min(ground_truth.cpu().numpy())

        model_psnr = psnr(
            output.detach().cpu().numpy(),
            ground_truth.detach().cpu().numpy(),
            data_range=data_range,
        )

        model_ssim = ssim(
            output.detach().cpu().numpy().squeeze(0),
            ground_truth.detach().cpu().numpy().squeeze(0),
            data_range=data_range,
        )
        tv_model_mse = torch.mean((tv_output - ground_truth) ** 2)
        tv_model_psnr = psnr(
            tv_output.detach().cpu().numpy(),
            ground_truth.detach().cpu().numpy(),
            data_range=data_range,
        )
        tv_model_ssim = ssim(
            tv_output.detach().cpu().numpy().squeeze(0),
            ground_truth.detach().cpu().numpy().squeeze(0),
            data_range=data_range,
        )

        if tv_model_psnr - model_psnr > 1.3 and model_mse > 1e-04:

            break

    center = np.argwhere(np.abs(ground_truth.detach().cpu().numpy().squeeze(0) - \
                        output.detach().cpu().numpy().squeeze(0)) == np.max(np.abs(ground_truth.detach().cpu().numpy().squeeze(0) - \
                                                                                   output.detach().cpu().numpy().squeeze(0))))[0]
    plt.figure(figsize=(15, 5))

    # Plot for Ground Truth
    ax1 = plt.subplot(1, 3, 1)
    # Hide ticks
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.imshow(output.detach().cpu().numpy().squeeze(0), vmin=0, vmax=1, cmap='gray')
    ax1.set_title("LPD Image")
    add_zoomed_inset(ax1, output.detach().cpu().numpy().squeeze(0), center, zoom_factor=5,
                     zoom_width=8,
                     ground_truth=ground_truth.detach().cpu().numpy().squeeze(0))
    add_zoomed_inset(ax1, output.detach().cpu().numpy().squeeze(0), [119, 12], zoom_factor=5,
                     zoom_width=8,
                     ground_truth=ground_truth.detach().cpu().numpy().squeeze(0),
                     loc='lower left')
    add_zoomed_inset(ax1, output.detach().cpu().numpy().squeeze(0), [25, 342], zoom_factor=5,
                     zoom_width=6,
                     ground_truth=ground_truth.detach().cpu().numpy().squeeze(0),
                     loc='right')

    # Plot for Ground Truth
    ax1 = plt.subplot(1, 3, 2)
    # Hide ticks
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.imshow(tv_output.detach().cpu().numpy().squeeze(0), vmin=0, vmax=1, cmap='gray')
    ax1.set_title("TV-LPD Image")
    add_zoomed_inset(ax1, tv_output.detach().cpu().numpy().squeeze(0), center, zoom_factor=5,
                     zoom_width=8,
                     ground_truth=ground_truth.detach().cpu().numpy().squeeze(0))
    add_zoomed_inset(ax1, tv_output.detach().cpu().numpy().squeeze(0), [119, 12], zoom_factor=5,
                     zoom_width=8,
                     ground_truth=ground_truth.detach().cpu().numpy().squeeze(0),
                    loc='lower left')
    add_zoomed_inset(ax1, tv_output.detach().cpu().numpy().squeeze(0), [25, 342], zoom_factor=5,
                     zoom_width=6,
                     ground_truth=ground_truth.detach().cpu().numpy().squeeze(0),
                     loc='right')
    # Plot for Ground Truth
    ax1 = plt.subplot(1, 3, 3)
    # Hide ticks
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.imshow(ground_truth.detach().cpu().numpy().squeeze(0), vmin=0, vmax=1, cmap='gray')
    ax1.set_title("Ground Truth")
    add_zoomed_inset(ax1, ground_truth.detach().cpu().numpy().squeeze(0), center, zoom_factor=5,
                     zoom_width=8,
                     ground_truth=ground_truth.detach().cpu().numpy().squeeze(0))
    add_zoomed_inset(ax1, ground_truth.detach().cpu().numpy().squeeze(0), [119, 12], zoom_factor=5,
                     zoom_width=8,
                     ground_truth=ground_truth.detach().cpu().numpy().squeeze(0),
                     loc='lower left')
    add_zoomed_inset(ax1, ground_truth.detach().cpu().numpy().squeeze(0), [25, 342], zoom_factor=5,
                     zoom_width=6,
                     ground_truth=ground_truth.detach().cpu().numpy().squeeze(0),
                     loc='right')

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig("figures/TV_LPD_LPD_comparisons/TV_LPD_comparison.png")
    plt.close()

    # Plot the differences between ground truth and the outputs
    plt.figure(figsize=(10, 5))
    # Plot for Ground Truth
    ax1 = plt.subplot(1, 2, 1)
    # Hide ticks
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.imshow(np.abs(ground_truth.detach().cpu().numpy().squeeze(0) - output.detach().cpu().numpy().squeeze(0)), vmin=0, vmax=0.1, cmap='Reds')
    ax1.set_title("LPD")

    # Plot for Ground Truth
    ax1 = plt.subplot(1, 2, 2)
    # Hide ticks
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.imshow(np.abs(ground_truth.detach().cpu().numpy().squeeze(0) - tv_output.detach().cpu().numpy().squeeze(0)), vmin=0, vmax=0.1, cmap='Reds')
    ax1.set_title("TV-LPD")
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig("figures/TV_LPD_LPD_comparisons/TV_LPD_difference.png")
    plt.close()
