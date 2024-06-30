"""
This script generates figures to compare the performances of different algorithms by
visualising their reconstructions zoomed in on particular subsections. Specifically,
figure 4.1 in the thesis document is generated using this script.
"""

import os
import torch
import matplotlib.pyplot as plt
import sys

sys.path.append("./src")
from dataloader import TestDataset
from torch.utils.data import DataLoader
import utils as utils
import tomosipo as ts
from ts_algorithms import tv_min2d
from ts_algorithms import fbp
from models.primal_dual_nets import PrimalDualNet as LPD
from models.learned_PDHG import PrimalDualNet as LPDHG
from models.learned_primal import LearnedPrimal as LP
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset


# Predict the output
def make_comparison_plots(lpd_model, lpdhg_model, lp_model):
    """
    This function generates figures to compare the performances of different algorithms by
    visualising their reconstructions zoomed in on particular subsections. The algorithms
    compared are the Learned Primal-Dual (LPD), Learned Primal-Dual Hybrid Gradient (LPDHG),
    Learned Primal (LP) algorithms, filtered backprojection (FBP) and total variation (TV).

    Parameters
    ----------
    lpd_model : torch.nn.Module
        The trained LPD network model.
    lpdhg_model : torch.nn.Module
        The trained LPDHG network model.
    lp_model : torch.nn.Module
        The trained LP network model.
    """
    # Put the model in evaluation mode
    lpd_model.eval()
    lpdhg_model.eval()
    lp_model.eval()

    input_dimension = 362
    n_detectors = 543
    n_angles = 1000

    vg = ts.volume(
        size=(1 / input_dimension, 1, 1), shape=(1, input_dimension, input_dimension)
    )
    pg = ts.parallel(
        angles=n_angles,
        shape=(1, n_detectors),
        size=(1 / input_dimension, n_detectors / input_dimension),
    )

    A = ts.operator(vg, pg)

    # Specify the paths
    target_path = "./data/ground_truth_test/"
    input_path = "./data/observation_test/"

    # Set a global seed for reproducibility
    torch.manual_seed(1029)

    # Create a dataset object
    dataset = TestDataset(target_path, input_path)
    test_dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    test_data = next(iter(test_dataloader))

    ground_truth = test_data[1].cuda()

    observation = utils.add_noise(
        ground_truth, n_detectors=543, n_angles=1000, input_dimension=362
    ).cuda()

    with torch.no_grad():
        lpd_output = lpd_model.forward(observation).squeeze(1)
        lpdhg_output = lpdhg_model.forward(observation).squeeze(1)
        lp_output = lp_model.forward(observation).squeeze(1)

    fbp_output = fbp(A, observation)

    tv_output = tv_min2d(A, observation, 0.0001, num_iterations=1000)

    plt.figure(figsize=(10, 15))
    # Plot for Ground Truth
    ax1 = plt.subplot(3, 2, 1)
    # Hide ticks
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.imshow(test_data[1].squeeze(0), vmin=0, vmax=1, cmap="gray")
    ax1.set_title("Ground Truth")
    add_zoomed_inset(ax1, test_data[1].squeeze(0), zoom_factor=3)

    # Plot for FBP Image
    ax3 = plt.subplot(3, 2, 2)
    # Hide ticks
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.imshow(
        fbp_output.detach().cpu().numpy().squeeze(0), vmin=0, vmax=1, cmap="gray"
    )
    ax3.set_title("FBP Image")
    add_zoomed_inset(ax3, fbp_output.detach().cpu().numpy().squeeze(0), zoom_factor=3)

    # Plot for TV Image
    ax4 = plt.subplot(3, 2, 3)
    # Hide ticks
    ax4.set_xticks([])
    ax4.set_yticks([])
    ax4.imshow(tv_output.detach().cpu().numpy().squeeze(0), vmin=0, vmax=1, cmap="gray")
    ax4.set_title("TV Image")
    add_zoomed_inset(ax4, tv_output.detach().cpu().numpy().squeeze(0), zoom_factor=3)

    # Plot for LPD Image
    ax2 = plt.subplot(3, 2, 4)
    # Hide ticks
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.imshow(
        lpd_output.detach().cpu().numpy().squeeze(0), vmin=0, vmax=1, cmap="gray"
    )
    ax2.set_title("LPD Image")
    add_zoomed_inset(ax2, lpd_output.detach().cpu().numpy().squeeze(0), zoom_factor=3)

    # Plot for LPDHG Image
    ax5 = plt.subplot(3, 2, 5)
    # Hide ticks
    ax5.set_xticks([])
    ax5.set_yticks([])
    ax5.imshow(
        lpdhg_output.detach().cpu().numpy().squeeze(0), vmin=0, vmax=1, cmap="gray"
    )
    ax5.set_title("Learned PDHG Image")
    add_zoomed_inset(ax5, lpdhg_output.detach().cpu().numpy().squeeze(0), zoom_factor=3)

    # Plot for LP Image
    ax6 = plt.subplot(3, 2, 6)
    # Hide ticks
    ax6.set_xticks([])
    ax6.set_yticks([])
    ax6.imshow(lp_output.detach().cpu().numpy().squeeze(0), vmin=0, vmax=1, cmap="gray")
    ax6.set_title("Learned Primal Image")
    add_zoomed_inset(ax6, lp_output.detach().cpu().numpy().squeeze(0), zoom_factor=3)

    plt.tight_layout()
    plt.savefig("figures/comparison_plots/comparison_plot.png")


def add_zoomed_inset(ax, image, zoom_factor, loc="upper right"):
    """
    This function adds a zoomed inset to the image plot.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes object to add the zoomed inset to.
    image : numpy.ndarray
        The image to add the zoomed inset to.
    zoom_factor : int
        The zoom factor of the inset.
    loc : str
        The location of the inset.
    """
    # Define the region to zoom in
    x1, x2, y1, y2 = 220, 250, 250, 280
    inset_ax = zoomed_inset_axes(ax, zoom_factor, loc=loc)
    inset_ax.imshow(image, vmin=0, vmax=0.2, cmap="gray")
    inset_ax.set_xlim(x1, x2)
    inset_ax.set_ylim(y2, y1)
    inset_ax.set_xticks([])
    inset_ax.set_yticks([])
    mark_inset(ax, inset_ax, loc1=3, loc2=4, fc="none", ec="red")


if __name__ == "__main__":
    # Create a directory to store the figures if it does not exist
    os.makedirs("figures/comparison_plots", exist_ok=True)

    input_dimension = 362
    n_detectors = 543
    n_angles = 1000
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

    lpd_model = LPD(
        input_dimension=input_dimension,
        vg=vg,
        pg=pg,
        n_primal=n_primal,
        n_dual=n_dual,
        n_iterations=n_iterations,
    ).cuda()
    lpdhg_model = LPDHG(
        input_dimension=input_dimension, vg=vg, pg=pg, n_iterations=n_iterations
    ).cuda()
    lp_model = LP(
        input_dimension=input_dimension,
        vg=vg,
        pg=pg,
        n_primal=n_primal,
        n_iterations=n_iterations,
    ).cuda()

    lpd_dicts = torch.load(
        "./checkpoints/LPD_checkpoints_default/checkpoint_epoch50.pt"
    )
    lpdhg_dicts = torch.load(
        "./checkpoints/learned_PDHG_checkpoints/checkpoint_epoch50.pt"
    )
    lp_dicts = torch.load(
        "./checkpoints/learned_primal_checkpoints/checkpoint_epoch50.pt"
    )
    lpd_model.load_state_dict(lpd_dicts["model_state_dict"])
    lpdhg_model.load_state_dict(lpdhg_dicts["model_state_dict"])
    lp_model.load_state_dict(lp_dicts["model_state_dict"])
    make_comparison_plots(
        lpd_model=lpd_model, lpdhg_model=lpdhg_model, lp_model=lp_model
    )
