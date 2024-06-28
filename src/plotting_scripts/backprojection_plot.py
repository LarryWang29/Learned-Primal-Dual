"""
This script is used to generate example plot of the comparison between the ground truth 
and the backprojection image, as well as an example plot of a sinogram. Specifically, figures
2.2 and 2.3 in the thesis document are generated using this script.
"""
import torch
import sys

sys.path.append("./src")
from dataloader import TestDataset
from torch.utils.data import DataLoader
import tomosipo as ts
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Set a global seed for reproducibility
    torch.manual_seed(1029)

    # Specify the paths
    target_path = "./data/ground_truth_test/"
    input_path = "./data/observation_test/"

    # Create a dataset object
    dataset = TestDataset(target_path, input_path)
    test_dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    _, ground_truth = next(iter(test_dataloader))

    # Define the projection geometries
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

    # Define two subplots
    plt.figure(figsize=(10, 5))
    ax1 = plt.subplot(1, 2, 1)
    ax1.imshow(
        ground_truth.squeeze(0).squeeze(0).cpu().numpy(), cmap="gray", vmin=0, vmax=1
    )

    # Hide the ticks
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_title("Ground Truth")

    # Plot the backprojection image
    backprojection = A.T(A(ground_truth))
    ax2 = plt.subplot(1, 2, 2)
    ax2.imshow(backprojection.squeeze(0).squeeze(0).cpu().numpy(), cmap="gray")
    ax2.set_title("Backprojection")

    # Hide the ticks
    ax2.set_xticks([])
    ax2.set_yticks([])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("figures/backprojection_figure/backprojection_plot.png")
    plt.close()

    sinogram = A(ground_truth)
    plt.figure(figsize=(10, 6))
    plt.imshow(sinogram.squeeze(0).cpu().numpy().T, cmap="gray")

    # Hide the ticks
    plt.xticks([])
    plt.yticks([])
    plt.title("Example of a sinogram")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("figures/backprojection_figure/sinogram_plot.png")
    plt.close()
