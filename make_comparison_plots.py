import torch
import matplotlib.pyplot as plt
from dataloader import TestDataset
from torch.utils.data import DataLoader
import utils
import tomosipo as ts
import numpy as np
from ts_algorithms import tv_min2d
from ts_algorithms import fbp
from primal_dual_nets import PrimalDualNet
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

# Predict the output
def calculate_metrics_and_make_plots(model):
    model.eval()

    input_dimension = 362
    n_detectors = 543
    n_angles = 1000

    vg = ts.volume(size=(1/input_dimension, 1, 1), shape=(1, input_dimension, input_dimension))
    pg = ts.parallel(angles=n_angles, shape=(1, n_detectors), 
                        size=(1/input_dimension, n_detectors/input_dimension))

    A = ts.operator(vg, pg)

    # Specify the paths
    target_path = "./data/ground_truth_test/"
    input_path = "./data/observation_test/"

    # Set a global seed for reproducibility
    torch.manual_seed(29)

    # Create a dataset object
    dataset = TestDataset(target_path, input_path)

    # Evaluate the image metrics for all the images in the test set

    test_dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    test_data = next(iter(test_dataloader))

    ground_truth = test_data[1].cuda()

    observation = utils.add_noise(ground_truth, n_detectors=543,
                                n_angles=1000, input_dimension=362).cuda()
    with torch.no_grad():
        output = model.forward(observation).squeeze(1)

    # Calculate the MSE loss, PSNR and SSIM of the outputs
    model_mse = torch.mean((output - ground_truth) ** 2)
    data_range = np.max(ground_truth.cpu().numpy()) - np.min(ground_truth.cpu().numpy())

    model_psnr = psnr(output.detach().cpu().numpy(), ground_truth.detach().cpu().numpy(), 
                        data_range=data_range)

    model_ssim = ssim(output.detach().cpu().numpy().squeeze(0),
                        ground_truth.detach().cpu().numpy().squeeze(0),
                        data_range=data_range)
    print("The MSE from the NN model is: ", model_mse)
    print("The PSNR from the NN model is: ", model_psnr)
    print("The SSIM from the NN model is: ", model_ssim)

    fbp_output = fbp(A, observation)
    fbp_mse = torch.mean((fbp_output - ground_truth) ** 2)
    fbp_psnr = psnr(fbp_output.detach().cpu().numpy(), ground_truth.detach().cpu().numpy(),
                        data_range=data_range)
    fbp_ssim = ssim(fbp_output.detach().cpu().numpy().squeeze(0),
                    ground_truth.detach().cpu().numpy().squeeze(0), 
                        data_range=data_range)

    print("The MSE from the FBP model is: ", fbp_mse)
    print("The PSNR from the FBP model is: ", fbp_psnr)
    print("The SSIM from the FBP model is: ", fbp_ssim)

    tv_output = tv_min2d(A, observation, 0.0001, num_iterations=1000)
    tv_mse = torch.mean((tv_output - ground_truth) ** 2)
    tv_psnr = psnr(tv_output.detach().cpu().numpy(), ground_truth.detach().cpu().numpy(),
                        data_range=data_range)
    tv_ssim = ssim(tv_output.detach().cpu().numpy().squeeze(0),
                    ground_truth.detach().cpu().numpy().squeeze(0),
                        data_range=data_range)
    print("The MSE from the TV model is: ", tv_mse)
    print("The PSNR from the TV model is: ", tv_psnr)
    print("The SSIM from the TV model is: ", tv_ssim)
    
    plt.figure(figsize=(10, 10))
    # Plot for Ground Truth
    ax1 = plt.subplot(2, 2, 1)
    # Hide ticks
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.imshow(test_data[1].squeeze(0), vmin=0, vmax=1)
    ax1.set_title("Ground Truth")
    add_zoomed_inset(ax1, test_data[1].squeeze(0), zoom_factor=3)

    # Plot for LPD Image
    ax2 = plt.subplot(2, 2, 2)
    # Hide ticks
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.imshow(output.detach().cpu().numpy().squeeze(0), vmin=0, vmax=1)
    ax2.set_title("LPD Image")
    add_zoomed_inset(ax2, output.detach().cpu().numpy().squeeze(0), zoom_factor=3)

    # Plot for FBP Image
    ax3 = plt.subplot(2, 2, 3)
    # Hide ticks
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.imshow(fbp_output.detach().cpu().numpy().squeeze(0), vmin=0, vmax=1)
    ax3.set_title("FBP Image")
    add_zoomed_inset(ax3, fbp_output.detach().cpu().numpy().squeeze(0), zoom_factor=3)

    # Plot for TV Image
    ax4 = plt.subplot(2, 2, 4)
    # Hide ticks
    ax4.set_xticks([])
    ax4.set_yticks([])
    ax4.imshow(tv_output.detach().cpu().numpy().squeeze(0), vmin=0, vmax=1)
    ax4.set_title("TV Image")
    add_zoomed_inset(ax4, tv_output.detach().cpu().numpy().squeeze(0), zoom_factor=3)

    plt.tight_layout()
    plt.savefig(f"figures/comparison_plots/checkpoint_epoch{checkpoint}.png")
    plt.show()

    # print("Done!")

def add_zoomed_inset(ax, image, zoom_factor, loc='upper right'):
    x1, x2, y1, y2 = 230, 260, 160, 200  # Coordinates for the zoomed area
    inset_ax = zoomed_inset_axes(ax, zoom_factor, loc=loc)
    inset_ax.imshow(image, vmin=0.25, vmax=0.75)
    # inset_ax.imshow(image, vmin=0, vmax=1)
    inset_ax.set_xlim(x1, x2)
    inset_ax.set_ylim(y2, y1)
    inset_ax.set_xticks([])
    inset_ax.set_yticks([])
    mark_inset(ax, inset_ax, loc1=3, loc2=4, fc="none", ec="red")

input_dimension = 362
n_detectors = 543
n_angles = 1000
n_primal = 5
n_dual = 5
n_iterations = 10

vg = ts.volume(size=(1/input_dimension, 1, 1), shape=(1, input_dimension, input_dimension))
pg = ts.parallel(angles=n_angles, shape=(1, n_detectors), 
                    size=(1/input_dimension, n_detectors/input_dimension))

# checkpoints = torch.linspace(1, 50, 50, dtype=int)
# checkpoints = torch.tensor([20, 50])
checkpoints = torch.tensor([6])

for checkpoint in checkpoints:
    model = PrimalDualNet(input_dimension=input_dimension,
                            vg=vg, pg=pg,
                            n_primal=n_primal, n_dual=n_dual,
                            n_iterations=n_iterations).cuda()
    # dicts = torch.load(f"/home/larrywang/Thesis project/dw661/checkpoints (1)/checkpoint_epoch{checkpoint}.pt")
    dicts = torch.load(f"/home/larrywang/Thesis project/dw661/full_data_checkpoints/checkpoint_epoch{checkpoint}.pt")
    model.load_state_dict(dicts["model_state_dict"])
    calculate_metrics_and_make_plots(model)
