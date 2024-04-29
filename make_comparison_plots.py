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
    torch.manual_seed(102)

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
    

    plt.figure()
    plt.subplot(1, 4, 1)
    plt.imshow(test_data[1].squeeze(0))
    plt.title("Ground Truth")
    plt.subplot(1, 4, 2)
    plt.imshow(output.detach().cpu().numpy().squeeze(0))
    plt.title("Reconstructed Image")
    plt.subplot(1, 4, 3)
    plt.imshow(fbp_output.detach().cpu().numpy().squeeze(0))
    plt.title("FBP Image")
    plt.subplot(1, 4, 4)
    plt.imshow(tv_output.detach().cpu().numpy().squeeze(0))
    plt.title("TV Image")
    plt.show()

    # print("Done!")

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
checkpoints = torch.tensor([20, 50])

for checkpoint in checkpoints:
    model = PrimalDualNet(input_dimension=input_dimension,
                            vg=vg, pg=pg,
                            n_primal=n_primal, n_dual=n_dual,
                            n_iterations=n_iterations).cuda()
    dicts = torch.load(f"/home/larrywang/Thesis project/dw661/checkpoints (1)/checkpoint_epoch{checkpoint}.pt")
    model.load_state_dict(dicts["model_state_dict"])
    calculate_metrics_and_make_plots(model)
