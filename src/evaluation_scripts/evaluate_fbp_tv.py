import torch
import sys
sys.path.append("./src")
from dataloader import TestDataset
from torch.utils.data import DataLoader
import utils as utils
import tomosipo as ts
import numpy as np
import matplotlib.pyplot as plt
from ts_algorithms import tv_min2d, fbp
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm



def evaluate_fbp_tv(target_path, input_path, option="default"):
    # Set a global seed for reproducibility
    torch.manual_seed(1029)
    fbp_mses = []
    fbp_psnrs = []
    fbp_ssims = []

    tv_mses = []
    tv_psnrs = []
    tv_ssims = []

    input_dimension = 362
    n_detectors = 543
    if option == "default":
        n_angles = 1000
        photons_per_pixel = 4096.0
    elif option == "limited":
        n_angles = 60
        photons_per_pixel = 1000
    elif option == "sparse":
        n_angles = torch.linspace(0, torch.pi / 3, 60)
        photons_per_pixel = 1000
    else:
        raise ValueError("Invalid option")
    
    vg = ts.volume(
        size=(1 / input_dimension, 1, 1), shape=(1, input_dimension, input_dimension)
    )
    pg = ts.parallel(
        angles=n_angles,
        shape=(1, n_detectors),
        size=(1 / input_dimension, n_detectors / input_dimension),
    )

    A = ts.operator(vg, pg)

    # Create a dataset object
    # dataset = TestDataset(target_path, input_path)
    dataset = TestDataset(target_path, input_path)

    # Evaluate the image metrics for all the images in the test set
    test_dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    for _, test_data in enumerate(tqdm(test_dataloader)):
        data_range = np.max(test_data[1].cpu().numpy()) - np.min(test_data[1].cpu().numpy())

        ground_truth = test_data[1].cuda()

        observation = utils.add_noise(
            ground_truth,
            n_detectors=543,
            n_angles=n_angles,
            input_dimension=362,
            photons_per_pixel=photons_per_pixel,
        ).cuda()
        fbp_output = fbp(A, observation)

        fbp_mse = torch.mean((fbp_output - ground_truth) ** 2)
        fbp_mse = fbp_mse.detach().cpu().numpy()
        fbp_mses.append(fbp_mse)

        fbp_psnr = psnr(
            fbp_output.detach().cpu().numpy(),
            ground_truth.detach().cpu().numpy(),
            data_range=data_range,
        )
        fbp_psnrs.append(fbp_psnr)

        fbp_ssim = ssim(
            fbp_output.detach().cpu().numpy().squeeze(0),
            ground_truth.detach().cpu().numpy().squeeze(0),
            data_range=data_range,
        )
        fbp_ssims.append(fbp_ssim)

        tv_output = tv_min2d(A, observation, 0.0001, num_iterations=1000)
        tv_mse = torch.mean((tv_output - ground_truth) ** 2)
        tv_mse = tv_mse.detach().cpu().numpy()
        tv_mses.append(tv_mse)

        tv_psnr = psnr(
            tv_output.detach().cpu().numpy(),
            ground_truth.detach().cpu().numpy(),
            data_range=data_range,
        )
        tv_psnrs.append(tv_psnr)

        tv_ssim = ssim(
            tv_output.detach().cpu().numpy().squeeze(0),
            ground_truth.detach().cpu().numpy().squeeze(0),
            data_range=data_range,
        )
        tv_ssims.append(tv_ssim)

    fbp_mse_avg = sum(fbp_mses) / len(fbp_mses)
    fbp_psnr_avg = sum(fbp_psnrs) / len(fbp_psnrs)
    fbp_ssim_avg = sum(fbp_ssims) / len(fbp_ssims)

    fbp_mse_std = np.std(np.array(fbp_mses))
    fbp_psnr_std = np.std(np.array(fbp_psnrs))
    fbp_ssim_std = np.std(np.array(fbp_ssims))

    tv_mse_avg = sum(tv_mses) / len(tv_mses)
    tv_psnr_avg = sum(tv_psnrs) / len(tv_psnrs)
    tv_ssim_avg = sum(tv_ssims) / len(tv_ssims)

    tv_mse_std = np.std(np.array(tv_mses))
    tv_psnr_std = np.std(np.array(tv_psnrs))
    tv_ssim_std = np.std(np.array(tv_ssims))

    return {
        "fbp_mse_avg": fbp_mse_avg,
        "fbp_mse_std": fbp_mse_std,
        "fbp_psnr_avg": fbp_psnr_avg,
        "fbp_psnr_std": fbp_psnr_std,
        "fbp_ssim_avg": fbp_ssim_avg,
        "fbp_ssim_std": fbp_ssim_std,
        "tv_mse_avg": tv_mse_avg,
        "tv_mse_std": tv_mse_std,
        "tv_psnr_avg": tv_psnr_avg,
        "tv_psnr_std": tv_psnr_std,
        "tv_ssim_avg": tv_ssim_avg,
        "tv_ssim_std": tv_ssim_std,
    }

if __name__ == "__main__":
    # Evaluate the FBP and TV algorithms on the test set
    outputs = evaluate_fbp_tv("./data/ground_truth_test/", "./data/observation_test/")

    # Print the metrics
    print(outputs)
