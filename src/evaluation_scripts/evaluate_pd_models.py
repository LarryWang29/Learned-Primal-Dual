import torch
import sys
sys.path.append("./src")
from dataloader import TestDataset
from torch.utils.data import DataLoader
import utils as utils
import tomosipo as ts
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from models.primal_dual_nets import PrimalDualNet as LPD
from models.learned_primal import LearnedPrimal as LP
from models.learned_PDHG import PrimalDualNet as LPDHG
from models.tv_primal_dual_nets import PrimalDualNet as TVLPD
from models.continuous_primal_dual_nets import ContinuousPrimalDualNet as cLPD
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


# Predict the output
def evaluate_model(
    target_path,
    input_path,
    checkpoint_path,
    checkpoints,
    model,
    option="default",
    generate_images=False,
):
    # Set a global seed for reproducibility
    torch.manual_seed(1029)
    mse_avg_array = []
    psnr_avg_array = []
    ssim_avg_array = []

    mse_std_array = []
    psnr_std_array = []
    ssim_std_array = []

    input_dimension = 362
    n_detectors = 543
    n_primal = 5
    n_dual = 5
    n_iterations = 10

    if option == "default":
        n_angles = 1000
        photons_per_pixel = 4096.0
    elif option == "limited":
        n_angles = torch.linspace(0, torch.pi / 3, 60)
        photons_per_pixel = 1000
    elif option == "sparse":
        n_angles = 60
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
    dataset = TestDataset(target_path, input_path)

    # Evaluate the image metrics for all the images in the test set
    test_dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    for checkpoint in checkpoints:
        if model == "cLPD":
            model = cLPD(
                input_dimension=input_dimension,
                vg=vg,
                pg=pg,
                n_primal=n_primal,
                n_dual=n_dual,
                n_iterations=n_iterations,
            ).cuda()
        elif model == "LPD":
            model = LPD(
                input_dimension=input_dimension,
                vg=vg,
                pg=pg,
                n_primal=n_primal,
                n_dual=n_dual,
                n_iterations=n_iterations,
            ).cuda()
        elif model == "LPDHG":
            model = LPDHG(vg=vg, pg=pg).cuda()
        elif model == "LP":
            model = LP(
                input_dimension=input_dimension,
                vg=vg,
                pg=pg,
                n_primal=n_primal,
                n_iterations=n_iterations,
            ).cuda()
        elif model == "TV_LPD":
            model = TVLPD(
                input_dimension=input_dimension,
                vg=vg,
                pg=pg,
                n_primal=n_primal,
                n_dual=n_dual,
                n_iterations=n_iterations,
            ).cuda()
        else:
            raise ValueError("Invalid model")

        dicts = torch.load(checkpoint_path + f"checkpoint_epoch{checkpoint}.pt")
        model.load_state_dict(dicts["model_state_dict"])

        model_mses = []
        model_psnrs = []
        model_ssims = []

        model.eval()

        for test_data in tqdm(test_dataloader):
            data_range = np.max(test_data[1].cpu().numpy()) - np.min(test_data[1].cpu().numpy())

            ground_truth = test_data[1].cuda()

            observation = utils.add_noise(
                ground_truth,
                n_detectors=543,
                n_angles=n_angles,
                input_dimension=362,
                photons_per_pixel=photons_per_pixel,
            ).cuda()
            with torch.no_grad():
                output = model.forward(observation).squeeze(1)

            # Calculate the MSE loss, PSNR and SSIM of the outputs
            model_mse = torch.mean((output - ground_truth) ** 2)
            model_mse = model_mse.detach().cpu().numpy()
            model_mses.append(model_mse)

            model_psnr = psnr(
                output.detach().cpu().numpy(),
                ground_truth.detach().cpu().numpy(),
                data_range=data_range,
            )
            model_psnrs.append(model_psnr)

            model_ssim = ssim(
                output.detach().cpu().numpy().squeeze(0),
                ground_truth.detach().cpu().numpy().squeeze(0),
                data_range=data_range,
            )
            model_ssims.append(model_ssim)

            if generate_images:
                poor_quality_range_array = []
                poor_quality_iqr_array = []

                good_quality_range_array = []
                good_quality_iqr_array = []

                if model_psnr < 25 or model_ssim < 0.6:
                    poor_quality_range_array.append(torch.max(test_data[1])
                                                    - torch.min(test_data[1]))
                    poor_quality_iqr_array.append(
                        torch.quantile(test_data[1], 0.75)
                        - torch.quantile(test_data[1], 0.25)
                    )
                    plt.figure(figsize=(15, 5))
                    plt.subplot(1, 3, 1)
                    plt.imshow(test_data[1].squeeze(0), vmin=0, vmax=1, cmap="gray")
                    plt.title("Ground Truth")
                    plt.subplot(1, 3, 2)
                    plt.imshow(
                        output.detach().cpu().numpy().squeeze(0),
                        vmin=0,
                        vmax=1,
                        cmap="gray",
                    )
                    plt.title("Reconstructed Image")
                    plt.subplot(1, 3, 3)
                    plt.imshow(
                        np.abs(
                            output.detach().cpu().numpy().squeeze(0)
                            - test_data[1].detach().cpu().numpy().squeeze(0)
                        ),
                        vmin=0,
                        vmax=0.1,
                        cmap="Reds",
                    )
                    plt.title("Difference")
                    plt.tight_layout()
                    plt.savefig(
                        "figures/bad_quality_reconstructions/"
                        + f"nn_model_epoch_{checkpoint}_psnr_{model_psnr}_ssim_{model_ssim}.png"
                    )
                    plt.close()

                if model_psnr > 45 or model_ssim > 0.98:
                    good_quality_range_array.append(torch.max(test_data[1])
                                                    - torch.min(test_data[1]))
                    good_quality_iqr_array.append(
                        torch.quantile(test_data[1], 0.75)
                        - torch.quantile(test_data[1], 0.25)
                    )
                    plt.figure(figsize=(15, 5))
                    plt.subplot(1, 3, 1)
                    plt.imshow(test_data[1].squeeze(0), vmin=0, vmax=1, cmap="gray")
                    plt.title("Ground Truth")
                    plt.subplot(1, 3, 2)
                    plt.imshow(
                        output.detach().cpu().numpy().squeeze(0),
                        vmin=0,
                        vmax=1,
                        cmap="gray",
                    )
                    plt.title("Reconstructed Image")
                    plt.subplot(1, 3, 3)
                    plt.imshow(
                        np.abs(
                            output.detach().cpu().numpy().squeeze(0)
                            - test_data[1].detach().cpu().numpy().squeeze(0)
                        ),
                        vmin=0,
                        vmax=0.1,
                        cmap="Reds",
                    )
                    plt.title("Difference")
                    plt.tight_layout()
                    plt.savefig(
                        "figures/good_quality_reconstructions/"
                        + f"nn_model_epoch_{checkpoint}_psnr_{model_psnr}_ssim_{model_ssim}.png"
                    )
                    plt.close()

                    # Print the ranges and interquartile ranges of the pixel values for the poor and good quality reconstructions
                    print("Poor quality reconstructions:")
                    print("Range: ", torch.stack(poor_quality_range_array).mean().item())
                    print("IQR: ", torch.stack(poor_quality_iqr_array).mean().item())

                    print("Good quality reconstructions:")
                    print("Range: ", torch.stack(good_quality_range_array).mean().item())
                    print("IQR: ", torch.stack(good_quality_iqr_array).mean().item())

                    # Make box plot and violin plot for each metric
                    utils.make_boxplot_and_violinplot(model_mses, 
                                                      model_psnrs, model_ssims, 
                                                      f"nn_model_epoch_{checkpoint}")

        # Return the averages of the metrics
        model_mse_avg = sum(model_mses) / len(model_mses)
        model_psnr_avg = sum(model_psnrs) / len(model_psnrs)
        model_ssim_avg = sum(model_ssims) / len(model_ssims)

        model_mse_std = np.std(np.array(model_mses))
        model_psnr_std = np.std(np.array(model_psnrs))
        model_ssim_std = np.std(np.array(model_ssims))

        mse_avg_array.append(model_mse_avg)
        psnr_avg_array.append(model_psnr_avg)
        ssim_avg_array.append(model_ssim_avg)

        mse_std_array.append(model_mse_std)
        psnr_std_array.append(model_psnr_std)
        ssim_std_array.append(model_ssim_std)

    return (
        mse_avg_array,
        psnr_avg_array,
        ssim_avg_array,
        mse_std_array,
        psnr_std_array,
        ssim_std_array
    )


checkpoints = [torch.tensor([50]), torch.tensor([50]), 
               torch.tensor([50]), torch.tensor([50]), 
               torch.tensor([50]), torch.tensor([50]),
               torch.tensor([50]), torch.tensor([50]),
               torch.tensor([6])]

model_paths = ["./LPD_checkpoints_default/", "./learned_PDHG_checkpoints/",
               "./learned_primal_checkpoints/", "./LPD_checkpoints_limited/", 
               "./LPD_checkpoints_sparse/", "./tv_checkpoints_default/",
               "./tv_checkpoints_limited/", "./tv_checkpoints_sparse/",
               "./full_data_checkpoints/"]

model_types = ["LPD", "LPDHG", "LP", "LPD", "LPD", "TV_LPD", "TV_LPD", "TV_LPD", "LPD"]

options = ["default", "default", "default", "limited", "sparse",
           "default", "limited", "sparse", "default"]
for checkpoint, model_path, model_type, option in zip(checkpoints, model_paths, model_types, options):
    outputs = evaluate_model(
        "./data/ground_truth_test/", "./data/observation_test/", model_path, checkpoint, model_type, option,
        generate_images=False
    )
    (
        mse_avg_array,
        psnr_avg_array,
        ssim_avg_array,
        mse_std_array,
        psnr_std_array,
        ssim_std_array,
    ) = outputs

    # Save the metrics to a csv file, where the columns are the metrics and the rows are the indices of the checkpoints
    metrics_dict = {
        "MSE_AVG": mse_avg_array,
        "MSE_STD": mse_std_array,
        "PSNR_AVG": psnr_avg_array,
        "PSNR_STD": psnr_std_array,
        "SSIM_AVG": ssim_avg_array,
        "SSIM_STD": ssim_std_array,
    }
    metrics_df = pd.DataFrame(metrics_dict)
    metrics_df.to_csv(f"{model_path}metrics.csv", index=False)
print("Evaluation complete")
