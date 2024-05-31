import torch
from dataloader import TestDataset
# from dataloader import TrainingDataset
from torch.utils.data import DataLoader
import utils
import tomosipo as ts
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ts_algorithms import tv_min2d, fbp
from tqdm import tqdm
from u_net import UNet
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# Predict the output
def evaluate_model(target_path, input_path, checkpoint_path, checkpoints):
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
    n_angles = 60
    # n_angles = torch.linspace(0, torch.pi/3, 60)
    # photons_per_pixel = 4096.0
    photons_per_pixel = 1000.0

    vg = ts.volume(size=(1/input_dimension, 1, 1), shape=(1, input_dimension, input_dimension))
    pg = ts.parallel(angles=n_angles, shape=(1, n_detectors), 
                    size=(1/input_dimension, n_detectors/input_dimension))

    A = ts.operator(vg, pg)

    # Create a dataset object
    dataset = TestDataset(target_path, input_path)
    # dataset = TrainingDataset(target_path, input_path)

    # Evaluate the image metrics for all the images in the test set
    test_dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    for checkpoint in checkpoints:
        model = UNet().cuda()
        dicts = torch.load(checkpoint_path + f"checkpoint_epoch{checkpoint}.pt")
        model.load_state_dict(dicts["model_state_dict"])

        model_mses = []
        model_psnrs = []
        model_ssims = []

        model.eval()

        poor_quality_range_array = []
        poor_quality_iqr_array = []


        good_quality_range_array = []
        good_quality_iqr_array = []

        for test_data in tqdm(test_dataloader):
            data_range = 1.0

            ground_truth = test_data[1].cuda()

            observation = utils.add_noise(ground_truth, n_detectors=543,
                                        n_angles=n_angles, input_dimension=362,
                                        photons_per_pixel=photons_per_pixel)
            observation = observation.cuda()
            
            fbp_recon = fbp(A, observation)
            with torch.no_grad():
                output = model.forward(fbp_recon.unsqueeze(0)).squeeze(1)

            # Calculate the MSE loss, PSNR and SSIM of the outputs
            model_mse = torch.mean((output - ground_truth) ** 2)
            model_mse = model_mse.detach().cpu().numpy()
            model_mses.append(model_mse)

            model_psnr = psnr(output.detach().cpu().numpy(), ground_truth.detach().cpu().numpy(), 
                                data_range=data_range)
            model_psnrs.append(model_psnr)

            model_ssim = ssim(output.detach().cpu().numpy().squeeze(0),
                                ground_truth.detach().cpu().numpy().squeeze(0), data_range=data_range)
            model_ssims.append(model_ssim)

            # if model_psnr < 30 or model_ssim < 0.6:
            #     # print the maximum value in the ground truth image, the range, the interquartile range and the standard deviation of pixel values
            #     poor_quality_range_array.append(torch.std(test_data[1]))
            #     poor_quality_iqr_array.append(torch.quantile(test_data[1], 0.75) - torch.quantile(test_data[1], 0.25))
            #     plt.figure(figsize=(15, 5))
            #     plt.subplot(1, 3, 1)
            #     plt.imshow(test_data[1].squeeze(0), vmin=0, vmax=1, cmap="gray")
            #     plt.title("Ground Truth")
            #     plt.subplot(1, 3, 2)
            #     plt.imshow(output.detach().cpu().numpy().squeeze(0), vmin=0, vmax=1, cmap="gray")
            #     plt.title("Reconstructed Image")
            #     plt.subplot(1, 3, 3)
            #     plt.imshow(np.abs(output.detach().cpu().numpy().squeeze(0) - test_data[1].detach().cpu().numpy().squeeze(0)), vmin=0, vmax=0.1, cmap="Reds")
            #     plt.title("Difference")
            #     plt.tight_layout()
            #     plt.savefig("figures/bad_quality_reconstructions/" + f"nn_model_epoch_{checkpoint}_psnr_{model_psnr}_ssim_{model_ssim}.png")
            #     plt.close()

            # if model_psnr > 47 or model_ssim > 0.99:
            #     # print the maximum value in the ground truth image, the range, the interquartile range and the standard deviation of pixel values
            #     good_quality_range_array.append(torch.std(test_data[1]))
            #     good_quality_iqr_array.append(torch.quantile(test_data[1], 0.75) - torch.quantile(test_data[1], 0.25))
            #     plt.figure(figsize=(15, 5))
            #     plt.subplot(1, 3, 1)
            #     plt.imshow(test_data[1].squeeze(0), vmin=0, vmax=1, cmap="gray")
            #     plt.title("Ground Truth")
            #     plt.subplot(1, 3, 2)
            #     plt.imshow(output.detach().cpu().numpy().squeeze(0), vmin=0, vmax=1, cmap="gray")
            #     plt.title("Reconstructed Image")
            #     plt.subplot(1, 3, 3)
            #     plt.imshow(np.abs(output.detach().cpu().numpy().squeeze(0) - test_data[1].detach().cpu().numpy().squeeze(0)), vmin=0, vmax=0.1, cmap="Reds")
            #     plt.title("Difference")
            #     plt.tight_layout()
            #     plt.savefig("figures/good_quality_reconstructions/" + f"nn_model_epoch_{checkpoint}_psnr_{model_psnr}_ssim_{model_ssim}.png")
            #     plt.close()

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

        # # Print the ranges and interquartile ranges of the pixel values for the poor and good quality reconstructions
        # print("Poor quality reconstructions:")
        # print("Range: ", torch.stack(poor_quality_range_array).mean().item())
        # print("IQR: ", torch.stack(poor_quality_iqr_array).mean().item())

        # print("Good quality reconstructions:")
        # print("Range: ", torch.stack(good_quality_range_array).mean().item())
        # print("IQR: ", torch.stack(good_quality_iqr_array).mean().item())

        # Make box plot and violin plot for each metric
        # make_boxplot_and_violinplot(model_mses, model_psnrs, model_ssims, f"nn_model_epoch_{checkpoint}")
    
    # Now compute the metrics for the FBP and TV models
    fbp_mses = []
    fbp_psnrs = []
    fbp_ssims = []
    
    tv_mses = []
    tv_psnrs = []
    tv_ssims = []

    # for test_data in tqdm(test_dataloader):
    for i, test_data in enumerate(tqdm(test_dataloader)):
        if i == 1:
            break
        data_range = 1.0

        ground_truth = test_data[1].cuda()

        observation = utils.add_noise(ground_truth, n_detectors=543,
                                    n_angles=n_angles, input_dimension=362,
                                    photons_per_pixel=photons_per_pixel).cuda()
        fbp_output = fbp(A, observation)

        fbp_mse = torch.mean((fbp_output - ground_truth) ** 2)
        fbp_mse = fbp_mse.detach().cpu().numpy()
        fbp_mses.append(fbp_mse)

        fbp_psnr = psnr(fbp_output.detach().cpu().numpy(), ground_truth.detach().cpu().numpy(),
                        data_range=data_range)
        fbp_psnrs.append(fbp_psnr)

        fbp_ssim = ssim(fbp_output.detach().cpu().numpy().squeeze(0),
                        ground_truth.detach().cpu().numpy().squeeze(0), 
                            data_range=data_range)
        fbp_ssims.append(fbp_ssim)

        tv_output = tv_min2d(A, observation, 0.0001, num_iterations=1000)
        tv_mse = torch.mean((tv_output - ground_truth) ** 2)
        tv_mse = tv_mse.detach().cpu().numpy()
        tv_mses.append(tv_mse)

        tv_psnr = psnr(tv_output.detach().cpu().numpy(), ground_truth.detach().cpu().numpy(),
                            data_range=data_range)
        tv_psnrs.append(tv_psnr)

        tv_ssim = ssim(tv_output.detach().cpu().numpy().squeeze(0),
                        ground_truth.detach().cpu().numpy().squeeze(0),
                            data_range=data_range)
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

    # Make box plot and violin plot for each metric
    # make_boxplot_and_violinplot(fbp_mses, fbp_psnrs, fbp_ssims, "fbp")
    # make_boxplot_and_violinplot(tv_mses, tv_psnrs, tv_ssims, "tv")

    return mse_avg_array, psnr_avg_array, ssim_avg_array, mse_std_array, psnr_std_array, \
    ssim_std_array, fbp_mse_avg, fbp_mse_std, fbp_psnr_avg, fbp_psnr_std, fbp_ssim_avg, fbp_ssim_std, \
    tv_mse_avg, tv_mse_std, tv_psnr_avg, tv_psnr_std, tv_ssim_avg, tv_ssim_std


def make_boxplot_and_violinplot(mses, psnrs, ssims, filename):
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

# checkpoints = torch.tensor([6])
# checkpoints = torch.linspace(5, 50, 10, dtype=int)
# checkpoints = torch.tensor([50])
checkpoints = torch.tensor([45])

# outputs = evaluate_model("./data/ground_truth_test/", "./data/observation_test/", "./UNet_checkpoints_default/", checkpoints)
outputs = evaluate_model("./data/ground_truth_test/", "./data/observation_test/", "./UNet_checkpoints_sparse/", checkpoints)
# outputs = evaluate_model("./data/ground_truth_train/", "./data/observation_train/", "./UNet_checkpoints_limited/", checkpoints)
mse_avg_array, psnr_avg_array, ssim_avg_array, mse_std_array, psnr_std_array, \
ssim_std_array, fbp_mse_avg, fbp_mse_std, fbp_psnr_avg, fbp_psnr_std, fbp_ssim_avg, fbp_ssim_std, \
tv_mse_avg, tv_mse_std, tv_psnr_avg, tv_psnr_std, tv_ssim_avg, tv_ssim_std = outputs

# Save the metrics to a csv file, where the columns are the metrics and the rows are the indices of the checkpoints
metrics_dict = {"MSE_AVG": mse_avg_array, "MSE_STD": mse_std_array,
                "PSNR_AVG": psnr_avg_array, "PSNR_STD": psnr_std_array,
                "SSIM_AVG": ssim_avg_array, "SSIM_STD": ssim_std_array}
metrics_df = pd.DataFrame(metrics_dict)
metrics_df.to_csv("UNet_metrics.csv")
print("Metrics saved to UNet_metrics.csv")

# Print the metrics for fbp and tv
print("FBP MSE: ", str(round(fbp_mse_avg, 4)) + "+-" + str(round(fbp_mse_std, 4)))
print("FBP PSNR: ", str(round(fbp_psnr_avg, 4)) + "+-" + str(round(fbp_psnr_std, 4)))
print("FBP SSIM: ", str(round(fbp_ssim_avg, 4)) + "+-" + str(round(fbp_ssim_std, 4)))

print("TV MSE: ", str(round(tv_mse_avg, 4)) + "+-" + str(round(tv_mse_std, 4)))
print("TV PSNR: ", str(round(tv_psnr_avg, 4)) + "+-" + str(round(tv_psnr_std, 4)))
print("TV SSIM: ", str(round(tv_ssim_avg, 4)) + "+-" + str(round(tv_ssim_std, 4)))
