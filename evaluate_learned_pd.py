import torch
from dataloader import TestDataset
from torch.utils.data import DataLoader
import utils
import tomosipo as ts
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ts_algorithms import tv_min2d, fbp
from tqdm import tqdm
from primal_dual_nets import PrimalDualNet
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
    n_angles = 1000
    n_primal = 5
    n_dual = 5
    n_iterations = 10

    vg = ts.volume(size=(1/input_dimension, 1, 1), shape=(1, input_dimension, input_dimension))
    pg = ts.parallel(angles=n_angles, shape=(1, n_detectors), 
                    size=(1/input_dimension, n_detectors/input_dimension))

    A = ts.operator(vg, pg)

    # Create a dataset object
    dataset = TestDataset(target_path, input_path)

    # Evaluate the image metrics for all the images in the test set
    test_dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    for checkpoint in checkpoints:
        model = PrimalDualNet(input_dimension=input_dimension,
                            vg=vg, pg=pg,
                            n_primal=n_primal, n_dual=n_dual,
                            n_iterations=n_iterations).cuda()
        dicts = torch.load(checkpoint_path + f"checkpoint_epoch{checkpoint}.pt")
        model.load_state_dict(dicts["model_state_dict"])

        model_mses = []
        model_psnrs = []
        model_ssims = []

        model.eval()

        for test_data in tqdm(test_dataloader):
            data_range = np.max(test_data[1].cpu().numpy()) - np.min(test_data[1].cpu().numpy())

            ground_truth = test_data[1].cuda()

            observation = utils.add_noise(ground_truth, n_detectors=543,
                                        n_angles=1000, input_dimension=362).cuda()
            with torch.no_grad():
                output = model.forward(observation).squeeze(1)

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

        # Make box plot and violin plot for each metric
        make_boxplot_and_violinplot(model_mses, model_psnrs, model_ssims, f"nn_model_epoch_{checkpoint}")
    
    # Now compute the metrics for the FBP and TV models
    fbp_mses = []
    fbp_psnrs = []
    fbp_ssims = []
    
    tv_mses = []
    tv_psnrs = []
    tv_ssims = []

    # for test_data in tqdm(test_dataloader):
    for i, test_data in enumerate(tqdm(test_dataloader)):
        if i == 5:
            break
        data_range = np.max(test_data[1].cpu().numpy()) - np.min(test_data[1].cpu().numpy())

        ground_truth = test_data[1].cuda()

        observation = utils.add_noise(ground_truth, n_detectors=543,
                                    n_angles=1000, input_dimension=362).cuda()
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
    make_boxplot_and_violinplot(fbp_mses, fbp_psnrs, fbp_ssims, "fbp")
    make_boxplot_and_violinplot(tv_mses, tv_psnrs, tv_ssims, "tv")

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

# checkpoints = torch.tensor([1])
# checkpoints = torch.linspace(2, 10, 5, dtype=int)
checkpoints = torch.tensor([50])

outputs = evaluate_model("./data/ground_truth_test/", "./data/observation_test/", "./checkpoints (1)/", checkpoints)
mse_avg_array, psnr_avg_array, ssim_avg_array, mse_std_array, psnr_std_array, \
ssim_std_array, fbp_mse_avg, fbp_mse_std, fbp_psnr_avg, fbp_psnr_std, fbp_ssim_avg, fbp_ssim_std, \
tv_mse_avg, tv_mse_std, tv_psnr_avg, tv_psnr_std, tv_ssim_avg, tv_ssim_std = outputs

# Save the metrics to a csv file, where the columns are the metrics and the rows are the indices of the checkpoints
metrics_dict = {"MSE_AVG": mse_avg_array, "MSE_STD": mse_std_array,
                "PSNR_AVG": psnr_avg_array, "PSNR_STD": psnr_std_array,
                "SSIM_AVG": ssim_avg_array, "SSIM_STD": ssim_std_array}
metrics_df = pd.DataFrame(metrics_dict)
metrics_df.to_csv("nn_metrics.csv")
print("Metrics saved to nn_metrics.csv")

# Print the metrics for fbp and tv
print("FBP MSE: ", str(round(fbp_mse_avg, 4)) + "+-" + str(round(fbp_mse_std, 4)))
print("FBP PSNR: ", str(round(fbp_psnr_avg, 4)) + "+-" + str(round(fbp_psnr_std, 4)))
print("FBP SSIM: ", str(round(fbp_ssim_avg, 4)) + "+-" + str(round(fbp_ssim_std, 4)))

print("TV MSE: ", str(round(tv_mse_avg, 4)) + "+-" + str(round(tv_mse_std, 4)))
print("TV PSNR: ", str(round(tv_psnr_avg, 4)) + "+-" + str(round(tv_psnr_std, 4)))
print("TV SSIM: ", str(round(tv_ssim_avg, 4)) + "+-" + str(round(tv_ssim_std, 4)))
