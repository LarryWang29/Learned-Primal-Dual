import torch
from dataloader import TestDataset
from torch.utils.data import DataLoader
import utils
import tomosipo as ts
import pandas as pd
import numpy as np
from ts_algorithms import tv_min2d, fbp
from tqdm import tqdm
from primal_dual_nets import PrimalDualNet
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# Predict the output
def evaluate_model(target_path, input_path, checkpoint_path, checkpoints):
    # Set a global seed for reproducibility
    torch.manual_seed(1029)
    mse_array = []
    psnr_array = []
    ssim_array = []
    
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
        mse_array.append(model_mse_avg)
        psnr_array.append(model_psnr_avg)
        ssim_array.append(model_ssim_avg)
    
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

    tv_mse_avg = sum(tv_mses) / len(tv_mses)
    tv_psnr_avg = sum(tv_psnrs) / len(tv_psnrs)
    tv_ssim_avg = sum(tv_ssims) / len(tv_ssims)

    return mse_array, psnr_array, ssim_array, fbp_mse_avg, fbp_psnr_avg, fbp_ssim_avg, tv_mse_avg, tv_psnr_avg, tv_ssim_avg

# checkpoints = torch.tensor([1])
# checkpoints = torch.linspace(2, 10, 5, dtype=int)
checkpoints = torch.tensor([50])

outputs = evaluate_model("./data/ground_truth_test/", "./data/observation_test/", "./checkpoints (1)/", checkpoints)
mse_array = outputs[0]
psnr_array = outputs[1]
ssim_array = outputs[2]

fbp_mse_avg = outputs[3]
fbp_psnr_avg = outputs[4]
fbp_ssim_avg = outputs[5]

tv_mse_avg = outputs[6]
tv_psnr_avg = outputs[7]
tv_ssim_avg = outputs[8]

# Save the metrics to a csv file, where the columns are the metrics and the rows are the indices of the checkpoints
metrics_dict = {"MSE": mse_array, "PSNR": psnr_array, "SSIM": ssim_array}
metrics_df = pd.DataFrame(metrics_dict)
metrics_df.to_csv("nn_metrics.csv")
print("NN metrics saved to metrics.csv")

# Print the metrics for fbp and tv
print("FBP MSE: ", fbp_mse_avg)
print("FBP PSNR: ", fbp_psnr_avg)
print("FBP SSIM: ", fbp_ssim_avg)

print("TV MSE: ", tv_mse_avg)
print("TV PSNR: ", tv_psnr_avg)
print("TV SSIM: ", tv_ssim_avg)
