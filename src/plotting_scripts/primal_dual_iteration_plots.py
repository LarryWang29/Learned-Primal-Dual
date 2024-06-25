import torch
import matplotlib.pyplot as plt
from src.dataloader import TestDataset
from torch.utils.data import DataLoader
import src.utils as utils
import numpy as np
from models.primal_dual_nets import PrimalDualNet
import tomosipo as ts

def make_iteration_plot(model):
    """
    This function makes plots for reconstructions at every second iteration of 
    the primal-dual algorithm.
    """
    target_path = "./data/ground_truth_test/"
    input_path = "./data/observation_test/"

    # Create a dataset object
    dataset = TestDataset(target_path, input_path)
    test_dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    test_data = next(iter(test_dataloader))

    ground_truth = test_data[1].cuda()

    observation = utils.add_noise(ground_truth, n_detectors=543,
                                n_angles=1000, input_dimension=362).cuda()
    
    height, width = observation.shape[1:]

    # Using 1 as the batch size
    primal = torch.zeros(1, model.n_primal, model.input_dimension, model.input_dimension).cuda()
    dual = torch.zeros(1, model.n_dual, height, width).cuda()

    A = model.op
    AT = model.adj_op

    fig_primal, axs_primal = plt.subplots(2, 5, figsize=(20, 8))
    fig_dual, axs_dual = plt.subplots(1, 5, figsize=(20, 2.5))

    for i in range(model.n_iterations):
        with torch.no_grad():
            dual = model.dual_list[i].forward(dual, A(primal[:, 1:2, ...]), observation.unsqueeze(1))
            primal = model.primal_list[i].forward(primal, AT(dual[:, 0:1, ...]))

            if (i+1) % 2 == 0:
                # Hide all the ticks
                for ax in axs_primal.flat:
                    ax.axis('off')

                for ax in axs_dual.flat:
                    ax.axis('off')
                axs_primal[0, (i-1)//2].imshow(primal[0, 0, ...].cpu().numpy(), cmap='gray')
                axs_primal[1, (i-1)//2].imshow(primal[0, 1, ...].cpu().numpy(), cmap='gray')

                # Calculate the quartile range of the dual, use it as the vmin and vmax to exclude outliers
                q1 = np.percentile(dual[0, 0, ...].cpu().numpy(), 10)
                q3 = np.percentile(dual[0, 0, ...].cpu().numpy(), 90)
                axs_dual[(i-1)//2].imshow(dual[0, 0, ...].cpu().numpy().T, vmin=q1, vmax=q3, cmap='gray')
    fig_dual.tight_layout()
    fig_primal.tight_layout()

    fig_primal.savefig("figures/iteration_plots/primal_iteration_plot.png")
    fig_dual.savefig("figures/iteration_plots/dual_iteration_plot.png")

if __name__ == "__main__":
    # Load the model from the checkpoint
    input_dimension = 362
    n_detectors = 543
    n_angles = 1000
    n_primal = 5
    n_dual = 5
    n_iterations = 10

    vg = ts.volume(size=(1/input_dimension, 1, 1), shape=(1, input_dimension, input_dimension))
    pg = ts.parallel(angles=n_angles, shape=(1, n_detectors), 
                        size=(1/input_dimension, n_detectors/input_dimension))

    model = PrimalDualNet(input_dimension=input_dimension,
                            vg=vg, pg=pg,
                            n_primal=n_primal, n_dual=n_dual,
                            n_iterations=n_iterations).cuda()

    dicts = torch.load(f"/home/larrywang/Thesis project/dw661/full_data_checkpoints/checkpoint_epoch6.pt")
    model.load_state_dict(dicts["model_state_dict"])

    make_iteration_plot(model)
