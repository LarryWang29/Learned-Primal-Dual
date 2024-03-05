import torch
from dataloader import TestDataset
from torch.utils.data import DataLoader
import utils
import tomosipo as ts
import matplotlib.pyplot as plt
from primal_dual_nets import PrimalDualNet
from ts_algorithms import fbp

# Predict the output

def evaluate_model(model, input_path, target_path):
    model.eval()

    # Specify the paths
    target_path = "./data/ground_truth_test/"
    input_path = "./data/observation_test/"

    # Set a global seed for reproducibility
    # torch.manual_seed(1029)

    # Create a dataset object
    dataset = TestDataset(target_path, input_path)

    # Obtain the first image
    test_dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    test_data = next(iter(test_dataloader))

    ground_truth = test_data[1].cuda()

    observation = utils.add_noise(ground_truth, n_detectors=543,
                                n_angles=1000, input_dimension=362).cuda()
    output = model.forward(observation).squeeze(1)

    # Also use the FBP algorithm to reconstruct the image
    vg = ts.volume(size=(1/362, 1, 1), shape=(1, 362, 362))
    pg = ts.parallel(angles=1000, shape=(1, 543), 
                    size=(1/362, 543/362))
    fbp_output = fbp(ts.operator(vg, pg), observation).squeeze(1)
    fbp_output = torch.clamp(fbp_output, 0, 1)

    # Visualiase the reconstruction with colour bar as well as the original image

    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(test_data[1].squeeze(0))
    plt.colorbar()
    plt.title("Ground Truth")
    plt.subplot(1, 3, 2)
    plt.imshow(output.detach().cpu().numpy().squeeze(0))
    plt.colorbar()
    plt.title("Reconstructed Image")
    plt.subplot(1, 3, 3)
    plt.imshow(fbp_output.detach().cpu().numpy().squeeze(0))
    plt.colorbar()
    plt.title("FBP Image")
    plt.show()

    print("Done!")

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
checkpoint = torch.load("/home/larrywang/Thesis project/dw661/checkpoint_2.pt")
model.load_state_dict(checkpoint["model_state_dict"])
target_path = "./data/ground_truth_train/"
input_path = "./data/observation_train/"
evaluate_model(model, target_path, input_path)