import torch
import torch.nn as nn
from primal_dual_nets import PrimalDualNet
import tomosipo as ts
import matplotlib.pyplot as plt

def plot_imgs(height=3, cmap="gray", clim=(None, None), **kwargs):
    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(kwargs),
        figsize=(height * len(kwargs), height)
    )
    if len(kwargs) == 1:
        axes = [axes]
    for ax, (k, v) in zip(axes, kwargs.items()):
        pcm = ax.imshow(v.detach().cpu().numpy().squeeze(), cmap=cmap, clim=clim)
        fig.colorbar(pcm, ax=ax)
        ax.set_title(k)
    fig.tight_layout()

    
# Volume is the unit cube with N^3 voxels. 
# projection geometry is 1.5 units wide and 1 unit high.
# This geometry ensures that the operator norm of full_A is reasonable.
N = 128
full_vg = ts.volume(size=1, pos=0, shape=N)
full_pg = ts.parallel(angles=3 * N // 2, shape=(N, 3 * N // 2), size=(1, 1.5))
full_A = ts.operator(full_vg, full_pg)

phantom = ts.phantom.hollow_box(ts.data(full_vg)).data
phantom = torch.from_numpy(phantom)
sino = full_A(phantom)
noisy_sino = sino + sino.max() / 20 * torch.randn_like(sino)

# Only train on one sample of the data
inp = noisy_sino[64:65,:,:]
tgt = phantom[64:65,:,:]
learning_rate = 0.001
beta = 0.99

# Fix a seed
torch.manual_seed(29)

# Create the network
model  = PrimalDualNet(full_vg, full_pg, input_dimension=N, n_primal=5, n_dual=5, n_iterations=10)

# Train the network
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),
                                lr=learning_rate, betas=(beta, 0.999))

epochs = 10

for i in range(epochs):
    # for batch in train_dataloader:
        # Obtain the ground truth
        # training_data = batch
        # print(len(train_dataloader))
    # training_data = next(iter(train_dataloader))
    # ground_truth = training_data[0]

    # # Print the model parameters
    # for i in model.parameters():
    #     print(i)

    # Add noise to the ground truth
    # observation = utils.add_noise(ground_truth)

    # observation = training_data[1]
    print("Training...")
    # Zero the gradients
    optimizer.zero_grad()

    # Pass the observation through the network
    output = model.forward(inp).squeeze(1)
    # for i in model.parameters():
    #     print(i)
    # print(output)
    loss = loss_function(output, tgt)

    # Backward pass
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0,
                                norm_type=2)

    optimizer.step()

    # Print out the loss, weights and biases in the model
    print(f"Epoch {i + 1}/{epochs}, Loss: {loss.item()}")

model.eval()
reconstruction = model(inp)

# Plot the ground truth and the reconstruction
reconstruction = reconstruction.detach().cpu().numpy().squeeze()

plt.subplot(1, 2, 1)
plt.imshow(reconstruction, cmap="gray")
plt.title("Reconstruction")
plt.colorbar()
plt.subplot(1, 2, 2)
plt.imshow(tgt.detach().cpu().numpy().squeeze(), cmap="gray")
plt.title("Ground Truth")
plt.colorbar()
plt.show()
