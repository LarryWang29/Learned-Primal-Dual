import torch
import tomosipo as ts
import matplotlib.pyplot as plt

def add_noise(ground_truth, n_detectors, n_angles, input_dimension=362,
              photons_per_pixel=4096.0):
    # Fix seed for reproduction
    torch.manual_seed(1029)

    # Function to add custom noise instead of using readily simulated noisy data
    vg = ts.volume(shape=(1, input_dimension, input_dimension),
                   size=(1/input_dimension, 1, 1))
    pg = ts.parallel(angles=n_angles, shape=(1, n_detectors),
                     size=(1/input_dimension, n_detectors/input_dimension))
    A = ts.operator(vg, pg)

    # Forward project input data
    projected_sinogram = A(ground_truth)

    max_pixel_value = torch.max(projected_sinogram)

    # Multiply by the mu coefficient then take the exponential
    transformed_sinogram = torch.exp(-projected_sinogram / max_pixel_value) * photons_per_pixel

    # Clamp the transformed sinogram
    transformed_sinogram = torch.clamp(transformed_sinogram, min=0.001)

    # Add Poisson noise to the sinogram
    noisy_sinogram = torch.poisson(transformed_sinogram) / photons_per_pixel
    noisy_sinogram = -torch.log(noisy_sinogram)
    noisy_sinogram *= max_pixel_value

    # Assign target data to the noisy sinogram
    return noisy_sinogram


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

def save_checkpoint(epoch, model, optimizer, scheduler, loss, file):
    torch.save( {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "loss": loss}, file
    )
