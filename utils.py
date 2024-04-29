import torch
import tomosipo as ts


def add_noise(ground_truth, n_detectors, n_angles, input_dimension=362):
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

    photons_per_pixel = 4096.0
    max_pixel_value = torch.max(projected_sinogram)

    # Multiply by the mu coefficient then take the exponential
    transformed_sinogram = torch.exp(-projected_sinogram / max_pixel_value) * photons_per_pixel

    # Clamp the transformed sinogram to ensure photon count is positive
    transformed_sinogram = torch.clamp(transformed_sinogram, min=0.001)

    # Add Poisson noise to the sinogram
    noisy_sinogram = torch.poisson(transformed_sinogram) / photons_per_pixel
    noisy_sinogram = -torch.log(noisy_sinogram)
    noisy_sinogram *= max_pixel_value

    # Assign target data to the noisy sinogram
    return noisy_sinogram