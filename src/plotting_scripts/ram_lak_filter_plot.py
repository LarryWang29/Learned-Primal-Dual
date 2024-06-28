"""
This scripts generates a figure that shows ram-lak filter's high frequency magnification
and low frequency suppression properties. Figure 3.1 in the thesis is generated using this
script.
"""

import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    x_array = np.linspace(-1, 1, 201)
    ram_lak_filter = np.abs(x_array)

    plt.figure(figsize=(10, 5))
    plt.plot(x_array, ram_lak_filter)
    # Shade the area under the curve where abs(x) > 0.75
    plt.fill_between(x_array, ram_lak_filter, where=np.abs(x_array) >= 0.75,
                    color="red", alpha=0.5, label="High Frequency")
    plt.fill_between(x_array, ram_lak_filter, where=np.abs(x_array) <= 0.75,
                    color="blue", alpha=0.5, label="Low Frequency")
    plt.title("Ram-Lak Filter")
    plt.xlabel(r"$\omega$", fontsize=12)
    plt.ylabel(r"$|\omega|$", fontsize=12)
    plt.grid()
    plt.legend(loc='upper center', fontsize=12)
    plt.tight_layout()
    plt.savefig("figures/ram_lak_filter.png")
