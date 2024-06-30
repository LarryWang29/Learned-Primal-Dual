"""
This script generates a plot to aid the demonstration of the parameterised lines 
in the Radon transform, as well as aiding demonstration the system matrix in discretisation
of Radon Transform. Figure 3.1 in the report is generated using this script.
"""

import os
import sys
sys.path.append("./src")

from curlyBrace import curlyBrace
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

if __name__ == "__main__":
    # Create a directory to store the figures if it does not exist
    os.makedirs("figures", exist_ok=True)

    # Create a new figure
    fig, ax = plt.subplots(figsize=(6, 6))

    # Set the limits for the axes
    ax.set_xlim(-4, 10)
    ax.set_ylim(-4, 10)

    # Hide the top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Move the bottom and left spines to the origin
    ax.spines['bottom'].set_position('zero')
    ax.spines['left'].set_position('zero')

    # Set ticks position
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    # Add grid lines for better visualization
    ax.grid(True, which='both')

    x_array = np.linspace(0, 4, 100)
    y_array = x_array
    plt.plot(x_array, y_array, linestyle='--')
    curlyBrace(fig, ax, (0, 0), (4, 4), 0.05, bool_auto=True, str_text='s', color='r')

    # Mark the angle between x-axis and the line
    arc = patches.Arc((0, 0), 4, 4, angle=0, theta1=0, theta2=45, color='blue')
    ax.add_patch(arc)
    ax.text(2.2, 0.5, r'$\theta$', fontsize=12)
    ax.text(8.5, 0.5, r'$L_{\theta, s}$', fontsize=12)

    perp_x_array = np.linspace(-2, 10, 100)
    perp_y_array = -1 * perp_x_array + 8
    plt.plot(perp_x_array, perp_y_array)
    plt.savefig("figures/radon_transform.png")
    plt.close()

    # Now plot the line model for the system matrix
    # Create a new figure
    fig, ax = plt.subplots(figsize=(6, 6))

    # Set the limits for the axes
    ax.set_xlim(-4, 10)
    ax.set_ylim(-4, 10)

    # Hide the top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Move the bottom and left spines to the origin
    ax.spines['bottom'].set_position('zero')
    ax.spines['left'].set_position('zero')

    # Set ticks position
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    # Add grid lines for better visualization
    ax.grid(True, which='both')

    x_array_1 = np.linspace(-2, 4, 100)
    y_array_1 = -1 * x_array_1 + 8
    plt.plot(x_array_1, y_array_1, color='b')

    x_array_2 = np.linspace(6, 10, 100)
    y_array_2 = -1 * x_array_2 + 8
    plt.plot(x_array_2, y_array_2, color='b')

    dotted_x_array = np.linspace(4, 6, 100)
    dotted_y_array = -1 * dotted_x_array + 8
    plt.plot(dotted_x_array, dotted_y_array, linestyle='--', color='r')
    curlyBrace(fig, ax, (4, 4), (6, 2), 0.05, bool_auto=True, str_text=r'$A_{ik}$', color='r')

    # Shade in the grid corresponding to (4, 4) and (6, 2)
    ax.add_patch(patches.Rectangle([4, 2], width=2, height=2, color='gray', alpha=0.5))

    ax.text(8.5, 0.5, r'$L_{i}$', fontsize=12)
    ax.text(6.2, 2.5, r'$\mu_{k}$', fontsize=12)

    plt.savefig("figures/system_matrix.png")
    plt.close()
