"""
This script generates a plot to aid the demonstration of the parameterised lines 
in the Radon transform. Figure 2.1 in the report is generated using this script.
"""

import sys
sys.path.append("./src")

from curlyBrace import curlyBrace
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

if __name__ == "__main__":
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
