"""
This module contains the implementation of the Learned Primal model, which is one
of the alternative models suggested by Adler et al. in their paper "Learned Primal-Dual
Reconstruction" (https://arxiv.org/abs/1707.06474). It's largely similar to Learned Primal
Dual, but it only learns the primal part of the algorithm; the dual step is instead just
a calculation on the current residual.
"""

import torch.nn as nn
import torch
import tomosipo as ts
from tomosipo.torch_support import to_autograd

class PrimalNet(nn.Module):
    """
    This class implements the 'Primal' networks, which are used to update the
    primal variables in the Learned Primal algorithm. The particular
    architecture used is a 3-layer CNN with PReLU activations and residual
    connections at the end.
    """
    def __init__(self, n_primal):
        """
        Initalisation function for the primal network. The architecture is
        3-layer CNN with PReLU activations, the first two layers have 32
        channels, and the last layer has n_primal channels.

        Parameters
        ----------
        n_primal : int
            The number of primal channels in "history"
        """
        super(PrimalNet, self).__init__()

        self.n_primal = n_primal

        self.conv1 = nn.Conv2d(in_channels=n_primal + 1, out_channels=32,
                               kernel_size=(3, 3), padding=1)
        self.act1 = nn.PReLU(num_parameters=32, init=0.0)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32,
                               kernel_size=(3, 3), padding=1)
        self.act2 = nn.PReLU(num_parameters=32, init=0.0)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=n_primal,
                               kernel_size=(3, 3), padding=1)

        # Intialise the weights and biases
        self._init_weights()

    def _init_weights(self):
        """
        A custom initialisation function for the weights and biases of the
        network. The weights are initialised using the Xavier initialisation
        method, and the biases are initialised to zero.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Initialise the weights using the Xavier initialisation method
                nn.init.xavier_uniform_(m.weight)

                # Initialise the biases to zero
                nn.init.zeros_(m.bias)

    def forward(self, primal, adj_h1):
        """
        Forward pass for the primal network. The inputs are all current
        primals and the backprojection of the first dual variable. The
        output is the updated primal.

        Parameters
        ----------
        primal : torch.Tensor
            primal variable at the current iteration.
        adj_h1 : torch.Tensor
            Backprojection of first dual variable at the current
            iteration.

        Returns
        -------
        result : torch.Tensor
            Update for primal variable.
        """
       
        # Concatenate primal and f1
        input = torch.cat((primal, adj_h1), 1)

        # Pass through the network
        result = self.conv1(input)
        result = self.act1(result)
        result = self.conv2(result)
        result = self.act2(result)
        result = self.conv3(result)

        # Add the result to primal and return the updated primal
        return primal + result


class LearnedPrimal(nn.Module):
    """
    This class implements the Learned Primal algorithm, which is an alternative
    to the Learned Primal-Dual algorithm. It combines the primal network with
    a fixed update scheme for the dual variables.
    """

    def __init__(self, vg, pg, input_dimension=362, 
                 n_primal=5, n_iterations=10):
        """
        Initialisation function for the LearnedPrimal class. The class contains
        the forward and adjoint operators, as well as the primal networks.

        Parameters
        ----------
        vg : tomosipo.VolumeGeometry
            The volume geometry of the object being reconstructed.
        pg : tomosipo.ProjectionGeometry
            The projection geometry of the sinogram.
        input_dimension : int
            The size of the input image.
        n_primal : int
            The number of primal channels in "history".
        n_iterations : int
            The number of unrolled iterations to run the Learned Primal
            algorithm.
        """
        super(LearnedPrimal, self).__init__()

        self.input_dimension = input_dimension

        # Create projection geometries
        self.vg = vg
        self.pg = pg

        # Define the forward projector
        self.forward_projector = ts.operator(self.vg, self.pg)

        self.op = to_autograd(self.forward_projector, is_2d=True, num_extra_dims=2)
        self.adj_op = to_autograd(self.forward_projector.T, is_2d=True, num_extra_dims=2)


        # Store the primal nets in ModuleLists
        self.primal_list = nn.ModuleList([PrimalNet(n_primal)
                                          for _ in range(n_iterations)])

        # Create class attributes for other parameters
        self.n_primal = n_primal
        self.n_iterations = n_iterations

    def forward(self, sinogram):

        """
        Forward pass for the LearnedPrimal class. The input is the noisy,
        observed sinogram, and the output is the reconstructed image.

        Parameters
        ----------
        sinogram : torch.Tensor
            The observed noisy sinogram.
        
        Returns
        -------
        torch.Tensor
            The reconstructed image.
        """

        primal = torch.zeros(1, self.n_primal, self.input_dimension, self.input_dimension).cuda()

        for i in range(self.n_iterations):
            # Calculate the current residual
            dual = self.op(primal[:, 1:2, ...]) - sinogram.unsqueeze(1)
            
            adj = self.adj_op(dual)
            
            primal = self.primal_list[i].forward(primal, adj)

        return primal[:, 0:1, ...]
