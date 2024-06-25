"""
This module contains the Pytorch implementation of the Learned Primal Dual 
algorithm from the paper "Learned Primal-Dual Reconstruction" by Adler et al 
(https://arxiv.org/abs/1707.06474); the original implementation is in TensorFlow,
and can be found at https://github.com/adler-j/learned_primal_dual.
"""

import torch.nn as nn
import torch
import tomosipo as ts
from tomosipo.torch_support import to_autograd


class DualNet(nn.Module):
    """
    This class implements the 'Dual' networks, which are used to update the
    dual variables in the Learned Primal-Dual algorithm. The particular 
    architecture used is a 3-layer CNN with PReLU activations and residual
    connection at the end.
    """

    def __init__(self, n_dual):
        """
        Initalisation function for the dual network. The architecture is
        3-layer CNN with PReLU activations, the first two layers have 32
        channels, and the last layer has n_dual channels.

        Parameters
        ----------
        n_dual : int
            The number of dual channels in "history"
        """
        super(DualNet, self).__init__()

        self.n_dual = n_dual

        self.conv1 = nn.Conv2d(in_channels=n_dual + 2, out_channels=32,
                               kernel_size=(3, 3), padding=1)
        
        self.act1 = nn.PReLU(num_parameters=32, init=0.0)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32,
                               kernel_size=(3, 3), padding=1)

        self.act2 = nn.PReLU(num_parameters=32, init=0.0)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=n_dual,
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

    def forward(self, dual, f2, g):
        """
        Forward pass for the dual network. The inputs are all current
        duals, forward projection of the second primal variable, and 
        the sinogram. The output is the updated dual.

        Parameters
        ----------
        dual : torch.Tensor
            Dual variable at the current iteration.
        f2 : torch.Tensor
            Forward projection of second primal variable at the current
            iteration.
        g : torch.Tensor
            The observed noisy sinogram.

        Returns
        -------
        update : torch.Tensor
            Update for dual variable.
        """

        # Concatenate dual, f2 and g
        input = torch.cat((dual, f2, g), 1)

        # Pass through the network
        result = self.conv1(input)
        result = self.act1(result)
        result = self.conv2(result)
        result = self.act2(result)
        result = self.conv3(result)

        # Add the result to dual and return the updated dual
        return dual + result


class PrimalNet(nn.Module):
    """
    This class implements the 'Primal' networks, which are used to update the
    primal variables in the Learned Primal-Dual algorithm. The particular
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

        # Concatenate primal and adj_h1
        input = torch.cat((primal, adj_h1), 1)

        # Pass through the network
        result = self.conv1(input)
        result = self.act1(result)
        result = self.conv2(result)
        result = self.act2(result)
        result = self.conv3(result)

        # Add the result to primal and return the updated primal
        return primal + result


class PrimalDualNet(nn.Module):
    """
    This class implements the Learned Primal-Dual algorithm from the paper
    "Learned Primal-Dual Reconstruction" by Adler et al. It combines the
    previously defined PrimalNet and DualNet classes to create the full
    reconstruction network.
    """

    def __init__(self, vg, pg, input_dimension=362, 
                 n_primal=5, n_dual=5, n_iterations=10):
        """
        Initalisation function for the PrimalDualNet class. The class contains
        the forward and adjoint operators, as well as the primal and dual
        networks.

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
        n_dual : int
            The number of dual channels in "history".
        n_iterations : int
            The number of unrolled iterations to run the Learned Primal-Dual 
            algorithm.
        """
        super(PrimalDualNet, self).__init__()

        self.input_dimension = input_dimension

        # Create projection geometries
        self.vg = vg
        self.pg = pg

        # Define the forward projector
        self.forward_projector = ts.operator(self.vg, self.pg)

        self.op = to_autograd(self.forward_projector, is_2d=True, num_extra_dims=2)
        self.adj_op = to_autograd(self.forward_projector.T, is_2d=True, num_extra_dims=2)


        # Store the primal nets and dual nets in ModuleLists
        self.primal_list = nn.ModuleList([PrimalNet(n_primal)
                                          for _ in range(n_iterations)])
        self.dual_list = nn.ModuleList([DualNet(n_dual)
                                        for _ in range(n_iterations)])

        # Create class attributes for other parameters
        self.n_primal = n_primal
        self.n_dual = n_dual

        self.n_iterations = n_iterations

    def forward(self, sinogram):
        """
        Forward pass for the PrimalDualNet class. The input is the noisy,
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
        height, width = sinogram.shape[1:]

        # Initialise the primal and dual variables to zero
        primal = torch.zeros(1, self.n_primal, self.input_dimension, self.input_dimension).cuda()
        dual = torch.zeros(1, self.n_dual, height, width).cuda()

        for i in range(self.n_iterations):
            # Forward projection of the second primal variable
            fp_f = self.op(primal[:, 1:2, ...])

            # Update the dual variable
            dual = self.dual_list[i].forward(dual, fp_f,
                                             sinogram.unsqueeze(1))

            # Backproject the first dual variable
            adj_h = self.adj_op(dual[:, 0:1, ...])
            
            # Update the primal variable
            primal = self.primal_list[i].forward(primal, adj_h)

        return primal[:, 0:1, ...]
