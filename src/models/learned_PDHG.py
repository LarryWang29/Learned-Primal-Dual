"""
This module contains the implementation of the Learned Primal-Dual Hybrid Gradient
algorithm, which is one of the alternative models suggested by Adler et al. in their
paper "Learned Primal-Dual Reconstruction" (https://arxiv.org/abs/1707.06474). It replaces
proximal operators with learned neural networks. However, compared to Learned Primal-Dual,
it has more structural constraints.
"""

import torch.nn as nn
import torch
import tomosipo as ts
from tomosipo.torch_support import to_autograd


class DualNet(nn.Module):
    """
    This class implements the 'Dual' networks, which are used to update the
    dual variables in the Learned PDHG algorithm. The particular 
    architecture used is a 3-layer CNN with PReLU activations and residual
    connection at the end.
    """

    def __init__(self):
        """
        Initalisation function for the dual network. The architecture is
        3-layer CNN with PReLU activations, the first two layers have 32
        channels, and the last layer has n_dual channels.
        """
        super(DualNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=2, out_channels=32,
                               kernel_size=(3, 3), padding=1)

        self.act1 = nn.PReLU(num_parameters=32, init=0.0)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32,
                               kernel_size=(3, 3), padding=1)
        self.act2 = nn.PReLU(num_parameters=32, init=0.0)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=1,
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

    def forward(self, dual, g):
        """
        Forward pass for the dual network. The inputs are the current dual
        variable and the observed sinogram. The output is the updated dual.

        Parameters
        ----------
        dual : torch.Tensor
            Dual variable at the current iteration.
        g : torch.Tensor
            Observed sinogram.

        Returns
        -------
        update : torch.Tensor
            Update for dual variable.
        """

        # Concatenate dual, g
        input = torch.cat((dual, g), 1)

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
    primal variables in the Learned PDHG algorithm. The particular
    architecture used is a 3-layer CNN with PReLU activations and residual
    connection at the end.
    """
    def __init__(self):
        """
        Initalisation function for the primal network. The architecture is
        3-layer CNN with PReLU activations, the first two layers have 32
        channels, and the last layer has 1 channel.
        """
        super(PrimalNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32,
                               kernel_size=(3, 3), padding=1)
        self.act1 = nn.PReLU(num_parameters=32, init=0.0)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32,
                               kernel_size=(3, 3), padding=1)
        self.act2 = nn.PReLU(num_parameters=32, init=0.0)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=1,
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

    def forward(self, primal):
        """
        Forward pass for the primal network. The inputs are all current
        primals and the first primal variable. The outputs are the updated
        primal and the update.

        Parameters
        ----------
        primal : torch.Tensor
            Primal variable at the current iteration.

        Returns
        -------
        result : torch.Tensor
            Update for primal variable.
        Updated primal : torch.Tensor
            Original primal variable plus the update.
        """

        # Pass through the network
        result = self.conv1(primal)
        result = self.act1(result)
        result = self.conv2(result)
        result = self.act2(result)
        result = self.conv3(result)

        # Add the result to primal and return the updated primal
        return primal + result, result


class PrimalDualNet(nn.Module):
    """
    This class implements the Learned Primal-Dual Hybrid Gradient algorithm,
    which is one of the alternative models suggested by Adler et al. It combines
    the previous PrimalNet and DualNet classes into a single class.
    """
    def __init__(self, vg, pg, input_dimension=362, 
                 n_iterations=10, sigma=0.5, tau=0.5, theta=1):
        """
        Initalisation function for the Learned Primal-Dual Hybrid Gradient
        algorithm. The class contains the forward projector, primal and dual
        networks, and the hyperparameters sigma, tau, and theta.
        
        Parameters
        ----------
        vg : tomosipo.VolumeGeometry
            The volume geometry of the reconstruction volume.
        pg : tomosipo.ProjectionGeometry
            The projection geometry of the sinogram.
        input_dimension : int
            The dimension of the input images.
        n_iterations : int
            The number of iterations to run the algorithm.
        sigma : float
            Hyperparameter for update stepsize of dual variable.
        tau : float
            Hyperparameter for update stepsize of primal variable.
        theta : float
            Hyperparameter for update stepsie of the average primal variable.
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
        self.primal_net = PrimalNet()
        self.dual_net = DualNet()

        self.theta = theta
        self.tau = tau
        self.sigma = sigma

        self.n_iterations = n_iterations

    def forward(self, sinogram):
        # Initialise the primal and dual variables

        height, width = sinogram.shape[1:]
        # Using 1 as the batch size
        primal = torch.zeros(1, 1, self.input_dimension, self.input_dimension).cuda()
        primal_avg = torch.zeros(1, 1, self.input_dimension, self.input_dimension).cuda()
        dual = torch.zeros(1, 1, height, width).cuda()

        for _ in range(self.n_iterations):
            dual = self.dual_net(dual + self.sigma * self.op(primal_avg), sinogram.unsqueeze(1))
            primal, update = self.primal_net(primal - self.tau * self.adj_op(dual))
            primal_avg = primal + self.theta * update

        return primal
