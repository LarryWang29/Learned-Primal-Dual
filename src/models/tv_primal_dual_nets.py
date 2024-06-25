"""
This module contains the implementation of Total Variation Learned Primal
Dual (TVLPD), a custom variant of the Learned Primal-Dual inspired by the 
Chambolle-Pock algorithm for Total Variation (exact algorithm can be found at
https://iopscience.iop.org/article/10.1088/0031-9155/57/10/3065).
"""

import torch.nn as nn
import torch
import tomosipo as ts
from tomosipo.torch_support import to_autograd
from ts_algorithms.tv_min import grad_2D, grad_2D_T


class DualNet(nn.Module):
    """
    This class implements the 'Dual' networks, which is used to update the
    first component of the dual variables in the Total Variation Learned Primal-Dual 
    algorithm. The particular architecture used is a 3-layer CNN with PReLU 
    activations and residual connection at the end.
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
        Forward pass for the dual network used to update the first dual component. 
        The inputs are the first components of all current duals, forward projection 
        of the second primal variable, and the sinogram. The output are the
        updated first components of the duals.

        Parameters
        ----------
        dual : torch.Tensor
            First components of dual variables at the current iteration.
        f2 : torch.Tensor
            Forward projection of second primal variable at the current
            iteration.
        g : torch.Tensor
            The observed noisy sinogram.

        Returns
        -------
        update : torch.Tensor
            Update for the first components of the dual variables.
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


class DivDualNet(nn.Module):
    """
    This class implements the 'DivDual' network, which is used to update the
    second component of the dual variables in the Total Variation Learned Primal-Dual
    algorithm. It's called DivDual because the second component of the dual variables
    is related to the div of the primal variables. The particular architecture used is a
    3-layer CNN with PReLU activations and residual connection at the end.
    """

    def __init__(self):
        """
        Initalisation function for the DivDual network. The architecture is
        3-layer CNN with PReLU activations, the first two layers have 32
        channels, and the last layer has n_dual channels.
        """
        super(DivDualNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32,
                               kernel_size=(3, 3), padding=1)

        self.act1 = nn.PReLU(num_parameters=32, init=0.0)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32,
                               kernel_size=(3, 3), padding=1)

        self.act2 = nn.PReLU(num_parameters=32, init=0.0)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=2,
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

    def forward(self, div_duals, div):
        """
        Forward pass for the DivDual network used to update the second dual component.
        The inputs are the second components of all current duals and the divergence
        of the second primal variable. The output are the updated second components of
        the duals.

        Parameters
        ----------
        div_duals : torch.Tensor
            Second components of dual variables at the current iteration.
        div : torch.Tensor
            Divergence of the second primal variable at the current iteration.

        Returns
        -------
        update : torch.Tensor
            Update for dual variable.
        """

        # Concatenate all inputs
        input = torch.cat((div_duals, div), 1)

        # Pass through the network
        result = self.conv1(input)
        result = self.act1(result)
        result = self.conv2(result)
        result = self.act2(result)
        result = self.conv3(result)

        # Return the updated div_duals
        return div_duals + result

class PrimalNet(nn.Module):
    """
    This class implements one of the 'Primal' networks, which is used to update the
    primal variables in the Total Variation Learned Primal-Dual algorithm. The particular
    architecture used is a 3-layer CNN with PReLU activations and residual connection at
    the end.
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

        self.conv1 = nn.Conv2d(in_channels=n_primal + 2, out_channels=32,
                               kernel_size=(3, 3), padding=1)
        self.act1 = nn.PReLU(num_parameters=32, init=0.0)
        # self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32,
                               kernel_size=(3, 3), padding=1)
        self.act2 = nn.PReLU(num_parameters=32, init=0.0)
        # self.act2 = nn.ReLU()
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

    def forward(self, primal, div_T, adj_h1):
        """
        Forward pass for the primal network. The inputs are all current
        primals, the backprojection of the first dual variable, and the
        Divergence Transpose of the second component of the dual variable. 
        The output is the updated primal.


        Parameters
        ----------
        primal : torch.Tensor
            primal variable at the current iteration.
        div_T : torch.Tensor
            Divergence Transpose of the second component of the dual variable.
        adj_h1 : torch.Tensor
            Backprojection of first dual variable at the current
            iteration.

        Returns
        -------
        result : torch.Tensor
            Update for primal variable.
        """

        # Concatenate all inputs
        input = torch.cat((primal, div_T, adj_h1), 1)

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
    This class implements the Total Variation Learned Primal-Dual (TVLPD) algorithm.
    It combines the PrimalNet, DualNet, and DivDualNet classes to implement the
    full reconstruction network.
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

        # Store the sub-networks in ModuleLists
        self.primal_list = nn.ModuleList([PrimalNet(n_primal)
                                          for _ in range(n_iterations)])
        self.dual_list = nn.ModuleList([DualNet(n_dual)
                                        for _ in range(n_iterations)])
        self.div_list = nn.ModuleList([DivDualNet()
                                        for _ in range(n_iterations)])

        # Create class attributes for other parameters
        self.n_primal = n_primal
        self.n_dual = n_dual

        self.n_iterations = n_iterations

    def forward(self, sinogram):
        """
        Forward pass for the Total Variation Learned Primal-Dual algorithm.
        The input is the noisy sinogram, and the output is the reconstructed
        image.

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
        # Using 1 as the batch size
        primal = torch.zeros(1, self.n_primal, self.input_dimension, self.input_dimension).cuda()
        dual = torch.zeros(1, self.n_dual, height, width).cuda()
        div_dual = torch.zeros(1, 2, self.input_dimension, self.input_dimension).cuda()

        for i in range(self.n_iterations):
            # Forward projection of the second primal variable
            fp_f = self.op(primal[:, 1:2, ...])

            # Div of the second primal variable
            div = grad_2D(primal[0, 1:2, ...])

            # Forward pass for the first component of the dual variable
            dual = self.dual_list[i].forward(dual, fp_f,
                                             sinogram.unsqueeze(1))
            
            # Forward pass for the second component of the dual variable
            div_dual = self.div_list[i].forward(div_dual, div)

            # Backproject the first dual variable
            adj_h = self.adj_op(dual[:, 0:1, ...])

            # Compute the divergence transpose of the second component 
            # of the dual variable
            div_T = grad_2D_T(div_dual)

            div_T = div_T.unsqueeze(1)
            
            # Update the primal variable
            primal = self.primal_list[i].forward(primal, div_T, adj_h)

        return primal[:, 0:1, ...]
