"""
This script contains the implementation of the continuous primal-dual network
for the learned primal-dual algorithm, based on the paper "Continuous
Learned Primal Dual" by C. Runkel et al. (https://arxiv.org/abs/2405.02478v1)
"""

import torch.nn as nn
import torch
import tomosipo as ts
from ts_algorithms import fbp
from ts_algorithms.operators import operator_norm
from tomosipo.torch_support import to_autograd
from torchdiffeq import odeint_adjoint

class ODELayer(nn.Module):
    """
    This function defines the function used in the ODE block, which is a 2-layer
    CNN with PReLU activations and instance normalisations.
    """
    def __init__(self):
        """
        Initalisation function for the ODE layer. The architecture is a 2-layer
        CNN with PReLU activations.
        """
        super(ODELayer, self).__init__()
        
        self.odefunc = nn.Sequential(
            nn.InstanceNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32,
                      kernel_size=(3, 3), padding=1),
            nn.PReLU(num_parameters=32, init=0.0),
            nn.InstanceNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32,
                      kernel_size=(3, 3), padding=1),
            nn.PReLU(num_parameters=32, init=0.0),
        )
    
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
    
    def forward(self, t, x):
        """
        Forward pass for the ODE layer. The input is the current state of the
        system, and the output is the updated state of the system.

        Parameters
        ----------
        t : torch.Tensor
            Time variable. (Not used in this implementation, but needed as an
            input for the odeint_adjoint function.)
        x : torch.Tensor
            Current state of the system.
        
        Returns
        -------
        result : torch.Tensor
            Updated state of the system.
        """
        result = self.odefunc(x)
        return result

class ODEBlock(nn.Module):
    """
    This class contains the implementation of the ODE block, which is used to
    solve the ODE using the adjoint method.
    """
    def __init__(self, module, tol=1e-3):
        super(ODEBlock, self).__init__()
        self.module = module
        self.tol = tol

    def forward(self, x):
        """
        Forward pass for the ODE block. The input is the current state of the
        system, and the output is the updated state of the system.

        Parameters
        ----------
        x : torch.Tensor
            Current state of the system.

        Returns
        -------
        x : torch.Tensor
            Updated state of the system.
        """
        # Solve the ODE using the adjoint method
        t = torch.tensor([0, 1]).float().cuda()
        x = odeint_adjoint(self.module, x, t, rtol=self.tol, atol=self.tol,
                           method='rk4')
        return x[1]


class ContinuousDualNet(nn.Module):
    """
    This class implements the 'Dual' networks, which are used to update the
    dual variables in the continuous Learned Primal-Dual algorithm. The particular architecture
    used consists of convolution blocks to upsample and downsample channels for dual
    variables, as well as an ODE block to solve the ODE using the adjoint method.
    """

    def __init__(self, n_dual):
        """
        Initalisation function for the dual network. The architecture consists of
        convolution blocks to upsample and downsample channels for dual variables,
        as well as an ODE block to solve the ODE using the adjoint method.

        Parameters
        ----------
        n_dual : int
            The number of dual channels in "history"
        """
        super(ContinuousDualNet, self).__init__()

        self.n_dual = n_dual

        self.upconv = nn.Conv2d(in_channels=n_dual + 2, out_channels=32,
                               kernel_size=(3, 3), padding=1)

        self.odeblock = ODEBlock(ODELayer())
        self.downconv = nn.Conv2d(in_channels=32, out_channels=n_dual,
                                    kernel_size=1, padding=0)

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
        result = self.upconv(input)
        # result = self.act1(result)
        result = self.odeblock(result)
        result = self.downconv(result)

        # Add the result to dual and return the updated dual
        return dual + result


class ContinuousPrimalNet(nn.Module):
    """
    This class implements the 'Primal' networks, which are used to update the
    primal variables in the continuous Learned Primal-Dual algorithm. The particular architecture
    used consists of convolution blocks to upsample and downsample channels for primal
    variables, as well as an ODE block to solve the ODE using the adjoint method.
    """
    def __init__(self, n_primal):
        """
        Initalisation function for the primal network. The architecture consists of
        convolution blocks to upsample and downsample channels for primal variables,
        as well as an ODE block to solve the ODE using the adjoint method.

        Parameters
        ----------
        n_primal : int
            The number of primal channels in "history"
        """
        super(ContinuousPrimalNet, self).__init__()

        self.n_dual = n_primal

        self.upconv = nn.Conv2d(in_channels=n_primal + 1, out_channels=32,
                               kernel_size=(3, 3), padding=1)
        
        self.odeblock = ODEBlock(ODELayer())
        self.downconv = nn.Conv2d(in_channels=32, out_channels=n_primal,
                                    kernel_size=1, padding=0)

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
        result = self.upconv(input)
        result = self.odeblock(result)
        result = self.downconv(result)

        # Add the result to primal and return the updated primal
        return primal + result


class ContinuousPrimalDualNet(nn.Module):
    """
    This class implements the continuous primal-dual network for the learned
    primal-dual algorithm, combining the previously defined ContinuousPrimalNet
    and ContinuousDualNet classes to create the full reconstruction network.
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
        super(ContinuousPrimalDualNet, self).__init__()

        self.input_dimension = input_dimension

        # Create projection geometries
        self.vg = vg
        self.pg = pg

        # Define the forward projector
        self.forward_projector = ts.operator(self.vg, self.pg)

        self.op = to_autograd(self.forward_projector, is_2d=True, num_extra_dims=2)
        self.adj_op = to_autograd(self.forward_projector.T, is_2d=True, num_extra_dims=2)


        # Store the primal nets and dual nets in ModuleLists
        self.primal_list = nn.ModuleList([ContinuousPrimalNet(n_primal)
                                          for _ in range(n_iterations)])
        self.dual_list = nn.ModuleList([ContinuousDualNet(n_dual)
                                        for _ in range(n_iterations)])

        # Create class attributes for other parameters
        self.n_primal = n_primal
        self.n_dual = n_dual

        self.n_iterations = n_iterations

        self.step_size = torch.tensor(1.0 / operator_norm(self.forward_projector))

        self.lambda_dual = nn.ParameterList(
                    [
                        nn.Parameter(
                            torch.ones(1)
                            * 10 ** torch.log10(self.step_size)
                        )
                        for _ in range(self.n_iterations)
                    ]
                )
        self.lambda_primal = nn.ParameterList(
                    [
                        nn.Parameter(
                            torch.ones(1)
                            * 10 ** torch.log10(self.step_size)
                        )
                        for _ in range(self.n_iterations)
                    ]
                )

    def forward(self, sinogram):
        """
        Forward pass for the ContinousPrimalDualNet class. The input is the noisy,
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
        # Use fbp as the initial guess
        fbp_recon = fbp(self.forward_projector, sinogram)
        fbp_recon = torch.clip(fbp_recon, 0, 1)

        primal = torch.zeros(1, self.n_primal, self.input_dimension, self.input_dimension).cuda()

        # Use fbp as reconstruction for all primal variables
        for i in range(self.n_primal):
            primal[:, i, ...] = fbp_recon

        dual = torch.zeros(1, self.n_dual, height, width).cuda()

        for i in range(self.n_iterations):
            fp_f = self.lambda_primal[i] * self.op(primal[:, 1:2, ...])

            dual = self.dual_list[i].forward(dual, fp_f,
                                             sinogram.unsqueeze(1))

            adj_h = self.lambda_dual[i] * self.adj_op(dual[:, 0:1, ...])
            
            primal = self.primal_list[i].forward(primal, adj_h)

        return primal[:, 0:1, ...]
