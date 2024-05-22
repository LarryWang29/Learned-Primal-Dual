import torch.nn as nn
import torch
import tomosipo as ts
from tomosipo.torch_support import to_autograd
from torchdiffeq import odeint_adjoint

class ODELayer(nn.Module):
    def __init__(self):
        super(ODELayer, self).__init__()
        
        self.odefunc = nn.Sequential(
            # nn.InstanceNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32,
                      kernel_size=(3, 3), padding=1),
            nn.PReLU(num_parameters=32, init=0.0),
            # nn.InstanceNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32,
                      kernel_size=(3, 3), padding=1),
            nn.PReLU(num_parameters=32, init=0.0),
        )
    
        # Intialise the weights and biases
        self._init_weights()

    def _init_weights(self):
        """
        Initialises the weights of the network using the Xavier initialisation
        method.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Initialise the weights using the Xavier initialisation method
                nn.init.xavier_uniform_(m.weight)

                # Initialise the biases to zero
                nn.init.zeros_(m.bias)
    
    def forward(self, t, x):
        return self.odefunc(x)

class ODEBlock(nn.Module):
    """
    Implementation of neural ordinary differential equation, in place of the
    original convolutions.
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
    Implementation of the dual network, using a 3-layer CNN.
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
        super(ContinuousDualNet, self).__init__()

        self.n_dual = n_dual

        self.upconv = nn.Conv2d(in_channels=n_dual + 2, out_channels=32,
                               kernel_size=(3, 3), padding=1)
        
        self.act1 = nn.PReLU(num_parameters=32, init=0.0)

        self.odeblock = ODEBlock(ODELayer())
        self.downconv = nn.Conv2d(in_channels=32, out_channels=n_dual,
                                    kernel_size=(3, 3), padding=1)

        # Intialise the weights and biases
        self._init_weights()

    def _init_weights(self):
        """
        Initialises the weights of the network using the Xavier initialisation
        method.
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
        duals, second dual variable, and the sinogram. The output is the
        updated dual.

        Parameters
        ----------
        dual : torch.Tensor
            Primal variable at the current iteration.
        f2 : torch.Tensor
            Forward projection of second dual variable at the current
            iteration.
        g : torch.Tensor
            Observed sinogram.

        Returns
        -------
        update : torch.Tensor
            Update for dual variable.
        """

        # Concatenate dual, f2 and g
        input = torch.cat((dual, f2, g), 1)

        # Pass through the network
        result = self.upconv(input)
        result = self.act1(result)
        result = self.odeblock(result)
        result = self.downconv(result)

        # Add the result to dual and return the updated dual
        return dual + result


class ContinuousPrimalNet(nn.Module):
    """
    Implementation of the primal network, using a 3-layer CNN.
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
        super(ContinuousPrimalNet, self).__init__()

        self.n_dual = n_primal

        self.upconv = nn.Conv2d(in_channels=n_primal + 1, out_channels=32,
                               kernel_size=(3, 3), padding=1)
        
        self.act1 = nn.PReLU(num_parameters=32, init=0.0)

        self.odeblock = ODEBlock(ODELayer())
        self.downconv = nn.Conv2d(in_channels=32, out_channels=n_primal,
                                    kernel_size=(3, 3), padding=1)

        # Intialise the weights and biases
        self._init_weights()

    def _init_weights(self):
        """
        Initialises the weights of the network using the Xavier initialisation
        method.
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
        primals and the first primal variable. The output is the updated primal.

        Parameters
        ----------
        primal : torch.Tensor
            Dual variable at the current iteration.
        fp_f1 : torch.Tensor
            Forward projection of the first primal variable at the current
            iteration.
        adj_h1 : torch.Tensor
            Adjoint projection of first primal variable at the current
            iteration.

        Returns
        -------
        result : torch.Tensor
            Update for primal variable.
        """

        # Concatenate primal and f1
        input = torch.cat((primal, adj_h1), 1)

        # Pass through the network
        result = self.upconv(input)
        result = self.act1(result)
        result = self.odeblock(result)
        result = self.downconv(result)

        # Add the result to primal and return the updated primal
        return primal + result


class ContinuousPrimalDualNet(nn.Module):

    def __init__(self, vg, pg, input_dimension=362, 
                 n_primal=5, n_dual=5, n_iterations=10):
        super(ContinuousPrimalDualNet, self).__init__()

        self.input_dimension = input_dimension

        # Create projection geometries
        self.vg = vg
        self.pg = pg

        # Define the forward projector
        self.forward_projector = ts.operator(self.vg, self.pg)
        # self.forward_projector = ts.operator(self.vg[:1], self.pg.to_vec()[:, :1, :])

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

    def forward(self, sinogram):
        # Initialise the primal and dual variables

        height, width = sinogram.shape[1:]
        # Using 1 as the batch size
        primal = torch.zeros(1, self.n_primal, self.input_dimension, self.input_dimension).cuda()
        dual = torch.zeros(1, self.n_dual, height, width).cuda()

        for i in range(self.n_iterations):
            fp_f = self.op(primal[:, 1:2, ...])

            dual = self.dual_list[i].forward(dual, fp_f,
                                             sinogram.unsqueeze(1))

            adj_h = self.adj_op(dual[:, 0:1, ...])
            
            primal = self.primal_list[i].forward(primal, adj_h)

        return primal[:, 0:1, ...]
