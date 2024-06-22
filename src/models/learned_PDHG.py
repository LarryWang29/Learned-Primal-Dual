import torch.nn as nn
import torch
import tomosipo as ts
from tomosipo.torch_support import to_autograd


class DualNet(nn.Module):
    """
    Implementation of the dual network, using a 3-layer CNN.
    """

    def __init__(self):
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

        self.conv1 = nn.Conv2d(in_channels=2, out_channels=32,
                               kernel_size=(3, 3), padding=1)
        
        # TODO: ReLU seems to promote more nonnegativity... Might switch
        # back to PReLU later

        self.act1 = nn.PReLU(num_parameters=32, init=0.0)
        # self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32,
                               kernel_size=(3, 3), padding=1)
        self.act2 = nn.PReLU(num_parameters=32, init=0.0)
        # self.act2 = nn.ReLU()
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=1,
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

    def forward(self, dual, g):
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
    Implementation of the primal network, using a 3-layer CNN.
    """
    def __init__(self):
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

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32,
                               kernel_size=(3, 3), padding=1)
        self.act1 = nn.PReLU(num_parameters=32, init=0.0)
        # self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32,
                               kernel_size=(3, 3), padding=1)
        self.act2 = nn.PReLU(num_parameters=32, init=0.0)
        # self.act2 = nn.ReLU()
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=1,
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

    def forward(self, primal):
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

        # Pass through the network
        result = self.conv1(primal)
        result = self.act1(result)
        result = self.conv2(result)
        result = self.act2(result)
        result = self.conv3(result)

        # Add the result to primal and return the updated primal
        return primal + result, result


class PrimalDualNet(nn.Module):

    def __init__(self, vg, pg, input_dimension=362, 
                 n_iterations=10, sigma=1, tau=0.5, theta=0.5):
        super(PrimalDualNet, self).__init__()

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
