import torch.nn as nn
import torch
import tomosipo as ts


class DualNet(nn.Module):
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

        # Permute the input to match the input shape of the dual network
        # input = input.permute(0, 3, 1, 2)

        # Pass through the network
        result = self.conv1(input)
        result = self.act1(result)
        result = self.conv2(result)
        result = self.act2(result)
        result = self.conv3(result)

        # Permute the result to match the output shape of the dual network
        # result = result.permute(0, 2, 3, 1)

        # Add the result to dual and return the updated dual
        return dual + result


class PrimalNet(nn.Module):
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

        # Permute the input to match the input shape of the primal network
        # input = input.permute(0, 3, 1, 2)

        # Pass through the network
        result = self.conv1(input)
        result = self.act1(result)
        result = self.conv2(result)
        result = self.act2(result)
        result = self.conv3(result)

        # Permute the result to match the output shape of the primal network
        # result = result.permute(0, 2, 3, 1)

        # Add the result to primal and return the updated primal
        return primal + result


class PrimalDualNet(nn.Module):

    def __init__(self, input_dimension=362, n_detectors=513,
                 n_angles=1000, n_primal=5, n_dual=5, n_iterations=10):
        super(PrimalDualNet, self).__init__()

        # Create projection geometries
        self.vg = ts.volume(shape=(1, input_dimension, input_dimension))
        self.pg = ts.parallel(angles=n_angles, shape=(1, n_detectors))

        # Define the forward projector
        self.forward_projector = ts.operator(self.vg, self.pg)

        # Store the primal nets and dual nets in ModuleLists
        self.primal_list = nn.ModuleList([PrimalNet(n_primal)
                                          for _ in range(n_iterations)])
        self.dual_list = nn.ModuleList([DualNet(n_dual)
                                        for _ in range(n_iterations)])

        # Create class attributes for other parameters
        self.n_primal = n_primal
        self.n_dual = n_dual

        self.n_iterations = n_iterations

        self.opnorm = 349530.28125


    def operator_norm(A, num_iter=10):
        x = torch.randn(A.domain_shape)
        for i in range(num_iter):
            x = A.T(A(x))
            x /= torch.norm(x) # L2 vector-norm
        return (torch.norm(A.T(A(x))) / torch.norm(x)).item()

    def forward(self, sinogram):
        # Initialise the primal and dual variables

        primal = torch.zeros((1, self.n_primal, 362, 362))
        dual = torch.zeros((1, self.n_dual, 1000, 513))

        # mu_water = 0.02

        # Feed forward 10 times
        for i in range(self.n_iterations):
            # print(dual)
            # Pass through the primal network; first forward project the primal
            # fp_f = torch.exp(-mu_water *
            #                  self.forward_projector(primal[:, 1:2, ...].squeeze(1)))

            fp_f = self.forward_projector(primal[:, 1:2, ...].squeeze(1)) / self.opnorm

            # TODO: Need to add exponential transform
            dual = self.dual_list[i].forward(dual, fp_f.unsqueeze(1),
                                             sinogram.unsqueeze(1) / self.opnorm)

            # Pass through the dual network; backward project the product of
            # the first primal and the first dual

            # # Forward project the first primal
            # input = primal[:, 0:1, ...].squeeze(1)
            # input = -mu_water * torch.exp(-mu_water * self.forward_projector(input))

            # input *= dual[:, 0:1, ...].squeeze(1)
            # adj_h = self.forward_projector.T(input)

            # adj_h = self.forward_projector.T(dual[:, 0:1, ...].squeeze(1))

            adj_h = self.forward_projector.T(dual[:, 0:1, ...].squeeze(1)) / self.opnorm

            # print(adj_h)

            primal = self.primal_list[i].forward(primal,
                                                 adj_h.unsqueeze(1))

        return primal[:, 0:1, ...]
