import torch.nn as nn
import torch
import tomosipo as ts


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
        super().__init__()

        self.n_primal = n_primal

        self.conv1 = nn.Conv2d(in_channels=n_primal + 2, out_channels=32,
                               kernel_size=(3, 3), padding=1)
        self.act1 = nn.PReLU()
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32,
                               kernel_size=(3, 3), padding=1)
        self.act2 = nn.PReLU()
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=n_primal,
                               kernel_size=(3, 3), padding=1)

    def forward(self, primal, f2, g):
        """
        Forward pass for the primal network. The inputs are all current
        primals, second dual variable, and the sinogram. The output is the
        updated primal.

        Parameters
        ----------
        primal : torch.Tensor
            Primal variable at the current iteration.
        f2 : torch.Tensor
            Forward projection of second dual variable at the current
            iteration.
        g : torch.Tensor
            Observed sinogram.

        Returns
        -------
        update : torch.Tensor
            Update for primal variable.
        """

        # Concatenate primal, f2 and g
        input = torch.cat((primal, f2, g), 0)

        # Pass through the network
        result = self.conv1(input)
        result = self.act1(result)
        result = self.conv2(result)
        result = self.act2(result)
        result = self.conv3(result)

        # Add the result to primal and return the updated primal
        return result


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
        super().__init__()

        self.n_dual = n_dual

        self.conv1 = nn.Conv2d(in_channels=n_dual + 1, out_channels=32,
                               kernel_size=(3, 3), padding=1)
        self.act1 = nn.PReLU()
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32,
                               kernel_size=(3, 3), padding=1)
        self.act2 = nn.PReLU()
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=n_dual,
                               kernel_size=(3, 3), padding=1)

    def forward(self, dual, fp_f1, adj_h1):
        """
        Forward pass for the dual network. The inputs are all current
        duals and the first dual variable. The output is the updated dual.

        Parameters
        ----------
        dual : torch.Tensor
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
            Update for dual variable.
        """

        # Concatenate dual and f1
        input = torch.cat((dual, fp_f1 * adj_h1), 0)

        # Pass through the network
        result = self.conv1(input)
        result = self.act1(result)
        result = self.conv2(result)
        result = self.act2(result)
        result = self.conv3(result)

        # Add the result to dual and return the updated dual
        return result


class PrimalDualNet(nn.Module):

    def __init__(self, sinogram, input_dimension=362, n_detectors=513,
                 n_angles=1000, n_primal=5, n_dual=5, n_iterations=10):
        super().__init__()

        self.sinogram = sinogram

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

        self.n_iterations = n_iterations

    def forward(self):
        # Initialise the primal and dual variables
        # TODO: Change the dimensions of the primal and dual variables to class
        #       attributes

        primal = torch.zeros((1, 1, 362, 362))
        dual = torch.zeros((1, 1, 513, 1000))

        # Feed forward 10 times
        for i in range(self.n_iterations):
            # Pass through the primal network; first forward project the primal
            fp_f = self.forward_projector(primal[..., 1:2])

            # TODO: Need to add exponential transform

            dual_update = self.primal_list[i].forward(dual, fp_f,
                                                      self.sinogram)
            dual += dual_update

            # Pass through the dual network; backward project the product of
            # the first primal and the first dual
            adj_h = self.forward_projector.T(primal[..., 0:1] *
                                             dual[..., 0:1])
            primal_update = self.dual_list[i].forward(primal, adj_h)
            primal += primal_update

        return primal[..., 0:1]


model = PrimalDualNet(0)
for name, param in model.named_parameters():
    print(name, param.size())
