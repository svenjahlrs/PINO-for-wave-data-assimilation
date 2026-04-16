
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
torch.manual_seed(0)
np.random.seed(0)


class SpectralConv2d(nn.Module):
    """
    2D Fourier spectral convolution layer performs a convolution in Fourier space by:
    (1) transforming the input to the spectral domain
    (2) applying learned complex-valued weights to a subset of low-frequency modes
    (3) transforming the result back to physical space
    This enables global receptive fields with fewer parameters compared to standard spatial convolutions.
    """
    def __init__(self, in_channels, out_channels, modes1, modes2):
        """"
        :param in_channels: number of input feature channels.
        :param out_channels: number of output feature channels.
        :param modes1: number of retained Fourier modes in spatial (x) direction
        :param modes2: number of retained Fourier modes in temporal (t) direction
        """
        super(SpectralConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        # --- scaling factor for stable initialization of complex weights ---
        scale = (1 / (in_channels * out_channels))

        # --- learnable complex weights for positive and negative frequency bands ---
        self.weights1 = nn.Parameter(scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))

    def compl_mul2d(self, input, weights):
        """
        Complex multiplication in Fourier domain.
        :param input: input tensor of shape (batch_size, width, modes1, modes2) in Fourier space.
        :param weights: complex weight tensor of shape (width, width, modes1, modes2)
        :return comp: tensor of shape (batch, width, modes1, modes2) after weighted multiplication.
        """

        # --- Einstein summation: channel-wise complex multiplication ---
        comp = torch.einsum("bixy,ioxy->boxy", input, weights)
        return comp

    def forward(self, x):
        """
        Forward pass of the SpectralConv2d layer.
        :param x: input tensor of SpecConv layer in physical space of shape (batch_size, width, nx, nt) in spatial domain
        :return out: output tensor of SpcConv layer in physical space of shape (batch_size, width, nx, nt) in spatial domain
        """

        batchsize = x.shape[0]

        # --- transform input to Fourier domain ---
        x_ft = torch.fft.rfft2(x)/(len(x)**2)

        # --- allocate output tensor in Fourier space ---
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)

        # --- apply learned weights to low-frequency modes (positive frequencies) ---
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # --- inverse Fourier transform back to physical space ---
        out = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))*(len(x)**2)

        return out


class FNO1d_x_to_2d(nn.Module):
    """
    Fourier Neural Operator (FNO) model that maps wave buoy input to 2D output.

    This model approximates high-dimensional operators using multiple Fourier
    convolutional layers with nonlinear activation and fully connected layers
    to predict outputs on a spatial grid.

    Attributes:
        modes1, modes2: Number of Fourier modes in two spatial dimensions.
        width: Width of the latent feature space (number of channels).
        num_layers: Number of Fourier convolutional layers.
        nx: Number of desired spatial points in the x-direction.
        padding_x, padding_t: Padding added to inputs to address non-periodicity.
        fc_interp: Linear layer interpolating input channel time series to spatial grid.
        fc0, fc1, fc2, fc3: Fully connected layers that map latent features to output.
        activation: Nonlinear activation function (GELU).
        sp_convs: List of spectral convolution layers.
        ws: List of 1x1 convolution layers for residual connections.
    """
    def __init__(self, modes1, modes2, width, in_channels, nx, num_layers, pad =12):
        super(FNO1d_x_to_2d, self).__init__()

        # --- padding to mitigate non-periodicity artifacts ---
        self.padding_x = pad
        self.padding_t = pad

        # --- define layers ---
        self.fc_interp = nn.Linear(in_channels, int(nx))  # interpolation: num_boys to spatial grid (nx)
        self.fc0 = nn.Linear(1, width)                    # lifting layer to higher-dimensional latent representation

        self.sp_convs = nn.ModuleList([SpectralConv2d(width, width, modes1, modes2) for _ in range(num_layers)])   # upper FNO-path: stacked spectral convolution layers

        self.ws = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(num_layers)])                           # lower FNO-path: (1x1) convolutions

        self.fc1 = nn.Linear(width, 128)                  # projection layers from latent representation back to scalar field
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 1)

        self.activation = F.gelu

    def forward(self, x):
        """
        Forward pass through the FNO model
        :param x: input measurement tensor of shape (batch_size, num_buoys, nt)
        :return: reconstructed wave field output tensor of shape (batch_size, nx, nt)
        """

        # --- lift to Fourier layers feature space ---
        x = x.unsqueeze(-1)             # reshape to (batch_size, n_buoys, nt, 1)
        x = x.permute(0, 3, 2, 1)       # reshape to (batch_size, 1, nt, n_buoy)
        x = self.fc_interp(x)           # output shape (batch_size, 1, nt, nx)
        x = x.permute(0, 3, 2, 1)       # reshape to (batch_size, nx, nt, 1)

        x = self.fc0(x)                 # output shape (batch_size, nt, nx, width)
        x = x.permute(0, 3, 1, 2)       # reshape to (batch_size, width, nx, nt)
        x = F.pad(x, [0, self.padding_t, 0, self.padding_x])  # (batch_size, width, nx+pad, nt+pad)

        # --- iterative Fourier layers ---
        for i, (speconv, w) in enumerate(zip(self.sp_convs, self.ws)):
            x1 = speconv(x)             # upper path in Fourier layer
            x2 = w(x)                   # lower path
            x = x1 + x2
            x = self.activation(x)

        # --- project back to physical scalar field ---
        x = x[..., :-self.padding_x, :-self.padding_t] # (batch_size, width, nx, nt)
        x = x.permute(0, 2, 3, 1)       # reshape to (batch_size, nt, nx, width)
        x = self.fc1(x)                 # output shape (batch_size, nt, nx, 128)
        x = self.fc2(x)                 # output shape (batch_size, nt, nx, 32)
        x = self.fc3(x)                 # output shape (batch_size, nt, nx, 1)
        x = x.squeeze(-1)               # reshape to (batch_size, nt, nx)
        return x

