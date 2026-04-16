import numpy as np
import torch
from torch import nn


def integral_trapz(y):
    """
    Approximates a definite integral using the trapezoidal rule (1D). It assumes uniform grid spacing and computes the
    integral as a weighted sum of function values.
    :param y: discrete function values sampled on a uniform grid.
    :return: approximation of the integral over the domain
    """
    return (y[0] + y[-1]) / 2.0 + np.sum(y[1:-1])


def integral_trapz_2d(y):
    """
    Approximates a 2D integral using a separable trapezoidal rule.
    The integral is computed by weighting corner points (1/4), edge points (1/2), interior points (1)
    :param y: 2D array representing function values on a regular grid.
    :return: approximation of the surface integral
    """

    tr1 = (y[0, 0] + y[0, -1] + y[-1, 0] + y[-1, -1]) / 4.0      # corner contributions
    tr2 = (np.sum(y[0, 1:-1])                                    # edge contributions
           + np.sum(y[-1, 1:-1])
           + np.sum(y[1:-1, 0])
           + np.sum(y[1:-1, -1])) / 2.0
    tr3 = np.sum(y[1:-1, 1:-1])                                  # interior contributions

    return tr1 + tr2 + tr3


def SSP_metric(y_true, y_pred):
    """
    Computes the Surface Similarity Parameter (Perlin  & Bustamante2015) between two 1D numpy surfaces
    :param y_true: true field (surface elevation or potential) of shape (nt,)
    :param y_pred: predicted field (surface elevation or potential) of shape (nt,)
    :return: SSP value in [0, 1], whereas 0 represents perfect agreement between y_true and y_pred
    """

    # --- Fourier transform of both signals ---
    spec1 = np.fft.fft(y_true)
    spec2 = np.fft.fft(y_pred)

    # --- normalized error ---
    nominator = np.sqrt(integral_trapz(np.square(np.abs(spec1 - spec2))))
    denominator = np.sqrt(integral_trapz(np.square(np.abs(spec1)))) + np.sqrt(integral_trapz(np.square(np.abs(spec2))))

    return np.divide(nominator, denominator)


def SSP_2D_metric(y_true, y_pred):
    """
    This method returns the Surface Similarity parameter for 2D numpy surfaces
    :param y_true: true field (surface elevation or potential) of shape (nx_, nt_)
    :param y_pred: predicted field (surface elevation or potential) of shape (nx_, nt_)
    :return: SSP value in [0, 1], whereas 0 represents perfect agreement between y_true and y_pred
    """

    # --- Fourier transform of both signals ---
    spec1 = np.fft.fft2(y_true)
    spec2 = np.fft.fft2(y_pred)

    # --- normalized error ---
    nominator = np.sqrt(integral_trapz_2d(np.square(np.abs(spec1 - spec2))))
    denominator = np.sqrt(integral_trapz_2d(np.square(np.abs(spec1)))) + np.sqrt(
        integral_trapz_2d(np.square(np.abs(spec2))))

    return np.divide(nominator, denominator)


class SurfaceSimilarityParameter2D(nn.Module):
    """
    PyTorch implementation of the Surface Similarity Parameter (SSP) allowing integration into PINO training or evaluation pipelines.
    """
    def __init__(self):
        super().__init__()

    def sobolev(self, y_f) :
        """
        Computes spectral L2 norm via trapezoidal integration.
        :param y_f: Fourier-transformed field of shape (batch_size, nx_, nt_)
        :return: norm per sample of shape (batch_size, )
        """

        y_f = torch.square(torch.abs(y_f))

        # --- integrate over spatial and temporal dimension explicitly ---
        for _ in range(2):
            y_f = torch.trapz(y_f)

        return torch.sqrt(y_f)

    def forward(self, y_true, y_pred):
        """
        :param y_true: true field (surface elevation or potential) of shape (batch_size, nx_, nt_)
        :param y_pred: predicted field (surface elevation or potential) of shape (batch_size, nx_, nt_)
        :return: SSP values of shape (batch_size, ) for each sample. SSP in [0, 1], whereas 0 represents perfect agreement between y_true and y_pred
        """

        # --- Fourier transform of both signals ---
        y_true_f = torch.fft.fft2(y_true)
        y_pred_f = torch.fft.fft2(y_pred)

        # --- normalized error ---
        SSP_per_samp = torch.divide(self.sobolev(torch.subtract(y_true_f, y_pred_f)),
                                    torch.add(self.sobolev(y_true_f), self.sobolev(y_pred_f)))

        return SSP_per_samp
