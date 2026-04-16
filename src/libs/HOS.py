import torch
torch.set_default_dtype(torch.float32)
from scipy.signal import tukey
from math import factorial


class HOSM_batchwise:
    """"
    High-Order Spectral Method (HOSM) based approach for batch-wise reconstruction of wave kinematics.
    This class implements a perturbation-based solution of potential flow theory to reconstruct vertical velocity
    and surface potential from the free surface elevation using a Taylor-series expansion in wave steepness.
    """
    def __init__(self, M, depth, x, t, g=9.81, device='cpu'):
        """
        Initialize
        :param M: perturbation order
        :param depth: water depth [m]
        :param x: spatial grid vector
        :param t: temporal grid vector
        :param g: gravitational acceleration
        :param device: device for tensor computation (GPU)
        """
        self.M = M
        self.g = g
        self.depth = depth
        self.spatial_dim = len(x)
        self.temporal_dim = len(t)
        self.device = device

        # --- windowing to stabilize FFT for non-periodic signals ---
        filter_window = torch.tensor(tukey(self.spatial_dim, alpha=0.05), device=self.device, dtype=torch.float32)
        self.filter_window = filter_window.unsqueeze(-1).expand(-1, self.temporal_dim,)

        # --- wavenumber vector and dispersion relation for wave angular frequency ---
        dx = x[1] - x[0]
        self.k_x = 2 * torch.pi * torch.fft.rfftfreq(len(x), d=dx, device=self.device)
        ome = torch.sqrt(g * self.k_x * torch.tanh(self.k_x * depth))
        ome[0] = float('inf')  # Avoid division by zero
        self.ome = ome.view(1, -1, 1)

        # --- precompute factorials and powers of k vector for repeated use ---
        self.factorials = torch.tensor([factorial(i) for i in range(self.M + 1)], device=self.device, dtype=torch.float32)
        kx_view = self.k_x.view(1, -1, 1)
        self.kx_powers = [torch.ones_like(kx_view)]
        for p in range(1, self.M + 2):
            self.kx_powers.append(kx_view ** p)

    def solve_perturbation_potentials(self, eta):
        """
        Compute perturbation potentials Phi^(m) up to order M. The potentials are computed recursively using a Taylor expansion
        in surface elevation. Each higher-order potential depends on lower-order terms and their vertical derivatives.
        :param eta: surface elevation (reconstructed by FNO) of shape (batch_size, nx, nt)
        :return phi_list: list of pertutbation potentials [Phi^(1), Phi^(2), ..., Phi^(M)], each of shape (batch, nx, nt).
        """

        # --- first-order potential (linear solution in Fourier space) ---
        f_eta = torch.fft.rfft(eta * self.filter_window, dim=1)
        phi1 = -torch.real(torch.fft.irfft((1j * self.g / self.ome) * f_eta, n=self.spatial_dim, dim=1))
        phi_list = [phi1]

        # --- higher-order perturbation potentials ---
        for m in range(2, self.M + 1):
            phi_m = torch.zeros_like(phi1)

            for l in range(1, m):
                phi_fft = torch.fft.rfft(phi_list[m - l - 1], dim=1)
                d_dz_phi = torch.real(torch.fft.irfft(self.kx_powers[l] * phi_fft, n=self.spatial_dim, dim=1))  # derivative in vertical direction via Fourier differentiation
                phi_m = phi_m - (eta ** l / self.factorials[l]) * d_dz_phi                                      # recursive accumulation

            phi_list.append(phi_m)

        return phi_list

    def calculate_vertical_velocity(self, eta):
        """
        This method combines perturbation potentials to reconstruct the vertical velocity W using the HOSM expansion.
        :param eta: surface elevation of shape (batch_size, nx, nt)
        :return W_tot: Vertical velocity field of shape (batch_size, nx, nt)
        :return phi_s: Surface potential of shape (batch_size, nx, nt)
        """

        # --- compute perturbation potentials up to order M ---
        phi_list = self.solve_perturbation_potentials(eta)

        # --- initialize vertical velocity ---
        batch_size = eta.shape[0]
        W_tot = torch.zeros((batch_size, self.spatial_dim, self.temporal_dim), device=self.device)

        # --- assemble vertical velocity via perturbation expansion ---
        for m in range(1, self.M + 1):
            for l in range(m):
                phi_fft = torch.fft.rfft(phi_list[m - l - 1], dim=1)
                d_dz_phi = torch.real(torch.fft.irfft((self.kx_powers[l+1]) * phi_fft, n=self.spatial_dim, dim=1))      # derivative in vertical direction via Fourier differentiation
                W_tot = W_tot + (eta ** l / self.factorials[l]) * d_dz_phi                                              # accumulate contribution of all orders

        phi_s = phi_list[0]  # surface potential is first-order term Phi^(1)

        return W_tot, phi_s

