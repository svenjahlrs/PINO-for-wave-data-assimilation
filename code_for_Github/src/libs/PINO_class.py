
import os.path
import time

import torch

from .HOS import HOSM_batchwise
from .utils import *
from .FNO_1D_to_2D import *
from .SSP import *
from scipy.signal import tukey
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float32)

class PINO_model:
    def __init__(self, cfg, x, t, list_buoy_indices):

        # --- full domain definition ---
        self.x = x
        self.t = t
        self.list_buoys = list_buoy_indices

        # --- reduced domain definition (required do to tukey window filtering) ---
        self.space_margin = int(len(self.x) * cfg.space_boundary_exclusion)
        self.time_margin = int(len(self.t) * cfg.time_boundary_exclusion)
        self.x_new = self.x[self.space_margin:-self.space_margin]
        self.t_new = self.t[self.time_margin:-self.time_margin]
        self.smaller_shape = np.array((len(self.x_new), len(self.t_new)))  # new nx, nt, 1

        # --- training progress save ---
        self.name_save = cfg.name_save
        self.FIGURE_PATH = f"../results/figures/{cfg.name_save}/"
        os.makedirs(self.FIGURE_PATH, exist_ok=True)
        self.MODEL_FILE = f'../results/models/model_{cfg.name_save}.pth'
        self.LOSS_FILE = os.path.join(f"../results/errors/loss_{cfg.name_save}.csv")
        self.w_data, self.w_dyn, self.w_kin, self.w_reg = cfg.w_data, cfg.w_dyn, cfg.w_kin, cfg.w_reg

        # --- wavenumber vectors for spectral differentiation ---
        self.k_x = 2 * torch.pi * torch.fft.rfftfreq(len(x), d=x[1] - x[0], device=device, dtype=torch.float32)[None, :, None]  # x is axis 1
        self.k_t = 2 * torch.pi * torch.fft.rfftfreq(len(t), d=t[1] - t[0], device=device, dtype=torch.float32)[None, None, :]  # t is axis 2

        # --- Tukey windows (stabilizes spectral derivatives for non-periodic boundary conditions) ---
        self.window_x = torch.tensor(tukey(len(x), alpha=cfg.tukey_alpha), device=device, dtype=torch.float32)[None, :, None]
        self.window_t = torch.tensor(tukey(len(t), alpha=cfg.tukey_alpha), device=device, dtype=torch.float32)[None, None, :]

        # --- setup of neural operator ---
        self.model_eta = FNO1d_x_to_2d(modes1=cfg.modes, modes2=cfg.modes, width=cfg.width, num_layers=cfg.fno_layers,
                                       in_channels=len(self.list_buoys), nx=len(x)).to(device)
        self.optimizer = torch.optim.AdamW(self.model_eta.parameters(), lr=cfg.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=cfg.step_scheduler, gamma=0.5)
        self.model_eta.apply(init_weight_bias)

        # --- initialize physics-module and metric ---
        self.hosm = HOSM_batchwise(M=cfg.M, g=9.81, depth=cfg.d, x=x, t=t, device=device)
        self.SSP = SurfaceSimilarityParameter2D()

        # --- training states initialization ---
        self.total_loss_train, self.total_loss_val = None, None
        self.best_loss_train, self.best_loss_val = np.inf, np.inf
        self.loss_nMSE_data, self.loss_nMSE_bc_kin, self.loss_nMSE_bc_dyn, self.loss_nMSE_reg, self.metric_SSP = 0, 0, 0, 0, 0
        self.val_nMSE_data, self.val_nMSE_bc_kin, self.val_nMSE_bc_dyn, self.val_nMSE_reg, self.val_metric_SSP = 0, 0, 0, 0, 0
        self.epoch = 0                                      # number of total training epochs
        self.epoch_time = time.time()                       # counter for training time per epoch
        self.early_stopping_count = 0                       # counter for training time per epoch
        self.early_stopping_max = cfg.early_stopping_at

        # --- prepare metrics/losses for logging and printing ---
        self.keys_train = ['epoch', 'train_loss', 'nMSE_data', 'nMSE_bc_kin', 'nMSE_bc_dyn', 'nMSE_reg', 'SSP']
        self.keys_val = ['epoch', 'val_loss', 'val_nMSE_data', 'val_nMSE_bc_kin', 'val_nMSE_bc_dyn', 'val_nMSE_reg', 'val_SSP']

    def cut_t_domain(self, var):
        """
        This operation crops the input tensor along the time axis to avoid artifacts from Fourier differentiation and Tukey window tapering near boundaries
        :param var: input field of shape (batch, n_buoys, nt)
        :return var_: cropped field of shape (batch, n_buoys, nt - 2*self.time_margin)
        """
        var_ = var[:, :, self.time_margin:-self.time_margin]
        return var_

    def cut_xt_domain(self, var):
        """
        This operation crops the input tensor along both space and time axis to avoid artifacts from Fourier differentiation and Tukey window tapering near boundaries
        :param var: input field of shape (batch, nx, nt)
        :return var_: cropped field of shape (batch, nx - 2*self.space_margin, nt - 2*self.time_margin)
        """
        var_ = var[:, self.space_margin:-self.space_margin, self.time_margin:-self.time_margin]
        return var_

    def fourier_derivative(self, var, dim):
        """
        Computes pseudo-spectral (Fourier-based) derivatives using a windowed differentiation scheme.
        (1) Applies a Tukey window to reduce boundary artifacts (as usually no periodic boundary conditions are available)
        (2) Transforms the field into Fourier space,
        (3) Applies the analytical derivative operator in spectral domain (multiplication by jk in Fourier space is differentiation in physical space)
        (4) Transforms back to physical space.
        :param var: input field (e.g., eta or phiS) of shape (batch, nx, nt)
        :param dim: dimension along which to compute the derivative (1: x-direction, 2: t-direction)
        :return deriv: Derivative of input field along specified dimension
        """

        # --- select derivative dimension ---
        if dim == 1:
            k = self.k_x
            win = self.window_x
        elif dim == 2:
            k = self.k_t
            win = self.window_t

        # --- windowing to reduce spectral leakage for non-periodic boundary data ---
        var_win = var * win

        # --- Fourier differentiation
        n = var.shape[dim]
        fft_var = torch.fft.rfft(var_win, dim=dim)
        d_fft_var = 1j * k * fft_var
        deriv = torch.fft.irfft(d_fft_var, n=n, dim=dim)

        return deriv

    def forward(self, eta_in):
        """
        Forward pass of the Physics-Informed Neural Operator (PINO). This function combines: a neural operator mapping sparse buoy measurements to a full wave field
        and a physics-based reconstruction of the vertical velocity and surface potential using a high-order spectral (HOSM) formulation.
        :param eta_in: sparse surface elevation measurements at buoy locations of shape (batch_size, num_buoys, nt)
        :return eta_out: full surface elevation field of shape (batch_size, nx, nt) from neural operator output
        :return phiS: surface potential field of shape (batch_size, nx, nt) reconstructed from eta_out via HOSM-coupling
        :return W: vertical velocity field of shape (batch_size, nx, nt) reconstructed from eta_out via HOSM-coupling
        """

        # --- neural operator prediction (wave field reconstruction) ---
        eta_out = self.model_eta(eta_in)

        # --- physics-based reconstruction of vertical velocity W and surface potential phiS with HOSM-based Taylor series expansion ---
        W, phiS = self.hosm.calculate_vertical_velocity(eta_out)

        return eta_out, phiS, W

    def loss(self, eta_in, eta_true, mode='trainset'):
        """
        Physics-informed multiobjective loss for PINO training combines:
        (1) data consistency at sparse buoy locations,
        (2) free-surface physics constraints derived from potential flow equations,
        (3) statistical regularization of the reconstructed wave field.
        The loss is computed per sample and normalized by the signal energy to ensure scale invariance across different wave conditions.
        :param eta_in: sparse surface elevation measurements at buoy locations of shape (batch_size, num_buoys, nt)
        :param eta_true: true surface elevation field of shape (batch_size, nx, nt) used for metric (SSP) evaluation and post-training comparison only (not as PINO input or in a loss term!)
        :param mode: specifies whether the loss is computed for training or validation set
        :return loss: scalar total loss for current batch combining data, physics, and regularization terms
        :return eta_out: reconstructed surface elevation from the PINO cropped to smaller evaluation domain of shape (batch_size, nx_, nt_)
        :return phisS: surface potential of shape (batch_size, nx_, nt_) derived from eta_out
        :return eta_true: true surface elevation field cropped to smaller evaluation domain of shape (batch_size, nx_, nt_)
        """

        # --- forward pass ---
        this_batch = eta_in.shape[0]  # required as last batch per epoch might be smaller
        eta_out, phiS, W = self.forward(eta_in=eta_in)  # fields of shape (this_batch, nx, nt)

        # --- calculate spectral derivatives (Fourier differentiation) for physics-loss ---
        eta_x = self.fourier_derivative(eta_out, dim=1)
        eta_t = self.fourier_derivative(eta_out, dim=2)
        phiS_x = self.fourier_derivative(phiS, dim=1)
        phiS_t = self.fourier_derivative(phiS, dim=2)

        # --- calculate data loss term of shape (this_batch, 1) at sparse buoy locations ---
        MSE_data = torch.mean(torch.square(eta_in - eta_out[:, self.list_buoys, :]), dim=(1, 2))

        # --- crop field to smaller domain (this_batch, nx, nt) -> (this_batch, nx_, nt_) required as boundary regions affected by Tukey window tapering and Fourier differentiation ---
        phiS = self.cut_xt_domain(phiS)
        W = self.cut_xt_domain(W)
        eta_true = self.cut_xt_domain(eta_true)
        eta_t = self.cut_xt_domain(eta_t)
        eta_x = self.cut_xt_domain(eta_x)
        phiS_x = self.cut_xt_domain(phiS_x)
        phiS_t = self.cut_xt_domain(phiS_t)
        eta_out = self.cut_xt_domain(eta_out)
        eta_in = self.cut_t_domain(eta_in)

        # --- calculate physics loss terms of shape (this_batch, 1)  for free surface boundary conditions ---
        MSE_bc_kin = torch.mean(torch.square(eta_t + phiS_x * eta_x - W * (1 + eta_x ** 2)), dim=(1, 2))
        MSE_bc_dyn = torch.mean(torch.square((phiS_t + 9.81 * eta_out + 0.5 * phiS_x ** 2 - 0.5 * W ** 2 * (1 + eta_x ** 2))), dim=(1, 2))

        # --- regularization loss terms of shape (this_batch, 1) ensure predicted field preserves variance structure and avoid trivial solution of zero-elevation ---
        std_out = torch.std(eta_out, dim=(1, 2))
        std_true = torch.std(eta_in, dim=(1, 2))
        MSE_reg = torch.square(std_true - std_out)

        # --- normalization factor for loss terms of shape (this_batch, 1) for energy-scaled loss terms ---
        norm_factor = torch.mean(torch.square(eta_in), dim=(1, 2))

        # --- normalization of loss terms and mean across samples in batch ---
        nMSE_data = torch.mean(torch.divide(MSE_data, norm_factor))
        nMSE_bc_kin = torch.mean(torch.divide(MSE_bc_kin, norm_factor))
        nMSE_bc_dyn = torch.mean(torch.divide(MSE_bc_dyn, norm_factor))
        nMSE_reg = torch.mean(torch.divide(MSE_reg, norm_factor))

        # --- evaluation metric (not part of loss) ---
        SSP = torch.mean(self.SSP(y_true=eta_true, y_pred=eta_out))

        # --- accumulation of training/validation terms for samples in all batches ---
        if mode == 'trainset':
            self.loss_nMSE_bc_kin += nMSE_bc_kin.item() * this_batch
            self.loss_nMSE_bc_dyn += nMSE_bc_dyn.item() * this_batch
            self.loss_nMSE_data += nMSE_data.item() * this_batch
            self.loss_nMSE_reg += nMSE_reg.item() * this_batch
            self.metric_SSP += SSP.item() * this_batch
        elif mode == 'valset':
            self.val_nMSE_bc_kin += nMSE_bc_kin.item() * this_batch
            self.val_nMSE_bc_dyn += nMSE_bc_dyn.item() * this_batch
            self.val_nMSE_data += nMSE_data.item() * this_batch
            self.val_nMSE_reg += nMSE_reg.item() * this_batch
            self.val_metric_SSP += SSP.item() * this_batch

        # --- final total multi-objective loss value for batch ---
        loss = self.w_data * nMSE_data + self.w_kin * nMSE_bc_kin + self.w_dyn * nMSE_bc_dyn + self.w_reg * nMSE_reg

        return loss, eta_out, phiS, eta_true

    def training(self, train_loader, val_loader, epochs):
        """
         Executes the full training loop of the PINO model, including:
        (1) optional model loading if a checkpoint exists
        (2) physics-informed training over multiple epochs
        (3) validation after each epoch
        (4) learning rate scheduling
        (5) logging, checkpointing and early stopping
        :param train_loader: data loader providing training batches
        :param val_loader: data loader providing validation batches
        :param epochs: number of training epochs to run
        """

        # --- load existing checkpoint if available ---
        if os.path.exists(self.MODEL_FILE):
            self.load_model()
            return  # training is skipped if model already exists

        self.optimizer.zero_grad()  # initialize optimizer state

        for ep in range(epochs):

            # --- training phase ---
            self.model_eta.train()                                            # set model to training mode
            for eta_in, eta_tr, info in train_loader:                         # iteration in batches over entire dataset
                eta_in = eta_in.to(device, non_blocking=True)                 # move data to GPU
                eta_tr = eta_tr.to(device, non_blocking=True)
                self.optimizer.zero_grad()
                loss, _, _, _ = self.loss(eta_in=eta_in, eta_true=eta_tr)     # physics-informed loss computation for batch
                loss.backward()                                               # backward pass
                self.optimizer.step()                                         # parameter update

            self.scheduler.step()  # learning rate decay

            # --- validation phase ---
            self.model_eta.eval()                                             # set model to validation mode
            with torch.no_grad():                                             # disables gradient tracking for efficiency
                for eta_in, eta_tr, info in val_loader:                       # iteration in batches over entire dataset
                    eta_in = eta_in.to(device, non_blocking=True)             # move data to GPU
                    eta_tr = eta_tr.to(device, non_blocking=True)
                    self.loss(eta_in=eta_in, eta_true=eta_tr, mode='valset')  # forward pass only for metric accumulation and val_loss tracking

            # --- mean loss from accumulated losses and reset running loss terms
            self.reset_batch_counters(num_train=len(train_loader.dataset), num_val=len(val_loader.dataset))

            # --- check early stopping condition at the end of each epoch ---
            if self.early_stopping_count == self.early_stopping_max:
                print('\nno further model improvement\n')
                break

    def reset_batch_counters(self, num_train, num_val):
        """
        During training, individual loss terms are accumulated batch-wise and weighted by batch size.
        This function converts these accumulated sums into mean values for the entire dataset in the current epoch.
        After epoch-level aggregation, these counters must be cleared to avoid leakage into the next epoch.
        :param num_train: total number of samples in the training dataset
        :param num_val: total number of samples in the validation dataset
        """

        # --- convert accumulated sums to mean losses over full training dataset ---
        self.loss_nMSE_bc_kin = self.loss_nMSE_bc_kin / num_train
        self.loss_nMSE_bc_dyn = self.loss_nMSE_bc_dyn / num_train
        self.loss_nMSE_data = self.loss_nMSE_data / num_train
        self.loss_nMSE_reg = self.loss_nMSE_reg / num_train
        self.metric_SSP = self.metric_SSP / num_train
        self.total_loss_train = self.w_data * self.loss_nMSE_data + self.w_kin * self.loss_nMSE_bc_kin + self.w_dyn * self.loss_nMSE_bc_dyn + self.w_reg * self.loss_nMSE_reg

        # --- convert accumulated sums to mean losses over full validation dataset ---
        self.val_nMSE_bc_kin = self.val_nMSE_bc_kin / num_val
        self.val_nMSE_bc_dyn = self.val_nMSE_bc_dyn / num_val
        self.val_nMSE_data = self.val_nMSE_data / num_val
        self.val_nMSE_reg = self.val_nMSE_reg / num_val
        self.val_metric_SSP = self.val_metric_SSP / num_val
        self.total_loss_val = self.w_data * self.val_nMSE_data + self.w_kin * self.val_nMSE_bc_kin + self.w_dyn * self.val_nMSE_bc_dyn + self.w_reg * self.val_nMSE_reg

        self.callback()  # logs and prints mean loss terms and metrics for this epoch to csv

        # --- save model if validation loss decreased ---
        if self.epoch > 10:
            if self.total_loss_val < self.best_loss_val:
                self.best_loss_train = self.total_loss_train
                self.best_loss_val = self.total_loss_val
                self.save_model()
                self.early_stopping_count = 0
            else:
                self.early_stopping_count += 1   # no improvement: increase early stopping counter

        # --- clear counters to avoid leakage into the next epoch ---
        self.loss_nMSE_bc_kin, self.loss_nMSE_bc_dyn, self.loss_nMSE_data, self.loss_nMSE_reg, self.metric_SSP = 0, 0, 0, 0, 0
        self.val_nMSE_bc_kin, self.val_nMSE_bc_dyn, self.val_nMSE_data, self.val_nMSE_reg, self.val_metric_SSP = 0, 0, 0, 0, 0

    def callback(self):
        """
        Logs epoch-level training and validation loss terms/ metrics. This function is called at the end of each epoch and performs:
        (1) computation and display of elapsed epoch time,
        (2) formatted console logging of training and validation loss terms/ metrics,
        (3) storage of metrics to a CSV file,
        (4) update of epoch counter and timing reference.
        """

        # --- timing for epoch ---
        elapsed_time = time.time() - self.epoch_time
        print(f'time: {np.round(elapsed_time, 3)} s')

        # --- print training and validation loss/metric values of current epoch to console ---
        vals_train = [self.epoch, self.total_loss_train, self.loss_nMSE_data, self.loss_nMSE_bc_kin, self.loss_nMSE_bc_dyn, self.loss_nMSE_reg, self.metric_SSP]
        vals_val = [self.epoch, self.total_loss_val,  self.val_nMSE_data, self.val_nMSE_bc_kin, self.val_nMSE_bc_dyn, self.val_nMSE_reg, self.val_metric_SSP]
        print("".join(str(key) + ": " + str(round(value, 10)) + ", " for key, value in zip(self.keys_train, vals_train)))
        print("".join(str(key) + ": " + str(round(value, 10)) + ", " for key, value in zip(self.keys_val, vals_val)))

        # --- CSV-logging ---
        if not os.path.exists(self.LOSS_FILE):
            write_csv_line(path=self.LOSS_FILE, line=self.keys_train + self.keys_val)
        write_csv_line(path=self.LOSS_FILE, line=vals_train + vals_val)

        # --- training state update ---
        self.epoch += 1
        self.epoch_time = time.time()

    def load_model(self):
        """
        Loads a previously saved model checkpoint, including network parameters, optimizer state, and training metadata.
        """

        checkpoint = torch.load(self.MODEL_FILE)
        self.model_eta.load_state_dict(checkpoint['net_eta'])
        self.epoch = checkpoint['epoch']
        self.smaller_shape = checkpoint['smaller_shape']
        self.optimizer.load_state_dict(checkpoint['optim']),
        self.best_loss_train = checkpoint['best_train'],
        self.best_loss_val = checkpoint['best_val']
        print(f'\nLoaded model from epoch {self.epoch}: \n\n')

    def save_model(self):
        """
        Saves the current model state as a checkpoint (pth) file.
        """

        torch.save(
            {'net_eta': self.model_eta.state_dict(),
             'epoch': self.epoch,
             'smaller_shape': self.smaller_shape,
             'optim': self.optimizer.state_dict(),
             'best_train': self.best_loss_train,
             'best_val': self.best_loss_val},
            self.MODEL_FILE)
        print(f'model checkpoint saved')

    def predict(self, loader):
        """
        Performs batch-wise inference using the trained PINO model.
        :param loader: DataLoader for test set providing batches of (eta_in, eta_true, info, phi_true), while eta_in is the only input provided to the PINO and eta_true, phi_true are reserved for post-training comparison
        :return eta_pred: predicted surface elevation by PINO cropped to smaller evaluation domain of shape (n_samp_test, nx_, nt_)
        :return phi_pred: surface potential calculated from eta_pred via HOSM-formulation cropped to smaller evaluation domain of shape (n_samp_test, nx_, nt_)
        :return eta_meas: input buoy measurements of shape (n_samp_test, num_buoys, nt_)
        :return eta_true : true surface elevation cropped to smaller evaluation domain of shape (n_samp_test, nx_, nt_) used for metric (SSP) evaluation and post-training comparison only (not as PINO input or in a loss term!)
        :return phi_true : true surface potential cropped to smaller evaluation domain of shape (n_samp_test, nx_, nt_) used for metric (SSP) evaluation and post-training comparison only (not as PINO input or in a loss term!)
        :return info: metadata (peak wavelength Lp and steepness eps) for each sample, shape (n_samp_test, 2)
        """

        self.model_eta.eval()

        # --- initialize dynamic containers (memory-efficient) ---
        eta_pred_list, phi_pred_list, eta_meas_list, eta_true_list, phi_true_list, info_list = [], [], [], [], [], []

        with torch.no_grad():
            for eta_in, eta_tr, info, phi_tr in loader:

                # --- move batch to device ---
                eta_in = eta_in.to(device)

                # --- forward inference ---
                tic = time.time()
                eta_out, phi_out, _ = self.forward(eta_in=eta_in)
                toc = time.time() - tic

                print(f'inference for batch (size: {eta_in.shape[0]}) took {toc:.5f} s')

                # --- crop to smaller domain (unaffected by Tukey window and boundary artefacts) & move to CPU ---
                eta_pred_list.append(torch_tensor_to_np(self.cut_xt_domain(eta_out)))
                phi_pred_list.append(torch_tensor_to_np(self.cut_xt_domain(phi_out)))
                eta_meas_list.append(torch_tensor_to_np(self.cut_t_domain(eta_in)))
                eta_true_list.append(torch_tensor_to_np(self.cut_xt_domain(eta_tr)))
                phi_true_list.append(torch_tensor_to_np(self.cut_xt_domain(phi_tr)))
                info_list.append(torch_tensor_to_np(info))

        # --- concatenate once at the end ---
        eta_pred = np.concatenate(eta_pred_list, axis=0)
        phi_pred = np.concatenate(phi_pred_list, axis=0)
        eta_meas = np.concatenate(eta_meas_list, axis=0)
        eta_true = np.concatenate(eta_true_list, axis=0)
        phi_true = np.concatenate(phi_true_list, axis=0)
        info = np.concatenate(info_list, axis=0)

        return eta_pred, phi_pred, eta_meas, eta_true, phi_true, info