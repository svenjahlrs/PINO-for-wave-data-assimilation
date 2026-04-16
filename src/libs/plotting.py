import pandas as pd
import matplotlib.pyplot as plt
import os
from .SSP import *
import matplotlib.ticker as tick
from matplotlib.gridspec import GridSpec

plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
plt.rc('font', family='serif')
plt.rc('font', size=8)
plt.rc('axes', labelsize=8)
plt.rc('axes', titlesize=8)
plt.rc('legend', title_fontsize=6)
plt.rc('legend', fontsize=6)
plt.rc('xtick', labelsize=7)
plt.rc('ytick', labelsize=7)

dyn_red = '#8d1f22'
dyn_cyan = '#1b7491'
dyn_grey = '#5b7382'
dyn_dark = '#0c333f'

dyn_blau = '#00344e'
dyn_lightblue = '#ecfbff'
dyn_pink = '#d19ea3'


def plot_epoch_results_eta_phi_specific_cross_sec(eta_in, eta_out, phi_out, eta_true, phi_true, space_margin, Lp, eps, x_new, t_new, list_buoys, path_save, num_plot=1):
    """
    Visualizes PINO reconstruction results for a single sample including:
    (1) 2D spatio-temporal fields of surface elevation and surface potential (true and PINO reconstruction)
    (2) Sparse buoy measurements
    (3) Cross-sectional time series comparisons at selected spatial locations
    (4) Surface Similarity Parameter (SSP) metrics
    :param eta_in: sparse buoy measurements of shape (num_buoys, nt_)
    :param eta_out: reconstructed surface elevation of shape (nx_, nt_)
    :param phi_out: reconstructed surface potential of shape (nx_, nt_)
    :param eta_true: true surface elevation of shape (nx_, nt_)
    :param phi_true: true surface potential of shape (nx_, nt_)
    :param space_margin: index whee the cropped space representation starts
    :param Lp: peak wavelength (used for annotation)
    :param eps: wave steepness parameter (used for annotation)
    :param x_new: spatial grid vector (cropped domain)
    :param t_new: temporal grid (cropped domain)
    :param list_buoys: indices of buoy locations in spatial grid
    :param path_save: directory to save figure
    :param num_plot: index of plotted sample (used for filename and title)
    """

    # --- compute global SSP metrics ---
    SSPeta = SSP_2D_metric(eta_out, eta_true)
    SSPphi = SSP_2D_metric(phi_out, phi_true)
    
    # --- create meshgrid for plotting and initialize figure layout---
    TT, XX = np.meshgrid(t_new, x_new)
    fig = plt.figure(figsize=(7, 5.5), dpi=1000)
    fig.suptitle(
        f'test set sample no. {num_plot}' + ' with $L_\mathrm{p}'
        + f'={np.round(Lp, 0)}$ m, $\epsilon={np.round(eps.astype(np.float64), 2)}:$\n'
        + r'$\mathrm{SSP}\left( \tilde{\eta}, \eta_\mathrm{HOSM} \right)=' + f'{np.round(SSPeta.item(), 4)}, '
        + r'\; \mathrm{SSP}\left( \tilde{\Phi}^\mathrm{s}, \Phi_\mathrm{HOSM}^\mathrm{s} \right)='
        + f'{np.round(SSPphi.item(), 4)}$', x=0.53)
    gs = GridSpec(2, 3, figure=fig, left=0.09, right=0.73, top=0.87, bottom=0.09, wspace=0.1, hspace=0.4)
    ax1 = fig.add_subplot(gs[0, 0])  # buoy input
    ax2 = fig.add_subplot(gs[0, 1])  # eta prediction
    ax3 = fig.add_subplot(gs[0, 2])  # eta true
    ax5 = fig.add_subplot(gs[1, 1])  # phi prediction
    ax6 = fig.add_subplot(gs[1, 2])  # phi true

    # --- reconstruct sparse buoy field for visualization ---
    eta_helper = np.empty(eta_true.shape)
    eta_helper[:] = np.nan
    for i, ind in enumerate(list_buoys):
        if i == 0:
            eta_helper[ind - space_margin, :] = eta_in[i]
            eta_helper[ind - (space_margin - 1), :] = eta_in[i]
            eta_helper[ind - (space_margin - 2), :] = eta_in[i]

        elif i == len(list_buoys) - 1:
            eta_helper[ind - space_margin, :] = eta_in[i]
            eta_helper[ind - (space_margin + 1), :] = eta_in[i]
            eta_helper[ind - (space_margin + 2), :] = eta_in[i]
        else:
            eta_helper[ind - space_margin, :] = eta_in[i]

    # ======================
    # --- ETA FIELD PLOTS ---
    # ======================

    plt.rcParams['axes.titlepad'] = 29
    mima = np.abs(np.max([np.max(eta_out), np.max(eta_true)]))  # for equal color scaling

    # --- input (buoy measurements) ---
    ax1.set_title(r'input buoy meas. $\eta_\mathrm{m}\, [\mathrm{m}]$')
    print(TT.shape, XX.shape, eta_helper.shape)
    pos1 = ax1.pcolormesh(TT, XX, eta_helper, vmin=-mima, vmax=mima, edgecolor='face', rasterized=True)
    ax1.set_xlabel('$t\, [\mathrm{s}]$')
    ax1.set_ylabel('$x\, [\mathrm{m}]$')
    cbar = fig.colorbar(pos1, ax=ax1, location='top')
    cbar.ax.tick_params(labelsize=7)

    # --- PINO reconstruction ---
    ax2.set_title(r'PINO reconstruction $\tilde{\eta}\, [\mathrm{m}]$')
    pos2 = ax2.pcolormesh(TT, XX, eta_out, vmin=-mima, vmax=mima, edgecolor='face', rasterized=True)
    ax2.set_xlabel('$t\, [\mathrm{s}]$')
    ax2.set_yticklabels([])
    cbar = fig.colorbar(pos2, ax=ax2, location='top')
    cbar.ax.tick_params(labelsize=7)

    # --- true reference for comparison ---
    ax3.set_title('true wavefield $\eta_\mathrm{HOSM}\, [\mathrm{m}]$')
    pos3 = ax3.pcolormesh(TT, XX, eta_true, vmin=-mima, vmax=mima, edgecolor='face', rasterized=True)
    ax3.set_xlabel('$t\, [\mathrm{s}]$')
    ax3.set_yticklabels([])
    cbar = fig.colorbar(pos3, ax=ax3, location='top')
    cbar.ax.tick_params(labelsize=7)

    # --- eta cross sections (time slices) ---
    gs_sub = GridSpec(3, 1, figure=fig, left=0.79, right=0.985, top=0.835, bottom=0.545)
    ax_sub = [fig.add_subplot(gs_sub[i]) for i in range(3)]
    x_slice_ind = np.linspace(0, len(x_new) - 1, 9).astype(int)     # select representative spatial indices
    j = [1, 4, 7]
    x_slice_ind = x_slice_ind[j]
    for ii, (idx, axi) in enumerate(zip(x_slice_ind[::-1], ax_sub)):
        SSP_i = SSP_metric(eta_true[idx, :], eta_out[idx, :])
        axi.plot(t_new, eta_true[idx, :], label='true $\eta_\mathrm{HOSM}(t) \, [\mathrm{m}]$', c=dyn_cyan, linewidth=0.9)
        axi.plot(t_new, eta_out[idx, :], '--', label=r'PINO $\tilde{\eta}(t) \, [\mathrm{m}]$', c=dyn_red, linewidth=0.9)
        axi.text(0.02, 0.94, f'$x = {x_new[idx]:.2f} \,' + '\mathrm{m}, \: \mathrm{SSP}='
                 + f'{np.round(SSP_i, 4)}$', transform=axi.transAxes, va='top', ha='left', fontsize=6)
        if ii < 2:
            axi.set_xticklabels([])
        if ii == 2:
            axi.set_xlabel('$t\, [\mathrm{s}]$')
        axi.set_xlim([min(t_new), max(t_new)])
        axi.set_ylim([-1.0 * mima, 1.4 * mima])
        if ii == 0:
            axi.legend(ncol=1, loc='upper center', bbox_to_anchor=(0.5, 1.75))

    # ======================
    # --- PHI FIELD PLOTS ---
    # ======================

    plt.rcParams['axes.titlepad'] = 33
    mima = np.abs(np.max([np.max(phi_out), np.max(phi_true)]))

    # --- PINO reconstruction ---
    ax5.set_title(r'PINO reconstruction $\tilde{\Phi}^\mathrm{s}\, [\frac{\mathrm{m}^2}{\mathrm{s}}]$')
    pos5 = ax5.pcolormesh(TT, XX, phi_out, vmin=-mima, vmax=mima, edgecolor='face', rasterized=True)
    ax5.set_xlabel('$t\, [\mathrm{s}]$')
    ax5.set_ylabel('$x\, [\mathrm{m}]$')
    cbar = fig.colorbar(pos5, ax=ax5, location='top')
    cbar.ax.tick_params(labelsize=7)

    # --- true reference for comparison ---
    ax6.set_title(r'true potential $\Phi^\mathrm{s}_\mathrm{HOSM}\, [\frac{\mathrm{m}^2}{\mathrm{s}}]$')
    pos6 = ax6.pcolormesh(TT, XX, phi_true, vmin=-mima, vmax=mima, edgecolor='face', rasterized=True)
    ax6.set_xlabel('$t\, [\mathrm{s}]$')
    ax6.set_yticklabels([])
    cbar = fig.colorbar(pos6, ax=ax6, location='top')
    cbar.ax.tick_params(labelsize=7)

    # --- phi cross sections (time slices) ---
    gs_sub2 = GridSpec(3, 1, figure=fig, left=0.79, right=0.985, top=0.38, bottom=0.09)
    ax_sub = [fig.add_subplot(gs_sub2[i]) for i in range(3)]
    for ii, (idx, axi) in enumerate(zip(x_slice_ind[::-1], ax_sub)):
        SSP_i = SSP_metric(phi_true[idx, :], phi_out[idx, :])
        axi.plot(t_new, phi_true[idx, :], label=r'true $\Phi^\mathrm{s}_\mathrm{HOSM}(t)\, \left[\mathrm{m}^2\mathrm{s}^{-1}\right]$', c=dyn_cyan, linewidth=0.9)
        axi.plot(t_new, phi_out[idx, :], '--', label=r'PINO $\tilde{\Phi}^\mathrm{s}(t)\, \left[\mathrm{m}^2\mathrm{s}^{-1}\right]$', c=dyn_red, linewidth=0.9)
        axi.text(0.02, 0.94, f'$x = {x_new[idx]:.2f} \,' + '\mathrm{m}, \: \mathrm{SSP}='
                 + f'{np.round(SSP_i, 4)}$', transform=axi.transAxes, va='top', ha='left', fontsize=6)
        axi.set_yticks([-10, 0, 10])
        if ii < 2:
            axi.set_xticklabels([])
        if ii == 2:
            axi.set_xlabel('$t\, [\mathrm{s}]$')
        axi.set_xlim([min(t_new), max(t_new)])
        axi.set_ylim([-1.05 * mima, 1.45 * mima])
        if ii == 0:
            axi.legend(ncol=1, loc='upper center', bbox_to_anchor=(0.5, 2.02))

    # --- finalize and save figure ---
    plt.tight_layout(w_pad=1.2)
    plt.savefig(f'{path_save}/samp_{num_plot}.png')


def plotting_losscurve_name(path_loss, path_save, figsize, **kwargs):
    """
    Plots training and validation loss curves from a CSV log file and saves the figures.

    This function generates three separate plots:
    (1) Individual normalized loss components (data, boundary conditions, regularization),
    (2) Total loss evolution with indication of the best validation model,
    (3) Surface Similarity Parameter (SSP) evolution.
    :param path_loss: Path to the CSV file containing logged training and validation metrics.
    :param path_save: base path (without file ending) where generated plots will be saved.
    :param figsize: figure size for all plots
    :param **kwargs: dict, optional plotting options:
            - yrange: tuple (ymin, ymax) to manually set y-axis limits
            - xmax: int to limit the number of epochs shown on x-axis
    """
    # --- load training log data ---
    df = pd.read_csv(path_loss)

    # ============================================================
    # (1) Plot individual loss components
    # ============================================================
    plt.figure(figsize=figsize)
    plt.plot(df['nMSE_data'], '--', color=dyn_red, label='nMSE data train', linewidth=0.8)
    plt.plot(df['val_nMSE_data'], color=dyn_red, label='nMSE data val', linewidth=1.2)
    plt.plot(df['nMSE_bc_kin'], '--', color=dyn_cyan, label='nMSE BC kin train', linewidth=0.8)
    plt.plot(df['val_nMSE_bc_kin'], color=dyn_cyan, label='nMSE BC kin val', linewidth=1.2)
    plt.plot(df['nMSE_bc_dyn'], '--', color='darkgreen', label='nMSE BC dyn train', linewidth=0.8)
    plt.plot(df['val_nMSE_bc_dyn'], color='darkgreen', label='nMSE BC dyn val', linewidth=1.2)
    plt.plot(df['nMSE_reg'], '--', color='peru', label='nMSE reg. train', linewidth=0.8)
    plt.plot(df['val_nMSE_reg'], color='peru', label='nMSE reg. val', linewidth=1.2)
    plt.yscale('log')   # log scaling for better visibility across magnitudes
    plt.yticks([0.01, 0.1, 1], ['0.01', '0.1', '1'])
    if 'yrange' in kwargs:
        plt.ylim(kwargs['yrange'])
    if 'xmax' in kwargs:
        plt.xlim([0, kwargs['xmax']])
    plt.xlabel('epochs')
    plt.ylabel('loss ')
    plt.grid(True, which="both")
    plt.tick_params(axis='both')
    plt.legend(loc=1, ncol=2)
    plt.tight_layout()
    plt.savefig(path_save + f'_all.png')

    # ============================================================
    # (2) Plot total loss and best validation model
    # ============================================================

    # --- determine best validation loss and corresponding epoch ---
    best_val = min(df.val_loss)
    index_best_val = df.val_loss.idxmin()

    plt.figure(figsize=figsize)
    plt.plot(df.train_loss, '--', color=dyn_cyan, label='total loss train', linewidth=0.8)
    plt.plot(df.val_loss, color=dyn_cyan, label='total loss val', linewidth=1.2)
    plt.plot(index_best_val, best_val, color=dyn_dark, linestyle='none', marker='x', markersize=4, label='best $\mathcal{M}$ on test set') # highlight best validation model
    plt.yscale('log')
    plt.yticks([0.1, 1], ['0.1', '1'])
    if 'yrange' in kwargs:
        plt.ylim(kwargs['yrange'])
    if 'xmax' in kwargs:
        plt.xlim([0, kwargs['xmax']])
    plt.xlabel('epochs')
    plt.ylabel('loss ')
    plt.grid(True, which="both")
    plt.tick_params(axis='both')
    plt.legend(loc=1)
    plt.tight_layout()
    plt.savefig(path_save + f'_total.png')

    # ============================================================
    # (3) Plot SSP metric evolution
    # ============================================================

    plt.figure(figsize=figsize)
    plt.plot(df.SSP, '--', color=dyn_cyan, label='SSP train', linewidth=0.8)
    plt.plot(df.val_SSP, color=dyn_cyan, label='SSP val', linewidth=1.2)
    plt.yscale('log')
    plt.yticks([0.1, 1], ['0.1', '1'])
    if 'yrange' in kwargs:
        plt.ylim(kwargs['yrange'])
    if 'xmax' in kwargs:
        plt.xlim([0, kwargs['xmax']])
    plt.xlabel('epochs')
    plt.ylabel('SSP metric')
    plt.grid(True, which="both")
    plt.tick_params(axis='both')
    plt.legend(loc=1)
    plt.tight_layout()
    plt.savefig(path_save + f'_SSP.png')


def SSP_vs_parameter_plot(y_true: np.ndarray, y_pred: np.ndarray, param1: np.ndarray, param2: np.ndarray, path_save, **kwargs):
    """
    Computes and visualizes the Surface Similarity Parameter (SSP) as a function of two independent wave field parameters (eps and Lp) 
    using a 2D binned error surface. For each sample, the SSP metric between predicted and true fields is computed. The results are then
    aggregated over all unique combinations of two parameters Lp and eps. The mean SSP per parameter combination is visualized as a
    2D color map (top-view surface plot).
    :param y_true: true reference wave fields of shape (num_samples, nx_, nt_)
    :param y_pred: predicted wave fields of shape (num_samples, nx_, nt_)
    :param param1: first parameter (peak wavelength), shape (num_samples,)
    :param param2: second parameter (steepness), shape (num_samples,)
    :param path_save: directory where the resulting plot will be saved
    :param **kwargs : dict, optional plotting parameters:
            - max_z : upper limit of color scale
            - min_z : lower limit of color scale
    """

    # --- (1) Compute SSP metric for each sample ---
    num_samp = y_true.shape[0]
    SSPs = np.zeros(num_samp)
    for i in range(num_samp):
        SSPs[i] = SSP_2D_metric(y_true[i], y_pred[i])
    mean_error = np.around(np.mean(SSPs, dtype=np.float64), 3)

    # --- (2) Define parameter bins (unique parameter values) ---
    bins1 = np.sort(np.unique(param1))
    bins2 = np.sort(np.unique(param2))
    bin_means = np.ones(shape=(len(bins1), len(bins2))) * mean_error  # initialize matrix with global mean (avoids empty entries)

    # --- (3) Aggregate SSP values for each parameter combination ---
    for i in range(len(bins1)):
        for j in range(len(bins2)):

            sum_err = 0
            count = 0

            for idx in range(len(SSPs)):                                # iterate over all samples and collect matching parameter pairs
                if param1[idx] == bins1[i] and param2[idx] == bins2[j]:
                    sum_err = sum_err + SSPs[idx]
                    count = count + 1

            if count > 0:
                # compute mean SSP for this parameter combination
                bin_means[i, j] = sum_err / count
            else:
                # fallback for missing combinations (interpolate from neighbors)
                bin_means[i, j] = (bin_means[i, j - 1] + bin_means[i - 1, j]) / 2
                print(f'Combination {bins1[i]} - {bins2[j]} not available')

    # --- (4) plot 2D SSP surface in top view ---
    vmax = kwargs['max_z'] if 'max_z' in kwargs else np.max(bin_means)
    vmin = kwargs['min_z'] if 'min_z' in kwargs else np.min(bin_means)

    fig, ax = plt.subplots(1, 1, figsize=(3, 2.3), dpi=100)
    X, Y = np.meshgrid(bins1, bins2)
    cp = ax.pcolor(X, Y, np.nan_to_num(bin_means.T), cmap=plt.cm.get_cmap('bone').reversed(), vmax=vmax, vmin=vmin, shading='auto')
    cb = fig.colorbar(cp, ax=ax, extend='max')  # Add a colorbar to a plot
    cb.set_label('mean SSP for $L_\mathrm{p}$-$\epsilon$-comb.')
    cb.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
    ax.set_xlabel('peak wavelength $L_\mathrm{p}\, [\mathrm{m}]$')
    ax.set_ylabel('steepness $\epsilon \, [-]$')
    ax.set_title(f"mean SSP = {np.round(np.mean(SSPs), 4)}, variance = {np.round(np.var(SSPs), 6)}")
    plt.tight_layout()
    plt.savefig(os.path.join(path_save, f'error_surface_SSP.png'))

