import torch
torch.set_default_dtype(torch.float32)
from libs.plotting import *
from libs.data_utils import *
from libs.PINO_class import *
from libs.config import Config as cfg


def main():
    # --- reproducibility ---
    set_torch_seed(seed=1234)

    # --- data loading and processing ---
    eta_true, eta_meas, phi_true, info_Ls_eps, x, t, buoy_ind = load_and_prepare_data(PATH_DATA=cfg.data_path, cfg=cfg)

    train_loader, val_loader, test_loader = split_data_and_create_dataloaders(eta_true=eta_true, phi_true=phi_true,
                                                                              eta_meas=eta_meas, info_Ls_eps=info_Ls_eps, cfg=cfg)

    # --- model ---
    model = PINO_model(cfg=cfg, x=x, t=t, list_buoy_indices=buoy_ind)

    # --- training ---
    model.training(epochs=cfg.epochs, train_loader=train_loader, val_loader=val_loader)

    # --- evaluation ---
    eta_pred_test_, phi_pred_test_, eta_meas_test_, eta_true_test_, phi_true_test_, info_test = \
        model.predict(loader=test_loader)

    # --- plotting ---
    SSP_vs_parameter_plot(path_save=model.FIGURE_PATH,
                          y_true=eta_true_test_, y_pred=eta_pred_test_,
                          param1=info_test[:, 0], param2=info_test[:, 1],
                          min_z=0.07, max_z=0.21)

    for i in [133, 134, 135, 136]:  # plot results for some samples from the  test set
        plot_epoch_results_eta_phi_specific_cross_sec(eta_in=eta_meas_test_[i], eta_out=eta_pred_test_[i],
                                                      phi_out=phi_pred_test_[i], eta_true=eta_true_test_[i],
                                                      phi_true=phi_true_test_[i], Lp=info_test[i, 0], eps=info_test[i, 1],
                                                      x_new=model.x_new, t_new=model.t_new, list_buoys=model.list_buoys,
                                                      path_save=model.FIGURE_PATH, space_margin=model.space_margin, num_plot=i)

    plotting_losscurve_name(path_loss=os.path.join(model.LOSS_FILE), path_save=os.path.join(model.FIGURE_PATH, f'loss'),
                            figsize=(5, 2.5), xmax=cfg.epochs+10)


if __name__ == "__main__":
    main()
