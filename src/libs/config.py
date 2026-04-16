import os

class Config:

    # --- physical parameters ---
    d = 500                 # water depth

    # --- training --
    batch_size = 8          # batch size
    epochs = 300            # max. training epochs
    lr = 0.001              # initial learning rate
    step_scheduler = 25     # step size for the learning rate scheduler
    early_stopping_at = 50  # epochs with no improvement for early stopping

    # --- model (FNO) ---
    fno_layers = 3          # layers of the FNO
    modes = 128             # modes in each FNO-layer
    width = 32              # latent width of the FNO

    # --- numerical / formulation ---
    tukey_alpha = 0.05      # alpha of the tukey window to create periodic boundary conditions for the HOSM-like formulation
    M = 4                   # order of the HOSM-like Taylor series expansion

    # --- loss weights ---
    w_data = 1.0            # loss weighting factor data term (at buoy positions)
    w_kin = 1.0             # loss weighting factor kinematic FSBC
    w_dyn = 1.0             # loss weighting factor dynamic FSBC
    w_reg = 0.25            # loss weighting factor kinematic regularization term

    # --- data ---
    num_buoys = 5                           # number of buoy measurements in the domain
    space_boundary_exclusion = 0.25         # fraction of the spatial domain excluded at each boundary when placing buoys (required as domain gets cropped due to tukey window tapering)
    time_boundary_exclusion = 0.04          # fraction of the time domain excluded at each boundary (required as domain gets cropped due to tukey window tapering)
    test_val_size = 0.4                     # fraction of test and validation set (later 50% for test sett and 50% for validation set)
    num_workers = min(6, os.cpu_count())    # number of workers for data loaders
    prefetch_factor = 2                     # prefetch factor for data loaders
    data_path = "../data/dataset_1056_samples_HOS_ele_pot_inte.npz"
    name_save = f'five_buoys_epochs_{epochs}_fno_layers_' + str(fno_layers) + '_modes_' + str(modes) + '_width_' + str(width)
