from torch import nn
import csv
import torch
import numpy as np
import random


def init_weight_bias(layer):
    """
    This function is intended to be passed to `nn.Module.apply()` and applies Xavier (Glorot) initialization to linear layers.
    Bias terms are initialized with a small positive constant to avoid dead neurons at initialization.
    :param layer: Layer of type `nn.Linear`.
    """

    if type(layer) == nn.Linear:
        nn.init.xavier_normal_(layer.weight)
        nn.init.constant_(layer.bias, 0.0001)


def write_csv_line(path, line):
    """
    Appends a single row of values to a CSV file. This function is used for logging training and validation metrics during training.
    If the file does not exist, it is created automatically.
    :param path: file path to the csv file
    :param line: list of values representing one row to be written
    """

    with open(path, 'a+', newline='') as write_obj:
        csv_writer = csv.writer(write_obj)
        csv_writer.writerow(line)


def set_torch_seed(seed=1234):
    """
    Sets random seeds for reproducible experiments across NumPy, PyTorch, and Python.
    :param seed: seed value used for all random number generators
    """

    # --- set seeds for all relevant libraries ---
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # --- ensure reproducibility on GPU ---
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # --- enforce deterministic behavior in cuDNN backend ---
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def torch_tensor_to_np(tensor):
    """
    Detaches a Pytorch tensor from the computational graph, transfers it to CPU memory, and converts it to a NumPy array.
    :param tensor: torch tensor
    :return: numpy representation of the input tensor
    """
    return tensor.detach().cpu().numpy()
