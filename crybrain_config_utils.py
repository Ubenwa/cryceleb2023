import os
import random

import numpy as np
import pandas as pd
import torch
from speechbrain.nnet.schedulers import CyclicLRScheduler, ReduceLROnPlateau


def choose_lrsched(lrsched_name, **kwargs):
    print(f"lrsched_name: {lrsched_name}")
    if lrsched_name == "onplateau":
        return ReduceLROnPlateau(
            lr_min=kwargs["lr_min"],
            factor=kwargs["factor"],
            patience=kwargs["patience"],
            dont_halve_until_epoch=kwargs["dont_halve_until_epoch"],
        )
    elif lrsched_name == "cyclic":
        return CyclicLRScheduler(
            base_lr=kwargs["base_lr"],
            max_lr=kwargs["max_lr"],
            step_size=kwargs["step_size"],
            mode=kwargs["mode"],
            gamma=kwargs["gamma"],
            scale_fn=kwargs["scale_fn"],
            scale_mode=kwargs["scale_mode"],
        )


def set_seed(seed):
    """Set seed in every way possible."""
    print(f"setting seeds to {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        print(f"setting cuda seeds to {seed}")
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def test_cuda_seed():
    """Print some random results from various libraries."""
    print(f"python random float: {random.random()}")
    print(f"numpy random int: {np.random.randint(100)}")
    print(f"torch random tensor (cpu): {torch.FloatTensor(100).uniform_()}")
    print(f"torch random tensor (cuda): {torch.cuda.FloatTensor(100).uniform_()}")


def get_n_classes(split_metadata_path, col_name):
    df = pd.read_csv(split_metadata_path)
    return df[col_name].nunique()
