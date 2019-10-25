import os
import csv
import datetime
import random
import numpy as np
import torch


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def fix_random_seed(seed=46):
    """Fix seed id for training
    """
    print("Random Seed: ", seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare_logdir(log_path="log", descript=None):
    """
    Prepare log directory whose structure is shown in following.

    - root/ (root directory of this project)
         |-log/
            |- <descript>_<datetime>/
            |           |- components (e.g., checkpoint, results...)
            |           |- ...
            |- ...
    """

    # Prepare log_name based on current time
    now = datetime.datetime.now()
    save_path = now.isoformat()

    # Add description of this trial in log_dir_name
    if descript is not None and isinstance(descript, str):
        save_path = descript + "_" + save_path

    # Create "log" dir if not exist
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    # Create sub-dir ("<descrpt>_<datetime>") in "log" dir
    dir_name = os.path.join(log_path, save_path)
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    return dir_name


def save_arguments(args, save_path):
    """
    TODO: Refine this function based on the following code.
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/options/base_options.py
    """
    file_list = []
    for k, v in sorted(vars(args).items()):
        file_list.append("{}\t{}\n".format(k, v))
    with open(os.path.join(save_path, "arguments.txt"), "w") as f:
        f.writelines(file_list)
