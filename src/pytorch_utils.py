#!/usr/bin/env python3
# author:liufeng
# datetime:2021/3/26 10:20 AM
# software: PyCharm


import glob
import os

import torch

# TODO: clean


# Unused
# def init_weights(m, mean=0.0, std=0.01) -> None:
#     """init weights with normal_"""
#     classname = m.__class__.__name__
#     if classname.find("Conv") != -1:
#         m.weight.data.normal_(mean, std)


# Unused
# def get_padding(kernel_size, dilation=1):
#     return int((kernel_size * dilation - dilation) / 2)


# Unused
# def scan_last_checkpoint(cp_dir, prefix):
#     pattern = os.path.join(cp_dir, prefix + "????????")
#     cp_list = glob.glob(pattern)
#     if len(cp_list) == 0:
#         return None
#     return sorted(cp_list)[-1]


def scan_and_load_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + "????????")
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return None
    model_path = sorted(cp_list)[-1]
    checkpoint_dict = torch.load(
        model_path, map_location="cpu", weights_only=False
    )
    print(f"Loading {model_path}")
    return checkpoint_dict


def get_latest_model(hdf5_dir, prefix):
    model_path, last_epoch = None, 0
    for name in os.listdir(hdf5_dir):
        if name.startswith(prefix):
            epoch = int(name.replace("-", ".").replace("_", ".").split(".")[1])
            if epoch > last_epoch:
                last_epoch = epoch
                model_path = os.path.join(hdf5_dir, name)
    return model_path, last_epoch


def get_model_with_epoch(hdf5_dir, prefix, model_epoch):
    for name in os.listdir(hdf5_dir):
        if name.startswith(prefix):
            local_epoch = int(
                name.replace("-", ".").replace("_", ".").split(".")[1]
            )
            if local_epoch == model_epoch:
                return os.path.join(hdf5_dir, name)
    return None


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        # Prodigy stores its adaptive step-size estimate in "d";
        # "lr" is always 1.0 (a fixed scale factor, not the effective rate)
        return param_group.get("d", param_group["lr"])
    return None


# Unused
# def average_model(model_path_list, new_model_path) -> None:
#     print(model_path_list)
#     avg = None
#     num = len(model_path_list)
#     for path in model_path_list:
#         print(f"Processing {path}")
#         states = torch.load(path, map_location=torch.device("cpu"))
#         if avg is None:
#             avg = states
#         else:
#             for k in avg:
#                 avg[k] += states[k]

#     # average
#     for k in avg:
#         if avg[k] is not None:
#             avg[k] = torch.true_divide(avg[k], num)

#     print(f"Saving to {new_model_path}")
#     torch.save(avg, new_model_path)
