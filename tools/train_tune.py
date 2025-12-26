"""
Created by @samuel-gauthier and @alanngnet in April-May 2024.

Tool to launch many sequential training runs for the purpose of discovering
optimal hyperparameter settings for a given dataset, aka "hyperparameter tuning."

"""

import glob, os, sys
import shutil, time, gc
import argparse
from datetime import date
import pprint
import torch
import torch.multiprocessing as mp
import numpy as np
import random
from collections import defaultdict
from src.trainer import Trainer
from src.model import Model
from src.utils import load_hparams, create_logger
from tensorboard.backend.event_processing.event_accumulator import (
    EventAccumulator,
)


def make_deterministic(seed):
    """
    @samuel-gauthier investigated non-deterministic training behavior on
    his CUDA platform which @alanngnet did not observe on his MPS platform.
    @samuel-gauthier reported: "my tests showed no variance with deterministic
    = true and benchmark = false, and variance if deterministic = false or
    benchmark = true". With benchmark = true, "I found myself with 3 or 4
    different results when launching 10 [training runs]." (quoted from
    correspondence with @alanngnet 29 May 2024). He arrived at this function's
    method of ensuring deterministic training on CUDA platforms.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def apply_early_stopping_config(hp, experiments):
    """
    Apply early stopping configuration from experiments to hp.
    
    Supports both val_loss (default) and mAP-based early stopping.
    """
    hp["every_n_epoch_to_save"] = 100
    hp["max_epochs"] = experiments.get("max_epochs", 15)
    
    # Early stopping mode
    early_stop_metric = experiments.get("early_stop_metric", "val_loss")
    hp["early_stop_metric"] = early_stop_metric
    
    if early_stop_metric == "mAP":
        hp["map_stopping_testsets"] = experiments.get("map_stopping_testsets", [])
        hp["map_stopping_patience"] = experiments.get("map_stopping_patience", 5)
        hp["map_smoothing_alpha"] = experiments.get("map_smoothing_alpha", 0.3)
        # val_loss patience still needed as fallback
        hp["early_stopping_patience"] = experiments.get("early_stopping_patience", 1000)
    else:
        hp["early_stopping_patience"] = experiments["early_stopping_patience"]


def get_slope_window(hp, experiments):
    """Return appropriate slope window based on early stopping mode."""
    if experiments.get("early_stop_metric") == "mAP":
        return experiments.get("map_stopping_patience", 5)
    return experiments["early_stopping_patience"]



def run_experiment(
    hp_summary,
    checkpoint_dir,
    hp,
    seed,
):
    hp["seed"] = seed
    make_deterministic(seed)
    # set multiprocessing method because 'fork'
    # has significant performance boost on MPS vs. default 'spawn'
    if torch.backends.mps.is_available():
        try:
            mp.set_start_method("fork")
        except RuntimeError:
            pass
    else:
        try:
            mp.set_start_method("spawn")
        except RuntimeError:
            pass

    log_path = os.path.join(
        model_dir,
        "logs",
        hp_summary + f"_seed_{seed}",
        today,
    )
    print("===========================================================")
    print(f"Running experiment with seed {seed}, {hp_summary}")
    os.makedirs(log_path, exist_ok=True)
    shutil.rmtree(checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    # must clear temp embeddings otherwise they will be reused for testsset metrics
    directories = glob.glob(os.path.join(model_dir, "embed_*_*"))
    for directory in directories:
        shutil.rmtree(directory)
    pprint.pprint(hp)

    t = Trainer(
        hp,
        Model,
        hp["device"],
        log_path,
        checkpoint_dir,
        model_dir,
        only_eval=False,
        first_eval=False,
    )

    t.configure_optimizer()
    t.load_model()
    t.configure_scheduler()
    t.train(max_epochs=hp["max_epochs"])
    # ensure log files are saved before retrieving metrics
    t.summary_writer.close()

    del t.model
    del t
    gc.collect()
    print(f"Completed experiment with seed {seed}")
    time.sleep(1)  # give OS time to save log file
    return log_path


def get_final_metrics_from_logs(log_dir, test_name, slope_window=3):
    """
    Extract training metrics from TensorBoard logs.

    Returns dict with:
        - val_loss: final validation focal loss
        - val_loss_slope: slope over last slope_window epochs
                          (negative = improving, ~0 = converged, positive = overfitting)
        - map: final mAP on test_name
        - map_slope: slope of mAP over last slope_window epochs
                     (positive = improving, ~0 = converged, negative = degrading)
    """
    event_file = max(
        glob.glob(os.path.join(log_dir, "events.out.tfevents.*")),
        key=os.path.getctime,
    )
    ea = EventAccumulator(event_file)
    ea.Reload()

    val_loss_series = ea.Scalars("csi_val/foc_loss")
    map_series = ea.Scalars(f"mAP/{test_name}")

    def calc_slope(series, window):
        if len(series) < 2:
            return 0.0
        window = min(window, len(series))
        x = np.array([s.step for s in series[-window:]], dtype=float)
        y = np.array([s.value for s in series[-window:]], dtype=float)
        x_mean, y_mean = x.mean(), y.mean()
        return ((x - x_mean) * (y - y_mean)).sum() / ((x - x_mean) ** 2).sum()

    val_loss = val_loss_series[-1].value
    val_loss_slope = calc_slope(val_loss_series, slope_window)
    final_map = map_series[-1].value
    map_slope = calc_slope(map_series, slope_window)

    print(
        f"val_loss={val_loss:.4f} (slope={val_loss_slope:+.4f}), "
        f"mAP={final_map:.4f} (slope={map_slope:+.4f})"
    )

    return {
        "val_loss": val_loss,
        "val_loss_slope": val_loss_slope,
        "map": final_map,
        "map_slope": map_slope,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train-tune: python3 -m tools.train-tune model_dir",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("model_dir")
    args = parser.parse_args()
    model_dir = args.model_dir
    checkpoint_dir = os.path.join(model_dir, "checkpoints")
    experiments = load_hparams(
        os.path.join(model_dir, "config/hp_tuning.yaml")
    )
    hp = load_hparams(os.path.join(model_dir, "config/hparams.yaml"))
    test_name = experiments["test_name"]
    # ensure at least one seed
    seeds = experiments.get("seeds", [hp["seed"]])
    chunk_frames = experiments["chunk_frames"]
    # ensure at least one mean_size
    mean_sizes = experiments.get("mean_sizes", [hp["mean_size"]])
    m_per_classes = experiments["m_per_classes"]
    num_blockss = experiments["num_blockss"]
    spec_augmentations = experiments["spec_augmentations"]
    losses = experiments["losses"]
    learning_rates = experiments["learning_rates"]
    lr_decays = experiments["lr_decays"]
    adam_betas = experiments["adam_betas"]

    os.makedirs(checkpoint_dir, exist_ok=True)
    logger = create_logger()
    today = date.today().strftime("%Y-%m-%d")

    match hp["device"]:  # noqa requires python 3.10
        case "mps":
            if not torch.backends.mps.is_available():
                logger.error(
                    "You requested 'mps' device in your hyperparameters"
                    "but you are not running on an Apple M-series chip or "
                    "have not compiled PyTorch for MPS support."
                )
                sys.exit()
            device = torch.device("mps")
        case "cuda":
            if not torch.cuda.is_available():
                logger.error(
                    "You requested 'cuda' device in your hyperparameters"
                    "but you do not have a CUDA-compatible GPU available."
                )
                sys.exit()
            device = torch.device("cuda")
        case _:
            logger.error(
                "You set device: %s in your hyperparameters but that is "
                "not a valid option or is an untested option.",
                hp["device"],
            )
            sys.exit()

    all_results = {}

    # chunk_frame experiments

# chunk_frame experiments
    apply_early_stopping_config(hp, experiments)
    results = defaultdict(list)
    for chunk_frame in chunk_frames:
        hp["chunk_frame"] = chunk_frame
        for mean_size in mean_sizes:
            hp["mean_size"] = mean_size
            hp["chunk_s"] = chunk_frame[0] * mean_size / 25
            for seed in seeds:
                hp_summary = (
                    "chunk_frame"
                    + "_".join([str(c) for c in chunk_frame])
                    + f"_mean_size{mean_size}"
                )
                log_path = run_experiment(hp_summary, checkpoint_dir, hp, seed)
                metrics = get_final_metrics_from_logs(
                    log_path, test_name, get_slope_window(hp, experiments)
                )
                for key, value in metrics.items():
                    results[key].append(value)
            all_results[hp_summary] = {
                key: {"mean": np.mean(vals), "std": np.std(vals)}
                for key, vals in results.items()
            }
            results.clear()
            print(f"Results for {hp_summary}")
            pprint.pprint(all_results[hp_summary])

    # m_per_class experiments
    hp = load_hparams(os.path.join(model_dir, "config/hparams.yaml"))
    apply_early_stopping_config(hp, experiments)
    results = defaultdict(list)
    for m_per_class in m_per_classes:
        hp["m_per_class"] = m_per_class
        for seed in seeds:
            hp_summary = f"m_per_class{m_per_class}"
            log_path = run_experiment(hp_summary, checkpoint_dir, hp, seed)
            metrics = get_final_metrics_from_logs(
                log_path, test_name, get_slope_window(hp, experiments)
            )
            for key, value in metrics.items():
                results[key].append(value)
        all_results[hp_summary] = {
            key: {"mean": np.mean(vals), "std": np.std(vals)}
            for key, vals in results.items()
        }
        results.clear()
        print(f"Results for {hp_summary}")
        pprint.pprint(all_results[hp_summary])

    # num_blocks experiments
    hp = load_hparams(os.path.join(model_dir, "config/hparams.yaml"))
    apply_early_stopping_config(hp, experiments)
    results = defaultdict(list)
    for num_blocks in num_blockss:
        hp["encoder"]["num_blocks"] = num_blocks
        for seed in seeds:
            hp_summary = f"num_blocks{num_blocks}"
            log_path = run_experiment(hp_summary, checkpoint_dir, hp, seed)
            metrics = get_final_metrics_from_logs(
                log_path, test_name, get_slope_window(hp, experiments)
            )
            for key, value in metrics.items():
                results[key].append(value)
        all_results[hp_summary] = {
            key: {"mean": np.mean(vals), "std": np.std(vals)}
            for key, vals in results.items()
        }
        results.clear()
        print(f"Results for {hp_summary}")
        pprint.pprint(all_results[hp_summary])

    # spec_aug experiments
    hp = load_hparams(os.path.join(model_dir, "config/hparams.yaml"))
    apply_early_stopping_config(hp, experiments)
    results = defaultdict(list)
    for spec_augmentation in spec_augmentations:
        hp["spec_augmentation"] = spec_augmentation
        random_erase_prob = spec_augmentation["random_erase"]["prob"]
        random_erase_num = spec_augmentation["random_erase"]["erase_num"]
        region_size = spec_augmentation["random_erase"]["region_size"]
        roll_pitch_prob = spec_augmentation["roll_pitch"]["prob"]
        roll_pitch_shift_num = spec_augmentation["roll_pitch"]["shift_num"]
        roll_pitch_method = spec_augmentation["roll_pitch"]["method"]

        for seed in seeds:
            hp_summary = (
                f"erase_prob{random_erase_prob}_num{random_erase_num}_size"
                + "_".join([str(c) for c in region_size])
                + f"_roll_prob{roll_pitch_prob}_shift{roll_pitch_shift_num}"
                + f"_{roll_pitch_method}"
            )
            log_path = run_experiment(hp_summary, checkpoint_dir, hp, seed)
            metrics = get_final_metrics_from_logs(
                log_path, test_name, get_slope_window(hp, experiments)
            )
            for key, value in metrics.items():
                results[key].append(value)
        all_results[hp_summary] = {
            key: {"mean": np.mean(vals), "std": np.std(vals)}
            for key, vals in results.items()
        }
        results.clear()
        print(f"Results for {hp_summary}")
        pprint.pprint(all_results[hp_summary])

    # loss experiments
    hp = load_hparams(os.path.join(model_dir, "config/hparams.yaml"))
    apply_early_stopping_config(hp, experiments)
    results = defaultdict(list)
    for loss in losses:
        hp["foc"] = foc = loss["foc"]
        foc_dims = foc["output_dims"]
        foc_weight = foc["weight"]
        foc_gamma = foc["gamma"]
        hp["triplet"] = triplet = loss["triplet"]
        triplet_margin = triplet["margin"]
        triplet_weight = triplet["weight"]
        hp["center"] = center = loss["center"]
        center_weight = center["weight"]

        for seed in seeds:
            hp_summary = (
                f"FOC_dims{foc_dims}_wt{foc_weight}_gamma{foc_gamma}_"
                + f"TRIP_marg{triplet_margin}_wt{triplet_weight}_"
                + f"CNTR_wt{center_weight}"
            )
            log_path = run_experiment(hp_summary, checkpoint_dir, hp, seed)
            metrics = get_final_metrics_from_logs(
                log_path, test_name, get_slope_window(hp, experiments)
            )
            for key, value in metrics.items():
                results[key].append(value)
        all_results[hp_summary] = {
            key: {"mean": np.mean(vals), "std": np.std(vals)}
            for key, vals in results.items()
        }
        results.clear()
        print(f"Results for {hp_summary}")
        pprint.pprint(all_results[hp_summary])

    # learning_rate experiments
    hp = load_hparams(os.path.join(model_dir, "config/hparams.yaml"))
    apply_early_stopping_config(hp, experiments)
    results = defaultdict(list)
    for learning_rate in learning_rates:
        hp["learning_rate"] = learning_rate
        for seed in seeds:
            hp_summary = f"lrate{learning_rate}"
            log_path = run_experiment(hp_summary, checkpoint_dir, hp, seed)
            metrics = get_final_metrics_from_logs(
                log_path, test_name, get_slope_window(hp, experiments)
            )
            for key, value in metrics.items():
                results[key].append(value)
        all_results[hp_summary] = {
            key: {"mean": np.mean(vals), "std": np.std(vals)}
            for key, vals in results.items()
        }
        results.clear()
        print(f"Results for {hp_summary}")
        pprint.pprint(all_results[hp_summary])

    # lr_decay experiments
    hp = load_hparams(os.path.join(model_dir, "config/hparams.yaml"))
    apply_early_stopping_config(hp, experiments)
    results = defaultdict(list)
    for lr_decay in lr_decays:
        hp["lr_decay"] = lr_decay
        for seed in seeds:
            hp_summary = f"lr_decay{lr_decay}"
            log_path = run_experiment(hp_summary, checkpoint_dir, hp, seed)
            metrics = get_final_metrics_from_logs(
                log_path, test_name, get_slope_window(hp, experiments)
            )
            for key, value in metrics.items():
                results[key].append(value)
        all_results[hp_summary] = {
            key: {"mean": np.mean(vals), "std": np.std(vals)}
            for key, vals in results.items()
        }
        results.clear()
        print(f"Results for {hp_summary}")
        pprint.pprint(all_results[hp_summary])

    # AdamW betas experiments
    hp = load_hparams(os.path.join(model_dir, "config/hparams.yaml"))
    apply_early_stopping_config(hp, experiments)
    results = defaultdict(list)
    for adam_beta in adam_betas:
        hp["adam_b1"] = adam_beta[0]
        hp["adam_b2"] = adam_beta[1]
        for seed in seeds:
            hp_summary = "adam_betas" + "_".join([str(c) for c in adam_beta])
            log_path = run_experiment(hp_summary, checkpoint_dir, hp, seed)
            metrics = get_final_metrics_from_logs(
                log_path, test_name, get_slope_window(hp, experiments)
            )
            for key, value in metrics.items():
                results[key].append(value)
        all_results[hp_summary] = {
            key: {"mean": np.mean(vals), "std": np.std(vals)}
            for key, vals in results.items()
        }
        results.clear()
        print(f"Results for {hp_summary}")
        pprint.pprint(all_results[hp_summary])

    print("\nSummary of Experiments:")
    for hp_summary, result in all_results.items():
        print(f"\nExperiment: {hp_summary}")
        print(
            f"  val_loss: {result['val_loss']['mean']:.4f}±{result['val_loss']['std']:.4f}  "
            f"(slope: {result['val_loss_slope']['mean']:+.4f})"
        )
        print(
            f"  mAP:      {result['map']['mean']:.4f}±{result['map']['std']:.4f}  "
            f"(slope: {result['map_slope']['mean']:+.4f})"
        )
