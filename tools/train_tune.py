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
from src.trainer import Trainer, TrainingDivergedException
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
        hp["map_stopping_testsets"] = experiments.get(
            "map_stopping_testsets", []
        )
        hp["map_smoothing_alpha"] = experiments.get("map_smoothing_alpha", 0.3)
    hp["early_stopping_patience"] = experiments["early_stopping_patience"]


def apply_hp_overrides(hp, experiments):
    """
    Apply optional per-key overrides from hp_tuning.yaml onto hp.
    Any key in the hp_overrides block with a non-null value replaces
    the corresponding value loaded from hparams.yaml. Keys set to null
    are ignored, leaving the hparams.yaml value intact.
    """

    def deep_merge(base, override):
        for key, val in override.items():
            if val is None:
                continue
            if isinstance(val, dict) and isinstance(base.get(key), dict):
                deep_merge(base[key], val)
            else:
                base[key] = val

    overrides = experiments.get("hp_overrides", {}) or {}
    deep_merge(hp, overrides)


def get_slope_window(hp, experiments):
    """Return slope window for summary statistics, matching the patience used by Trainer."""
    return experiments["early_stopping_patience"]


def run_experiment(
    hp_summary,
    checkpoint_dir,
    hp,
    seed,
    foundation_checkpoint=None,
    restore_optimizer_state=False,
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

    if foundation_checkpoint is not None:
        dest = os.path.join(
            checkpoint_dir, os.path.basename(foundation_checkpoint)
        )
        shutil.copy2(foundation_checkpoint, dest)
        if restore_optimizer_state:
            g_basename = os.path.basename(foundation_checkpoint)
            do_basename = g_basename.replace("g_", "do_", 1)
            do_source = os.path.join(
                os.path.dirname(foundation_checkpoint), do_basename
            )
            if os.path.exists(do_source):
                shutil.copy2(
                    do_source,
                    os.path.join(checkpoint_dir, do_basename),
                )
                print(
                    f"Seeding from checkpoint: {foundation_checkpoint} (with optimizer state)"
                )
            else:
                print(
                    f"WARNING: restore_optimizer_state is true but {do_source} "
                    f"not found. Optimizer will start cold."
                )
        else:
            print(
                f"Seeding from checkpoint: {foundation_checkpoint} (weights only)"
            )

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
    if restore_optimizer_state and foundation_checkpoint is not None:
        # Warm Adam moments were restored from the do_ file, but the
        # checkpoint's epoch/step counters and early stopping state must
        # be reset so the experiment gets its full max_epochs budget and
        # fresh stopping patience.
        t.step = 1
        t.epoch = -1
        t.reset_early_stopping_state()
        t._pending_scheduler_step_count = 0
        # Override the checkpoint's saved LR with the experiment's desired LR
        # so that the sweep variable or hp_overrides value takes effect, not
        # the LR the foundation checkpoint happened to stop at.
        for pg in t.optimizer.param_groups:
            pg["lr"] = hp["learning_rate"]
            pg["initial_lr"] = hp["learning_rate"]
    t.configure_scheduler()

    t.configure_scheduler()
    diverged = False
    try:
        t.train(max_epochs=hp["max_epochs"])
    except TrainingDivergedException as exc:
        print(f"  *** DIVERGED: {exc} ***")
        diverged = True
    # ensure log files are saved before retrieving metrics
    t.summary_writer.close()

    bootstrap_results = getattr(t, "last_bootstrap_results", {})
    del t.model
    del t
    gc.collect()
    if diverged:
        print(
            f"  Experiment {hp_summary} seed {seed} aborted due to divergence"
        )
    else:
        print(f"Completed experiment with seed {seed}")
    time.sleep(1)  # give OS time to save log file
    return log_path, bootstrap_results, diverged


def get_final_metrics_from_logs(
    log_dir, test_name, slope_window=3, benchmark_testsets=None
):
    """
    Extract training metrics from TensorBoard logs.

        - val_loss_slope: slope over last slope_window epochs
        - map: final mAP on test_name
        - map_slope: slope of mAP over last slope_window epochs
                     (positive = improving, ~0 = converged, negative = degrading)
    Returns flat scalar dict with:
        - val_loss, val_loss_slope: final validation focal loss and its trend
            (slope: negative = improving, ~0 = converged, positive = overfitting)
        - map, map_slope: final mAP on test_name and its trend
            (slope: positive = improving, ~0 = converged, negative = degrading)
        - bmap_{name}: final mAP for each benchmark testset (if benchmark_testsets provided)
    All values are plain floats suitable for np.mean/np.std accumulation across seeds.
    Bootstrap CI data is returned separately from run_experiment() to avoid TensorBoard writes.
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

    # Best (peak) values and their epochs
    best_val_loss_entry = min(val_loss_series, key=lambda s: s.value)
    best_val_loss = best_val_loss_entry.value
    best_val_loss_epoch = best_val_loss_entry.step
    best_map_entry = max(map_series, key=lambda s: s.value)
    best_map = best_map_entry.value
    best_map_epoch = best_map_entry.step

    print(
        f"val_loss={val_loss:.4f} (slope={val_loss_slope:+.4f}), "
        f"best={best_val_loss:.4f}@epoch{best_val_loss_epoch}, "
        f"mAP={final_map:.4f} (slope={map_slope:+.4f}), "
        f"best={best_map:.4f}@epoch{best_map_epoch}"
    )

    result = {
        "val_loss": val_loss,
        "val_loss_slope": val_loss_slope,
        "best_val_loss": best_val_loss,
        "best_val_loss_epoch": best_val_loss_epoch,
        "map": final_map,
        "map_slope": map_slope,
        "best_map": best_map,
        "best_map_epoch": best_map_epoch,
    }

    for bname in benchmark_testsets or []:
        try:
            bseries = ea.Scalars(f"mAP/{bname}")
            result[f"bmap_{bname}"] = bseries[-1].value
        except KeyError:
            pass  # testset not evaluated in this run (e.g. every_n_epoch_to_test skipped it)
    return result


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
    foundation_checkpoint = experiments.get("foundation_checkpoint", None)
    restore_optimizer_state = experiments.get("restore_optimizer_state", False)
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
    optimizers = experiments.get("optimizers", ["adamw"])
    benchmark_testsets = hp.get("benchmark_testsets", [])

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
    apply_early_stopping_config(hp, experiments)
    results = defaultdict(list)
    for chunk_frame in chunk_frames:
        hp["chunk_frame"] = chunk_frame
        for mean_size in mean_sizes:
            hp["mean_size"] = mean_size
            hp["chunk_s"] = chunk_frame[0] * mean_size / 25
            n_diverged = 0
            for seed in seeds:
                hp_summary = (
                    "chunk_frame"
                    + "_".join([str(c) for c in chunk_frame])
                    + f"_mean_size{mean_size}"
                )
                log_path, bootstrap_results, diverged = run_experiment(
                    hp_summary,
                    checkpoint_dir,
                    hp,
                    seed,
                    foundation_checkpoint,
                    restore_optimizer_state,
                )
                if diverged:
                    print(
                        f"  *** {hp_summary} seed {seed}: DIVERGED — skipping seed ***"
                    )
                    n_diverged += 1
                    continue
                metrics = get_final_metrics_from_logs(
                    log_path,
                    test_name,
                    get_slope_window(hp, experiments),
                    benchmark_testsets=benchmark_testsets,
                )
                # Flatten bootstrap CIs into metrics for unified mean/std
                # accumulation across seeds.
                # Keyed as bci_{testset}_{ci_low|ci_high|std}
                for ts_name, boot_ci in bootstrap_results.items():
                    for ci_key in ("ci_low", "ci_high", "std"):
                        val = boot_ci.get("mean_ap", {}).get(ci_key)
                        if val is not None:
                            metrics[f"bci_{ts_name}_{ci_key}"] = val
                for key, value in metrics.items():
                    results[key].append(value)
            if not results:
                all_results[hp_summary] = {
                    "__all_diverged__": True,
                    "__n_seeds__": len(seeds),
                }
            else:
                all_results[hp_summary] = {
                    key: {"mean": np.mean(vals), "std": np.std(vals)}
                    for key, vals in results.items()
                }
                if n_diverged:
                    all_results[hp_summary]["__n_diverged__"] = n_diverged
                    all_results[hp_summary]["__n_seeds__"] = len(seeds)
            results.clear()
            print(f"Results for {hp_summary}")
            pprint.pprint(all_results[hp_summary])

    # m_per_class experiments
    hp = load_hparams(os.path.join(model_dir, "config/hparams.yaml"))
    apply_hp_overrides(hp, experiments)
    apply_early_stopping_config(hp, experiments)
    results = defaultdict(list)
    for m_per_class in m_per_classes:
        hp["m_per_class"] = m_per_class
        n_diverged = 0
        for seed in seeds:
            hp_summary = f"m_per_class{m_per_class}"
            log_path, bootstrap_results, diverged = run_experiment(
                hp_summary,
                checkpoint_dir,
                hp,
                seed,
                foundation_checkpoint,
                restore_optimizer_state,
            )
            if diverged:
                print(
                    f"  *** {hp_summary} seed {seed}: DIVERGED — skipping seed ***"
                )
                n_diverged += 1
                continue
            metrics = get_final_metrics_from_logs(
                log_path,
                test_name,
                get_slope_window(hp, experiments),
                benchmark_testsets=benchmark_testsets,
            )
            # Flatten bootstrap CIs into metrics for unified mean/std
            # accumulation across seeds.
            # Keyed as bci_{testset}_{ci_low|ci_high|std}
            for ts_name, boot_ci in bootstrap_results.items():
                for ci_key in ("ci_low", "ci_high", "std"):
                    val = boot_ci.get("mean_ap", {}).get(ci_key)
                    if val is not None:
                        metrics[f"bci_{ts_name}_{ci_key}"] = val
            for key, value in metrics.items():
                results[key].append(value)
        if not results:
            all_results[hp_summary] = {
                "__all_diverged__": True,
                "__n_seeds__": len(seeds),
            }
        else:
            all_results[hp_summary] = {
                key: {"mean": np.mean(vals), "std": np.std(vals)}
                for key, vals in results.items()
            }
            if n_diverged:
                all_results[hp_summary]["__n_diverged__"] = n_diverged
                all_results[hp_summary]["__n_seeds__"] = len(seeds)
        results.clear()
        print(f"Results for {hp_summary}")
        pprint.pprint(all_results[hp_summary])

    # num_blocks experiments
    hp = load_hparams(os.path.join(model_dir, "config/hparams.yaml"))
    apply_hp_overrides(hp, experiments)
    apply_early_stopping_config(hp, experiments)
    results = defaultdict(list)
    for num_blocks in num_blockss:
        hp["encoder"]["num_blocks"] = num_blocks
        n_diverged = 0
        for seed in seeds:
            hp_summary = f"num_blocks{num_blocks}"
            log_path, bootstrap_results, diverged = run_experiment(
                hp_summary,
                checkpoint_dir,
                hp,
                seed,
                foundation_checkpoint,
                restore_optimizer_state,
            )
            if diverged:
                print(
                    f"  *** {hp_summary} seed {seed}: DIVERGED — skipping seed ***"
                )
                n_diverged += 1
                continue
            metrics = get_final_metrics_from_logs(
                log_path,
                test_name,
                get_slope_window(hp, experiments),
                benchmark_testsets=benchmark_testsets,
            )
            # Flatten bootstrap CIs into metrics for unified mean/std
            # accumulation across seeds.
            # Keyed as bci_{testset}_{ci_low|ci_high|std}
            for ts_name, boot_ci in bootstrap_results.items():
                for ci_key in ("ci_low", "ci_high", "std"):
                    val = boot_ci.get("mean_ap", {}).get(ci_key)
                    if val is not None:
                        metrics[f"bci_{ts_name}_{ci_key}"] = val
            for key, value in metrics.items():
                results[key].append(value)
        if not results:
            all_results[hp_summary] = {
                "__all_diverged__": True,
                "__n_seeds__": len(seeds),
            }
        else:
            all_results[hp_summary] = {
                key: {"mean": np.mean(vals), "std": np.std(vals)}
                for key, vals in results.items()
            }
            if n_diverged:
                all_results[hp_summary]["__n_diverged__"] = n_diverged
                all_results[hp_summary]["__n_seeds__"] = len(seeds)
        results.clear()
        print(f"Results for {hp_summary}")
        pprint.pprint(all_results[hp_summary])

    # spec_aug experiments
    hp = load_hparams(os.path.join(model_dir, "config/hparams.yaml"))
    apply_hp_overrides(hp, experiments)
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
        n_diverged = 0
        for seed in seeds:
            hp_summary = (
                f"erase_prob{random_erase_prob}_num{random_erase_num}_size"
                + "_".join([str(c) for c in region_size])
                + f"_roll_prob{roll_pitch_prob}_shift{roll_pitch_shift_num}"
                + f"_{roll_pitch_method}"
            )
            log_path, bootstrap_results, diverged = run_experiment(
                hp_summary,
                checkpoint_dir,
                hp,
                seed,
                foundation_checkpoint,
                restore_optimizer_state,
            )
            if diverged:
                print(
                    f"  *** {hp_summary} seed {seed}: DIVERGED — skipping seed ***"
                )
                n_diverged += 1
                continue
            metrics = get_final_metrics_from_logs(
                log_path,
                test_name,
                get_slope_window(hp, experiments),
                benchmark_testsets=benchmark_testsets,
            )
            # Flatten bootstrap CIs into metrics for unified mean/std
            # accumulation across seeds.
            # Keyed as bci_{testset}_{ci_low|ci_high|std}
            for ts_name, boot_ci in bootstrap_results.items():
                for ci_key in ("ci_low", "ci_high", "std"):
                    val = boot_ci.get("mean_ap", {}).get(ci_key)
                    if val is not None:
                        metrics[f"bci_{ts_name}_{ci_key}"] = val
            for key, value in metrics.items():
                results[key].append(value)
        if not results:
            all_results[hp_summary] = {
                "__all_diverged__": True,
                "__n_seeds__": len(seeds),
            }
        else:
            all_results[hp_summary] = {
                key: {"mean": np.mean(vals), "std": np.std(vals)}
                for key, vals in results.items()
            }
            if n_diverged:
                all_results[hp_summary]["__n_diverged__"] = n_diverged
                all_results[hp_summary]["__n_seeds__"] = len(seeds)
        results.clear()
        print(f"Results for {hp_summary}")
        pprint.pprint(all_results[hp_summary])

    # loss experiments
    hp = load_hparams(os.path.join(model_dir, "config/hparams.yaml"))
    apply_hp_overrides(hp, experiments)
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
        n_diverged = 0
        for seed in seeds:
            hp_summary = (
                f"FOC_dims{foc_dims}_wt{foc_weight}_gamma{foc_gamma}_"
                + f"TRIP_marg{triplet_margin}_wt{triplet_weight}_"
                + f"CNTR_wt{center_weight}"
            )
            log_path, bootstrap_results, diverged = run_experiment(
                hp_summary,
                checkpoint_dir,
                hp,
                seed,
                foundation_checkpoint,
                restore_optimizer_state,
            )
            if diverged:
                print(
                    f"  *** {hp_summary} seed {seed}: DIVERGED — skipping seed ***"
                )
                n_diverged += 1
                continue
            metrics = get_final_metrics_from_logs(
                log_path,
                test_name,
                get_slope_window(hp, experiments),
                benchmark_testsets=benchmark_testsets,
            )
            # Flatten bootstrap CIs into metrics for unified mean/std
            # accumulation across seeds.
            # Keyed as bci_{testset}_{ci_low|ci_high|std}
            for ts_name, boot_ci in bootstrap_results.items():
                for ci_key in ("ci_low", "ci_high", "std"):
                    val = boot_ci.get("mean_ap", {}).get(ci_key)
                    if val is not None:
                        metrics[f"bci_{ts_name}_{ci_key}"] = val
            for key, value in metrics.items():
                results[key].append(value)
        if not results:
            all_results[hp_summary] = {
                "__all_diverged__": True,
                "__n_seeds__": len(seeds),
            }
        else:
            all_results[hp_summary] = {
                key: {"mean": np.mean(vals), "std": np.std(vals)}
                for key, vals in results.items()
            }
            if n_diverged:
                all_results[hp_summary]["__n_diverged__"] = n_diverged
                all_results[hp_summary]["__n_seeds__"] = len(seeds)
        results.clear()
        print(f"Results for {hp_summary}")
        pprint.pprint(all_results[hp_summary])

    # learning_rate experiments
    hp = load_hparams(os.path.join(model_dir, "config/hparams.yaml"))
    apply_hp_overrides(hp, experiments)
    apply_early_stopping_config(hp, experiments)
    results = defaultdict(list)
    for learning_rate in learning_rates:
        hp["learning_rate"] = learning_rate
        n_diverged = 0
        for seed in seeds:
            hp_summary = f"lrate{learning_rate}"
            log_path, bootstrap_results, diverged = run_experiment(
                hp_summary,
                checkpoint_dir,
                hp,
                seed,
                foundation_checkpoint,
                restore_optimizer_state,
            )
            if diverged:
                print(
                    f"  *** {hp_summary} seed {seed}: DIVERGED — skipping seed ***"
                )
                n_diverged += 1
                continue
            metrics = get_final_metrics_from_logs(
                log_path,
                test_name,
                get_slope_window(hp, experiments),
                benchmark_testsets=benchmark_testsets,
            )
            # Flatten bootstrap CIs into metrics for unified mean/std
            # accumulation across seeds.
            # Keyed as bci_{testset}_{ci_low|ci_high|std}
            for ts_name, boot_ci in bootstrap_results.items():
                for ci_key in ("ci_low", "ci_high", "std"):
                    val = boot_ci.get("mean_ap", {}).get(ci_key)
                    if val is not None:
                        metrics[f"bci_{ts_name}_{ci_key}"] = val
            for key, value in metrics.items():
                results[key].append(value)
        if not results:
            all_results[hp_summary] = {
                "__all_diverged__": True,
                "__n_seeds__": len(seeds),
            }
        else:
            all_results[hp_summary] = {
                key: {"mean": np.mean(vals), "std": np.std(vals)}
                for key, vals in results.items()
            }
            if n_diverged:
                all_results[hp_summary]["__n_diverged__"] = n_diverged
                all_results[hp_summary]["__n_seeds__"] = len(seeds)
        results.clear()
        print(f"Results for {hp_summary}")
        pprint.pprint(all_results[hp_summary])

    # lr_decay experiments
    hp = load_hparams(os.path.join(model_dir, "config/hparams.yaml"))
    apply_hp_overrides(hp, experiments)
    apply_early_stopping_config(hp, experiments)
    results = defaultdict(list)
    for lr_decay in lr_decays:
        hp["lr_decay"] = lr_decay
        n_diverged = 0
        for seed in seeds:
            hp_summary = f"lr_decay{lr_decay}"
            log_path, bootstrap_results, diverged = run_experiment(
                hp_summary,
                checkpoint_dir,
                hp,
                seed,
                foundation_checkpoint,
                restore_optimizer_state,
            )
            if diverged:
                print(
                    f"  *** {hp_summary} seed {seed}: DIVERGED — skipping seed ***"
                )
                n_diverged += 1
                continue
            metrics = get_final_metrics_from_logs(
                log_path,
                test_name,
                get_slope_window(hp, experiments),
                benchmark_testsets=benchmark_testsets,
            )
            # Flatten bootstrap CIs into metrics for unified mean/std
            # accumulation across seeds.
            # Keyed as bci_{testset}_{ci_low|ci_high|std}
            for ts_name, boot_ci in bootstrap_results.items():
                for ci_key in ("ci_low", "ci_high", "std"):
                    val = boot_ci.get("mean_ap", {}).get(ci_key)
                    if val is not None:
                        metrics[f"bci_{ts_name}_{ci_key}"] = val
            for key, value in metrics.items():
                results[key].append(value)
        if not results:
            all_results[hp_summary] = {
                "__all_diverged__": True,
                "__n_seeds__": len(seeds),
            }
        else:
            all_results[hp_summary] = {
                key: {"mean": np.mean(vals), "std": np.std(vals)}
                for key, vals in results.items()
            }
            if n_diverged:
                all_results[hp_summary]["__n_diverged__"] = n_diverged
                all_results[hp_summary]["__n_seeds__"] = len(seeds)

        results.clear()
        print(f"Results for {hp_summary}")
        pprint.pprint(all_results[hp_summary])

    # AdamW betas experiments
    hp = load_hparams(os.path.join(model_dir, "config/hparams.yaml"))
    apply_hp_overrides(hp, experiments)
    apply_early_stopping_config(hp, experiments)
    results = defaultdict(list)
    for adam_beta in adam_betas:
        hp["adam_b1"] = adam_beta[0]
        hp["adam_b2"] = adam_beta[1]
        n_diverged = 0
        for seed in seeds:
            hp_summary = "adam_betas" + "_".join([str(c) for c in adam_beta])
            log_path, bootstrap_results, diverged = run_experiment(
                hp_summary,
                checkpoint_dir,
                hp,
                seed,
                foundation_checkpoint,
                restore_optimizer_state,
            )
            if diverged:
                print(
                    f"  *** {hp_summary} seed {seed}: DIVERGED — skipping seed ***"
                )
                n_diverged += 1
                continue
            metrics = get_final_metrics_from_logs(
                log_path,
                test_name,
                get_slope_window(hp, experiments),
                benchmark_testsets=benchmark_testsets,
            )
            # Flatten bootstrap CIs into metrics for unified mean/std
            # accumulation across seeds.
            # Keyed as bci_{testset}_{ci_low|ci_high|std}
            for ts_name, boot_ci in bootstrap_results.items():
                for ci_key in ("ci_low", "ci_high", "std"):
                    val = boot_ci.get("mean_ap", {}).get(ci_key)
                    if val is not None:
                        metrics[f"bci_{ts_name}_{ci_key}"] = val
            for key, value in metrics.items():
                results[key].append(value)
        if not results:
            all_results[hp_summary] = {
                "__all_diverged__": True,
                "__n_seeds__": len(seeds),
            }
        else:
            all_results[hp_summary] = {
                key: {"mean": np.mean(vals), "std": np.std(vals)}
                for key, vals in results.items()
            }
            if n_diverged:
                all_results[hp_summary]["__n_diverged__"] = n_diverged
                all_results[hp_summary]["__n_seeds__"] = len(seeds)
        results.clear()
        print(f"Results for {hp_summary}")
        pprint.pprint(all_results[hp_summary])

    # optimizer experiments
    hp = load_hparams(os.path.join(model_dir, "config/hparams.yaml"))
    apply_hp_overrides(hp, experiments)
    apply_early_stopping_config(hp, experiments)
    results = defaultdict(list)
    for optimizer_name in optimizers:
        hp["optimizer"] = optimizer_name
        n_diverged = 0
        for seed in seeds:
            hp_summary = f"optimizer_{optimizer_name}"
            log_path, bootstrap_results, diverged = run_experiment(
                hp_summary,
                checkpoint_dir,
                hp,
                seed,
                foundation_checkpoint,
                restore_optimizer_state,
            )
            if diverged:
                print(
                    f"  *** {hp_summary} seed {seed}: DIVERGED — skipping seed ***"
                )
                n_diverged += 1
                continue
            metrics = get_final_metrics_from_logs(
                log_path,
                test_name,
                get_slope_window(hp, experiments),
                benchmark_testsets=benchmark_testsets,
            )
            for ts_name, boot_ci in bootstrap_results.items():
                for ci_key in ("ci_low", "ci_high", "std"):
                    val = boot_ci.get("mean_ap", {}).get(ci_key)
                    if val is not None:
                        metrics[f"bci_{ts_name}_{ci_key}"] = val
            for key, value in metrics.items():
                results[key].append(value)
        if not results:
            all_results[hp_summary] = {
                "__all_diverged__": True,
                "__n_seeds__": len(seeds),
            }
        else:
            all_results[hp_summary] = {
                key: {"mean": np.mean(vals), "std": np.std(vals)}
                for key, vals in results.items()
            }
            if n_diverged:
                all_results[hp_summary]["__n_diverged__"] = n_diverged
                all_results[hp_summary]["__n_seeds__"] = len(seeds)
        results.clear()
        print(f"Results for {hp_summary}")
        pprint.pprint(all_results[hp_summary])

    print("\nSummary of Experiments:")
    for hp_summary, result in all_results.items():
        print(f"\nExperiment: {hp_summary}")
        if result.get("__all_diverged__"):
            print(
                f"  *** ALL {result['__n_seeds__']} SEEDS DIVERGED "
                f"(NaN loss / gradient explosion) — no metrics available ***"
            )
            continue
        if "__n_diverged__" in result:
            n_div = result["__n_diverged__"]
            n_tot = result["__n_seeds__"]
            print(
                f"  WARNING: {n_div}/{n_tot} seeds diverged — "
                f"statistics computed from {n_tot - n_div} seeds only"
            )
        print(
            f"  val_loss: {result['val_loss']['mean']:.4f}±{result['val_loss']['std']:.4f}  "
        )
        # Primary testset mAP with optional bootstrap CI
        map_line = (
            f"  mAP [{test_name}]: {result['map']['mean']:.4f}±{result['map']['std']:.4f}  "
            f"(slope: {result['map_slope']['mean']:+.4f})  "
            f"(best: {result['best_map']['mean']:.4f} @epoch {result['best_map_epoch']['mean']:.1f})"
        )

        ci_lo = result.get(f"bci_{test_name}_ci_low", {}).get("mean")
        ci_hi = result.get(f"bci_{test_name}_ci_high", {}).get("mean")
        ci_sd = result.get(f"bci_{test_name}_std", {}).get("mean")
        if ci_lo is not None:
            map_line += f"  95%CI=[{ci_lo:.4f},{ci_hi:.4f}] ±{ci_sd:.4f}"
        print(map_line)
        # Benchmark testsets (sorted for stable output order)
        for key in sorted(result.keys()):
            if not key.startswith("bmap_"):
                continue
            bname = key[5:]
            bmap_line = f"  mAP [{bname}]: {result[key]['mean']:.4f}±{result[key]['std']:.4f}  (benchmark)"
            ci_lo = result.get(f"bci_{bname}_ci_low", {}).get("mean")
            ci_hi = result.get(f"bci_{bname}_ci_high", {}).get("mean")
            ci_sd = result.get(f"bci_{bname}_std", {}).get("mean")
            if ci_lo is not None:
                bmap_line += f"  95%CI=[{ci_lo:.4f},{ci_hi:.4f}] ±{ci_sd:.4f}"
            print(bmap_line)
