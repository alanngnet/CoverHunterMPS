#!/usr/bin/env python3
"""
Production training with stratified K-fold cross-validation.

Trains a production-ready model on the complete dataset using stratified K-fold
cross-validation to handle realistic class imbalance in CSI datasets. Concludes
with a final training run on the full dataset.

Early Stopping Modes
--------------------
Controlled by hp["early_stop_metric"] (inherits from Trainer):

"val_loss" (default):
    Stop each fold when validation loss plateaus. Checkpoint selection uses
    retrospective mAP peak detection via TensorBoard log parsing. All testsets
    remain uncontaminated.

"mAP":
    Stop each fold when smoothed mAP plateaus on hp["map_stopping_testsets"].
    Checkpoint selection uses Trainer's tracked best_raw_map_epoch directly.
    Testsets not in map_stopping_testsets are benchmark-only (logged but
    never influence training), preserving research-grade separation.

For production deployment where benchmark purity is not required, set
map_stopping_testsets to include all available testsets for maximum mAP
optimization.

Hyperparameters (hparams_prod.yaml)
-----------------------------------
Expects all standard hparams.yaml parameters plus:

    train_path: path to full.txt from extract_csi_features.py
    val_path: Ignored (validation sets generated per fold)
    test_path: Used only for final full-dataset training run as the validation set

    early_stop_metric: "val_loss" or "mAP"
    map_stopping_testsets: List of testsets for mAP-based stopping
    map_stopping_patience: Epochs without improvement (mAP mode)
    map_smoothing_alpha: EMA weight for mAP smoothing

    k_folds: Number of cross-validation folds
    early_stopping_patience: Patience for fold training (shorter than research)
        Set a lower early_stopping_patience than you likely used in experiments,
        to avoidoverfitting in each fold.

    final_early_stopping_patience: Optional longer patience for final run
    fold_lr_decay: Learning rate decay for folds after the first
    final_lr_decay: Learning rate decay for final full-dataset run
    full_dataset_lr_boost: LR boost factor for final run


Example launch command:
    python -m tools.train_prod training/yourdata --runid='prod_v1'

Output
------
    training/yourdata/prod_checkpoints/: Model checkpoint files
    training/yourdata/logs/{runid}_fold_N/: TensorBoard logs per fold
    training/yourdata/logs/{runid}_full/: TensorBoard logs for final run

Legacy Support
--------------
The module constant MAP_TESTSETS provides backward compatibility when
hp["map_stopping_testsets"] is not specified. New configurations should
use the hyperparameter instead.

Optionally specify in MAP_TESTSETS which testsets to use to define peak mAP.
Add or remove from this list based on your relevant testsets.
This allows train_prod to optimize quality of training by
selecting the best epoch (defined by recent peak mAP in your selected testsets)
to return to before starting the next fold. Note: Setting testsets here does
mean that checkpoint data for epochs after the best epoch in each fold
will be deleted.

Created on Wed May 22 19:27:14 2024
@author: Alan Ng

"""

# Kept for backward compatibility when map_stopping_testsets not specified
# When early_stop_metric="val_loss": These testsets are used for retrospective
# peak mAP detection via TensorBoard log parsing after each fold.
#
# When early_stop_metric="mAP": This constant is ignored; Trainer uses
# hp["map_stopping_testsets"] directly for live stopping decisions.
MAP_TESTSETS = ["reels50easy", "reels50hard", "reels50transpose"]

import argparse
import os, glob, shutil
import time
import sys
import json
import numpy as np
import torch
import torch.multiprocessing as mp
from sklearn.model_selection import StratifiedKFold
from src.trainer import Trainer
from src.model import Model
from src.utils import create_logger, get_hparams_as_string, load_hparams
from src.dataset import AudioFeatDataset, read_lines
from tensorboard.backend.event_processing.event_accumulator import (
    EventAccumulator,
)


def extract_labels(dataset):
    labels = []
    for i in range(len(dataset)):
        _, _, label = dataset[i]
        labels.append(label.item())
    return np.array(labels)


def write_temp_file(data_lines, indices, filepath):
    with open(filepath, "w") as f:
        for idx in indices:
            f.write(data_lines[idx] + "\n")


def save_fold_indices(fold_indices, filepath):
    with open(filepath, "w") as f:
        json.dump(fold_indices, f)


def load_fold_indices(filepath):
    with open(filepath, "r") as f:
        return json.load(f)


def get_map_from_logs(log_dir, testset_name, epoch):
    """
    Get mAP value for specific testset and epoch
    """
    event_files = sorted(
        [f for f in os.listdir(log_dir) if "events.out.tfevents" in f],
        key=lambda x: os.path.getctime(os.path.join(log_dir, x)),
    )
    if not event_files:
        return None

    event_file = os.path.join(log_dir, event_files[-1])
    ea = EventAccumulator(event_file)
    ea.Reload()

    try:
        map_values = ea.Scalars(f"mAP/{testset_name}")
        # Find the closest event to our target epoch
        closest_event = min(
            map_values, key=lambda x: abs(x.step - epoch), default=None
        )
        # Allow events within 1 step of target epoch
        if closest_event and abs(closest_event.step - epoch) <= 1:
            return closest_event.value
    except KeyError:
        return None
    return None


def get_average_map_for_epoch(log_dir, testsets, epoch):
    """
    Calculate average mAP across all specified testsets for a given epoch.

    Helper for find_peak_map_epoch(). Reads mAP values from TensorBoard
    event files and returns their arithmetic mean.

    Note:
    Uses arithmetic mean for aggregation. For alternative strategies
    (weighted, min), modify Trainer.aggregate_stopping_map_scores()
    when using mAP-based early stopping instead of this log-parsing path.

    """
    maps = []
    for testset in testsets:
        map_value = get_map_from_logs(log_dir, testset, epoch)
        if map_value is not None:
            maps.append(map_value)

    return np.mean(maps) if maps else None


def find_peak_map_epoch(
    log_dir, testsets, current_epoch, early_stopping_window
):
    """
    Find the epoch with highest average mAP within the early stopping window

    Used for retrospective checkpoint selection when early_stop_metric="val_loss".
    When early_stop_metric="mAP", Trainer tracks best_raw_map_epoch directly
    and this function serves only as a fallback.

    Args:
        log_dir: Directory containing tensorboard logs
        testsets: List of testset names to check
        current_epoch: The epoch at which training stopped
        early_stopping_window: Number of epochs to look back

    Returns:
        int or None: Epoch number with highest average mAP within the window,
        or None if no mAP data found.

    Note:
        This function parses TensorBoard logs after training completes,
        which requires a brief sleep for filesystem sync. The mAP-based
        early stopping mode in Trainer avoids this latency by tracking
        the best epoch during training.

    """
    event_files = sorted(
        [f for f in os.listdir(log_dir) if "events.out.tfevents" in f],
        key=lambda x: os.path.getctime(os.path.join(log_dir, x)),
    )
    if not event_files:
        return None

    event_file = os.path.join(log_dir, event_files[-1])
    ea = EventAccumulator(event_file)
    ea.Reload()

    # Calculate the earliest epoch to consider
    earliest_epoch = max(0, current_epoch - early_stopping_window)

    # Get all epochs where we have mAP values, filtered by window
    epochs_to_check = set()
    for testset in testsets:
        try:
            events = ea.Scalars(f"mAP/{testset}")
            epochs_to_check.update(
                event.step
                for event in events
                if earliest_epoch <= event.step <= current_epoch
            )
        except KeyError:
            continue

    if not epochs_to_check:
        return None

    # Find epoch with highest average mAP within window
    best_epoch = None
    best_map = -float("inf")
    for epoch in epochs_to_check:
        avg_map = get_average_map_for_epoch(log_dir, testsets, epoch)
        if avg_map is not None and avg_map > best_map:
            best_map = avg_map
            best_epoch = epoch

    return best_epoch


def cleanup_checkpoints(checkpoint_dir, best_epoch):
    """
    Remove all checkpoints and testset embeddings after the best epoch
    """
    for filename in os.listdir(checkpoint_dir):
        if filename.startswith(("g_", "do_")):
            epoch = int(filename.split("_")[1])
            if epoch > best_epoch:
                os.remove(os.path.join(checkpoint_dir, filename))
    embed_dirs = glob.glob(
        os.path.join(os.path.dirname(checkpoint_dir), "embed_*_*")
    )
    for directory in embed_dirs:
        shutil.rmtree(directory)


def cross_validate(
    hp, model_class, device, checkpoint_dir, model_dir, run_id, n_splits=5
):
    """
    Perform stratified K-fold cross-validation for production model training.

    Trains the model across K folds to ensure exposure to all works and
    performances, then concludes with a final training run on the complete
    dataset. Each fold uses early stopping and checkpoint selection based
    on hp["early_stop_metric"].

    Args:
        hp: dict
            Hyperparameters including early stopping configuration.
        model_class: class
            Model class to instantiate (typically src.model.Model).
        device: torch.device
            Computation device (mps, cuda).
        checkpoint_dir: str
            Directory for saving model checkpoints (shared across folds).
        model_dir: str
            Base directory for the training run.
        run_id: str
            Identifier for TensorBoard log subdirectories.
        n_splits: int
            Number of cross-validation folds (default: 5).

    Returns:
        list[dict]: Results for each fold and final run, containing:
            - "fold": Fold number or "full" for final run
            - "best_validation_loss": Lowest validation loss achieved

    Checkpoint Selection
    --------------------
    After each fold completes:

    If early_stop_metric="mAP" and Trainer tracked best_raw_map_epoch:
        Uses Trainer's live-tracked peak mAP epoch directly.

    If early_stop_metric="val_loss" with testsets available:
        Falls back to retrospective log parsing via find_peak_map_epoch().

    Otherwise:
        Uses the final checkpoint from early stopping.

    Checkpoints after the selected best epoch are cleaned up to save disk
    space and ensure the best model is used for subsequent folds.

    Learning Rate Strategy
    ----------------------
    Folds after the first use progressively lower learning rates, linearly
    interpolated from hp["lr_initial"] to hp["min_lr"]. This prevents
    catastrophic forgetting of patterns learned in earlier folds.

    The final full-dataset run uses hp["full_dataset_lr_boost"] to set
    a learning rate slightly above the minimum.

    Resumption
    ----------
    Training can be interrupted and resumed safely. The function tracks
    completed folds via marker files (fold_N_started.txt, active_fold.txt)
    and skips already-completed folds on restart.
    """

    logger = create_logger()

    # Read data lines from the original file
    data_lines = read_lines(hp["train_path"])

    # Load the dataset once for splitting purposes
    initial_dataset = AudioFeatDataset(
        hp,
        data_path=None,
        data_lines=data_lines,
        train=False,  # No augmentation at this point
        mode=hp["mode"],
        chunk_len=hp["chunk_frame"][0] * hp["mean_size"],
    )

    # Extract labels
    labels = extract_labels(initial_dataset)

    # Generate indices for stratified k-fold
    indices = np.arange(len(initial_dataset))

    # Save folds in case training needs to be restarted after an interruption
    fold_indices_file = os.path.join(model_dir, "fold_indices.json")
    active_fold_file = os.path.join(model_dir, "active_fold.txt")
    last_completed_fold = -1

    if os.path.exists(active_fold_file):
        with open(active_fold_file, "r") as f:
            last_completed_fold = int(f.readline().strip())
        logger.info(f"Resuming from fold {last_completed_fold + 1}")
        fold_indices = load_fold_indices(fold_indices_file)
    else:
        kf = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=hp["seed"]
        )
        fold_indices = [
            (train_idx.tolist(), val_idx.tolist())
            for train_idx, val_idx in kf.split(indices, labels)
        ]
        save_fold_indices(fold_indices, fold_indices_file)

    fold_results = []

    original_train_path = hp["train_path"]
    original_test_path = hp.pop("test_path")  # save for final full dataset

    # Identify available testsets from those specified in MAP_TESTSETS
    available_testsets = [t for t in MAP_TESTSETS if t in hp]
    if not available_testsets:
        logger.warning(
            "No specified testsets found in hyperparameters. Using validation loss for peak detection."
        )

    for fold, (train_idx, val_idx) in enumerate(fold_indices):
        if fold <= last_completed_fold:
            continue

        logger.info(f"Training on fold {fold+1}/{n_splits}")

        # Write temporary train and val files
        train_file = os.path.join(model_dir, f"train_fold_{fold+1}.txt")
        val_file = os.path.join(model_dir, f"val_fold_{fold+1}.txt")
        write_temp_file(data_lines, train_idx, train_file)
        write_temp_file(data_lines, val_idx, val_file)
        logger.info(
            f"Temporary fold {fold+1} data files created at {train_file} and {val_file}"
        )

        # Temporarily change hp paths
        hp["train_path"] = train_file
        hp["val_path"] = val_file

        # Create a unique log path for this fold
        log_path = os.path.join(model_dir, "logs", f"{run_id}_fold_{fold+1}")
        os.makedirs(log_path, exist_ok=True)

        # Check if this is a new fold start
        fold_start_file = os.path.join(model_dir, f"fold_{fold+1}_started.txt")
        is_new_fold_start = not os.path.exists(fold_start_file)

        # Instantiate and train a new Trainer instance for this fold
        trainer = Trainer(
            hp=hp,
            model=model_class,
            device=device,
            log_path=log_path,
            checkpoint_dir=checkpoint_dir,
            model_dir=model_dir,
            only_eval=False,
            first_eval=False,
        )
        trainer.configure_optimizer()
        trainer.load_model()
        trainer.configure_scheduler()

        # different learning-rate strategy for all folds after the first
        if fold > 0 and is_new_fold_start:
            # Calculate evenly spaced learning rates from 0.0001 to min_lr
            lr_range = hp["lr_initial"] - hp["min_lr"]
            lr_step = lr_range / (
                n_splits - 1
            )  # -1 since first fold uses original lr
            new_lr = (
                hp["lr_initial"] - (fold - 1) * lr_step
            )  # fold-1 since we start at fold 1
            trainer.reset_learning_rate(new_lr=new_lr)
            hp["lr_decay"] = hp["fold_lr_decay"]
            logger.info(
                f"Adjusted learning rate for fold {fold+1}: lr={new_lr}, min_lr={hp['min_lr']}, decay={hp['lr_decay']}"
            )
        else:
            logger.info(f"Resuming learning rate for fold {fold+1}")

        # Mark this fold as started
        with open(fold_start_file, "w") as f:
            f.write(f"Fold {fold+1} started")

        trainer.train(max_epochs=500)

        # Find peak mAP epoch achieved so far
        # After trainer.train() completes for each fold:
        trainer.summary_writer.flush()
        trainer.summary_writer.close()

        # Use Trainer's tracked best epoch when in mAP mode
        if (
            hp.get("early_stop_metric") == "mAP"
            and trainer.best_raw_map_epoch is not None
        ):
            logger.info(
                f"Peak mAP {trainer.best_raw_map:.4f} at epoch {trainer.best_raw_map_epoch}"
            )
            cleanup_checkpoints(checkpoint_dir, trainer.best_raw_map_epoch)
        elif available_testsets:
            # Fallback to log parsing for val_loss mode with testsets
            time.sleep(1)
            best_epoch = find_peak_map_epoch(
                log_path,
                available_testsets,
                trainer.epoch,
                hp["early_stopping_patience"] + 1,
            )
            if best_epoch is not None:
                logger.info(f"Peak mAP achieved at epoch {best_epoch}")
                cleanup_checkpoints(checkpoint_dir, best_epoch)
            else:
                logger.warning("Could not determine peak mAP epoch")

        fold_results.append(
            {
                "fold": fold,
                "best_validation_loss": trainer.best_validation_loss,
            }
        )

        # Save the last completed fold
        with open(active_fold_file, "w") as f:
            f.write(str(fold))

    logger.info("Cross-validation completed")

    # Train on the full dataset, and use a testset as the validation set
    logger.info("Training on the full dataset")
    hp["train_path"] = original_train_path
    hp["val_path"] = original_test_path

    # Create a unique log path for the full dataset training
    log_path = os.path.join(model_dir, "logs", f"{run_id}_full")
    os.makedirs(log_path, exist_ok=True)

    # Check if this is a new full dataset training start
    full_start_file = os.path.join(model_dir, "full_dataset_started.txt")
    is_new_full_start = not os.path.exists(full_start_file)

    # Instantiate and train a new Trainer instance for the full dataset
    full_trainer = Trainer(
        hp=hp,
        model=model_class,
        device=device,
        log_path=log_path,
        checkpoint_dir=checkpoint_dir,
        model_dir=model_dir,
        only_eval=False,
        first_eval=False,
    )
    full_trainer.configure_optimizer()
    full_trainer.load_model()
    full_trainer.configure_scheduler()

    # Adjust learning rate for full dataset training if it's a new start
    # let full dataset have a little more learning rate than the final folds did
    if is_new_full_start:
        # recalculate in case resuming after an interruption
        lr_range = hp["lr_initial"] - hp["min_lr"]
        lr_step = lr_range / (
            n_splits - 1
        )  # -1 since first fold uses original lr
        new_lr = hp["min_lr"] + lr_step * hp["full_dataset_lr_boost"]
        hp["lr_decay"] = hp["final_lr_decay"]
        logger.info(
            f"Adjusted learning rate for fold {fold+1}: lr={new_lr}, min_lr={hp['min_lr']}, decay={hp['lr_decay']}"
        )
        full_trainer.reset_learning_rate(new_lr=new_lr)
    else:
        logger.info("Resuming learning rate for full dataset training")

    # allow user to define longer patience for this final run
    if "final_early_stopping_patience" in hp:
        hp["early_stopping_patience"] = hp["final_early_stopping_patience"]

    # Mark full dataset training as started
    with open(full_start_file, "w") as f:
        f.write("Full dataset training started")

    full_trainer.train(max_epochs=500)
    fold_results.append(
        {
            "fold": "full",
            "best_validation_loss": full_trainer.best_validation_loss,
        }
    )
    logger.info("Full dataset training completed")

    return fold_results


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Train: python3 -m tools.train_prod model_dir",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("model_dir")
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="give more debug log",
    )
    parser.add_argument(
        "--runid",
        default="",
        action="store",
        help="put TensorBoard logs in these subfolders of ../logs/ one per fold",
    )

    args = parser.parse_args()
    model_dir = args.model_dir
    run_id = args.runid

    logger = create_logger()
    hp = load_hparams(os.path.join(model_dir, "config/hparams_prod.yaml"))
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
            # set multiprocessing method because 'fork'
            # has significant performance boost on MPS vs. default 'spawn'
            mp.set_start_method("fork")
        case "cuda":
            if not torch.cuda.is_available():
                logger.error(
                    "You requested 'cuda' device in your hyperparameters"
                    "but you do not have a CUDA-compatible GPU available."
                )
                sys.exit()
            device = torch.device("cuda")
            mp.set_start_method("spawn")

        case _:
            logger.error(
                "You set device: %s"
                " in your hyperparameters but that is not a valid option or is an untested option.",
                hp["device"],
            )
            sys.exit()

    # Validate testset specifications
    # Determine stopping testsets: prefer hp config, fall back to module constant
    if "map_stopping_testsets" in hp:
        available_testsets = [
            t for t in hp["map_stopping_testsets"] if t in hp
        ]
        if not available_testsets:
            logger.warning(
                "map_stopping_testsets specified but none found in hyperparameters. "
                "Check that each testset name has a matching configuration block. "
                "Falling back to val_loss early stopping."
            )
    else:
        available_testsets = [t for t in MAP_TESTSETS if t in hp]
        if not available_testsets:
            logger.info(
                "No map_stopping_testsets configured and no MAP_TESTSETS found. "
                "Using val_loss for early stopping."
            )

    if available_testsets:
        logger.info(f"Using testsets for mAP tracking: {available_testsets}")

    logger.info("%s", get_hparams_as_string(hp))

    torch.manual_seed(hp["seed"])

    checkpoint_dir = os.path.join(model_dir, "prod_checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    fold_results = cross_validate(
        hp=hp,
        model_class=Model,
        device=device,
        checkpoint_dir=checkpoint_dir,
        model_dir=model_dir,
        run_id=run_id,
        n_splits=hp["k_folds"],
    )

    for result in fold_results:
        logger.info(
            f"Fold {result['fold']} - Best Validation Loss: {result['best_validation_loss']}"
        )


if __name__ == "__main__":
    _main()
