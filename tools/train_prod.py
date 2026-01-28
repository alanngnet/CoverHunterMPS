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
    the epoch with lowest validation loss (tracked live by Trainer). Use this
    mode when no benchmark testsets are available for your musical culture.

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
        to avoid overfitting in each fold.

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
use the hyperparameter instead. In val_loss mode, these
testsets are still evaluated and logged for monitoring, but checkpoint
selection uses validation loss (tracked by Trainer).

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
import os, glob, shutil, time
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

# from tensorboard.backend.event_processing.event_accumulator import (
#    EventAccumulator,
# )


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


def load_global_best(filepath):
    """Load global best checkpoint tracking across folds."""
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            return json.load(f)
    return {"epoch": None, "value": None}


def save_global_best(filepath, epoch, value):
    """Persist global best checkpoint across abort/resume events."""
    with open(filepath, "w") as f:
        json.dump({"epoch": epoch, "value": value}, f)


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
    After each fold completes, uses Trainer's tracked best_checkpoint_epoch,
    which reflects the best epoch for the configured early_stop_metric
    (lowest val_loss or highest raw mAP). Checkpoints after the selected
    best epoch are cleaned up to save disk space.

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
        logger.info(
            f"Last completed fold: {last_completed_fold}; resuming at fold {last_completed_fold + 1}"
        )
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

    # Track global best checkpoint across all folds
    global_best_file = os.path.join(model_dir, "global_best.json")
    global_best = load_global_best(global_best_file)
    global_best_epoch = global_best["epoch"]
    global_best_value = global_best["value"]
    early_stop_metric = hp.get("early_stop_metric", "val_loss")

    if global_best_epoch is not None:
        logger.info(
            f"Loaded global best: epoch {global_best_epoch}, "
            f"{early_stop_metric}={global_best_value:.4f}"
        )

    for fold, (train_idx, val_idx) in enumerate(fold_indices, start=1):
        if fold <= last_completed_fold:
            continue

        logger.info(f"Training on fold {fold}/{n_splits}")

        # Write temporary train and val files
        train_file = os.path.join(model_dir, f"train_fold_{fold}.txt")
        val_file = os.path.join(model_dir, f"val_fold_{fold}.txt")
        write_temp_file(data_lines, train_idx, train_file)
        write_temp_file(data_lines, val_idx, val_file)
        logger.info(
            f"Temporary fold {fold} data files created at {train_file} and {val_file}"
        )

        # Temporarily change hp paths
        hp["train_path"] = train_file
        hp["val_path"] = val_file

        # Create a unique log path for this fold
        log_path = os.path.join(model_dir, "logs", f"{run_id}_fold_{fold}")
        os.makedirs(log_path, exist_ok=True)

        # Check if this is a new fold start
        fold_start_file = os.path.join(model_dir, f"fold_{fold}_started.txt")
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

        # Reset early stopping state for new folds
        if is_new_fold_start:
            trainer.reset_early_stopping_state()

        # =================================================================
        # LEARNING RATE SCHEDULING FOR LATER FOLDS
        # =================================================================
        # Folds 2-K use progressively lower starting learning rates,
        # linearly interpolated from lr_initial (fold 2) toward min_lr.
        # This prevents catastrophic forgetting of patterns learned in
        # earlier folds while still allowing refinement.

        #
        # lr_step is calculated so that each fold including fold 5 and
        # the final fold have room to descend to min_lr, allowing some
        # annealing between folds.
        #

        # Fold 1: learning_rate (training from scratch)
        # Fold 2: lr_initial (first refinement from checkpoint)
        # Fold 3: lr_initial - 1*lr_step
        # ...
        # Fold K: min_lr + lr_step (one step above floor)
        # =================================================================
        if fold > 1 and is_new_fold_start:
            lr_range = hp["lr_initial"] - hp["min_lr"]
            lr_step = lr_range / (
                n_splits - 1
            )  # Example: 3 intervals for folds 2→3→4→5
            new_lr = (
                hp["lr_initial"] - (fold - 2) * lr_step
            )  # fold 2 uses lr_initial
            trainer.reset_learning_rate(new_lr=new_lr)
            hp["lr_decay"] = hp["fold_lr_decay"]
            # Enable hold period for later folds to stabilize after checkpoint load
            hp["hold_steps"] = hp.get("fold_hold_steps", 0)
            logger.info(
                f"Adjusted learning rate for fold {fold}: lr={new_lr}, hold_steps={hp['hold_steps']}, decay={hp['lr_decay']}"
            )
            trainer.configure_scheduler()  # rebuild scheduler with new hold_steps
        elif is_new_fold_start:
            # Fold 1: no hold period needed when training from scratch
            hp["hold_steps"] = 0
            logger.info(
                f"Fold {fold}/{n_splits}: using initial lr={hp['lr_initial']}"
            )
        else:
            logger.info(f"Fold {fold}/{n_splits}: resuming from checkpoint")

        # Mark this fold as started (for recovery)
        with open(fold_start_file, "w") as f:
            f.write(f"Fold {fold} started")

        trainer.train(max_epochs=500)

        # =================================================================
        # CHECKPOINT SELECTION - GLOBAL BEST TRACKING
        # =================================================================
        # Compare this fold's best to global best across all folds.
        # Only the true global best is retained; later checkpoints are cleaned up.
        # =================================================================
        trainer.summary_writer.flush()
        trainer.summary_writer.close()

        if getattr(trainer, "best_checkpoint_epoch", None) is not None:
            fold_best_epoch = trainer.best_checkpoint_epoch
            fold_best_value = trainer.best_checkpoint_value

            # Determine if this fold's best beats global best
            update_global = False
            if global_best_value is None:
                update_global = True
            elif early_stop_metric == "mAP":
                update_global = (
                    fold_best_value > global_best_value
                )  # higher is better
            else:  # val_loss - lower is better
                update_global = fold_best_value < global_best_value

            if update_global:
                global_best_epoch = fold_best_epoch
                global_best_value = fold_best_value
                save_global_best(
                    global_best_file, global_best_epoch, global_best_value
                )
                logger.info(
                    f"New global best: epoch {global_best_epoch} "
                    f"({early_stop_metric}={global_best_value:.4f})"
                )
            else:
                logger.info(
                    f"Fold {fold} best (epoch {fold_best_epoch}, "
                    f"{early_stop_metric}={fold_best_value:.4f}) "
                    f"did not beat global best (epoch {global_best_epoch}, "
                    f"{early_stop_metric}={global_best_value:.4f})"
                )

            cleanup_checkpoints(checkpoint_dir, global_best_epoch)
        #            best_epoch = global_best_epoch  # for fold_results
        else:
            # No evaluations occurred (unusual—only if max_epochs=0 or similar)
            logger.warning(
                "No checkpoint tracking during training—using final epoch"
            )
        #            best_epoch = trainer.epoch

        fold_results.append(
            {
                "fold": fold,
                # use getattr() to handle older versions of Trainer and checkpoints
                "fold_best_epoch": getattr(
                    trainer, "best_checkpoint_epoch", None
                ),
                "fold_best_value": getattr(
                    trainer, "best_checkpoint_value", None
                ),
                "global_best_epoch": global_best_epoch,
                "global_best_value": global_best_value,
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
            f"Adjusted learning rate for final full run: lr={new_lr}, min_lr={hp['min_lr']}, decay={hp['lr_decay']}"
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
            "best_epoch": full_trainer.best_checkpoint_epoch,
            "best_checkpoint_value": full_trainer.best_checkpoint_value,
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
    start_time = time.time()

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
    elapsed = time.time() - start_time
    hours, remainder = divmod(elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)
    logger.info(
        f"Total training time: {int(hours)}h {int(minutes)}m {seconds:.1f}s"
    )


if __name__ == "__main__":
    _main()
