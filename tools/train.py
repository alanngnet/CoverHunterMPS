#!/usr/bin/env python3

import argparse
import os
import sys
import time
import torch
import torch.multiprocessing as mp

from src.trainer import Trainer

from src.model import Model
from src.utils import create_logger, get_hparams_as_string, load_hparams


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Train: python3 -m tools.train model_dir",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("model_dir")
    parser.add_argument(
        "--first_eval",
        default=False,
        action="store_true",
        help="Set for run eval first before train",
    )
    parser.add_argument(
        "--only_eval",
        default=False,
        action="store_true",
        help="Set for run eval first before train",
    )
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
        help="put TensorBoard logs in this subfolder of ../logs/",
    )
    args = parser.parse_args()
    model_dir = args.model_dir
    first_eval = args.first_eval
    only_eval = args.only_eval
    run_id = args.runid
    first_eval = True if only_eval else first_eval

    logger = create_logger()
    start_time = time.time()
    hp = load_hparams(os.path.join(model_dir, "config/hparams.yaml"))

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
        case _:
            logger.error(
                "You set device: %s"
                " in your hyperparameters but that is not a valid option or is an untested option.",
                hp["device"],
            )
            sys.exit()

    logger.info("%s", get_hparams_as_string(hp))

    torch.manual_seed(hp["seed"])

    # Note only for distributed-computing use from original CoverHunter:
    # We use map-reduce mode to update model when its parameters changed
    # (model.join), that means we do not need to wait one step of all gpu to
    # complete. Pytorch distribution support variable trained samples of different
    # gpus.
    # And also, we compute val/test/testset on different gpu within epoch.
    # For example: we compute test at rank0 when epoch 1, when test is computing,
    # rank1 is going on training and update parameters. When epoch 2, we change
    # to compute test at rank1, to make sure all ranks run the same train steps
    # almost.

    checkpoint_dir = os.path.join(model_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    log_path = os.path.join(model_dir, "logs", run_id)
    os.makedirs(log_path, exist_ok=True)

    t = Trainer(
        hp,
        Model,
        device,
        log_path,
        checkpoint_dir,
        model_dir,
        only_eval,
        first_eval,
    )

    t.configure_optimizer()
    t.load_model()
    t.configure_scheduler()
    t.reset_learning_rate()
    t.train(max_epochs=hp.get("max_epochs", 100))

    # Final summary
    early_stop_metric = hp.get("early_stop_metric", "val_loss")
    if t.best_checkpoint_epoch is not None:
        logger.info(
            f"Training complete. Best checkpoint: epoch {t.best_checkpoint_epoch} "
            f"({early_stop_metric}={t.best_checkpoint_value:.4f})"
        )
    else:
        logger.info("Training complete.")
    elapsed = time.time() - start_time
    hours, remainder = divmod(elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)
    logger.info(
        f"Total training time: {int(hours)}h {int(minutes)}m {seconds:.1f}s"
    )


if __name__ == "__main__":
    _main()
