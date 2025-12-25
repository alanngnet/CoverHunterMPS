#!/usr/bin/env python3
# author: liufeng
# datetime: 2023/3/16 3:41 PM


import argparse
import os
import torch

from src.eval_testset import eval_for_map_with_feat
from src.model import Model
from src.utils import (
    create_logger,
    get_hparams_as_string,
    load_hparams,
    read_lines,
)

torch.backends.cudnn.benchmark = True


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a pretrained model against a test dataset",
    )
    parser.add_argument("model_dir")
    parser.add_argument("query_path")
    parser.add_argument("ref_path")
    parser.add_argument("-query_in_ref_path", default="", type=str)
    parser.add_argument(
        "-plot_name",
        default="",
        type=str,
        help="Save a t-SNE plot of the distance matrix to this path. Default path is model_dir if plot_name is just a filename.",
    )
    parser.add_argument(
        "-marks",
        default="markers",
        type=str,
        help="plot marker mode 'markers' or 'ids'",
    )
    parser.add_argument(
        "-test_only_labels",
        default="",
        type=str,
        help="Path to list of work_ids reserved for test dataset for use in t-SNE plot.",
    )
    parser.add_argument(
        "-dist_name",
        default="",
        type=str,
        help="Save the distance matrix to this path",
    )
    parser.add_argument(
        "--reuse-embeddings",
        action="store_true",
        help="Reuse existing embeddings in embed_NN_tmp if present (faster but risky if data or parameters have changed)",
    )

    args = parser.parse_args()
    model_dir = args.model_dir
    query_path = args.query_path
    ref_path = args.ref_path
    query_in_ref_path = args.query_in_ref_path
    logger = create_logger()

    hp = load_hparams(os.path.join(model_dir, "config/hparams.yaml"))
    logger.info(f"{get_hparams_as_string(hp)}")

    # Auto-detect optimal device (evaluation device is independent of training device)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS device for evaluation")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using CUDA device for evaluation")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device for evaluation")
    hp["device"] = device  # src/model.py expects hp["device"]

    torch.manual_seed(hp["seed"])

    model = Model(hp).to(device)
    checkpoint_dir = os.path.join(model_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    epoch = model.load_model_parameters(checkpoint_dir, device=device)

    embed_dir = os.path.join(model_dir, "embed_{}_{}".format(epoch, "tmp"))

    if args.plot_name:
        plot_name = args.plot_name
        path = os.path.dirname(plot_name)
        if path != "":
            assert os.path.isdir(path), f"Invalid plot path: {plot_name}"
        else:
            # put the plot in model_dir as default location
            plot_name = os.path.join(model_dir, plot_name)
    else:
        plot_name = ""

    if args.test_only_labels:
        # convert list of work IDs from strings to integers as _cluster_plot() expects
        test_only_labels = [int(n) for n in read_lines(args.test_only_labels)]
    else:
        test_only_labels = None

    mean_ap, hit_rate, rank1 = eval_for_map_with_feat(
        hp,
        model,
        embed_dir,
        device=device,
        query_path=query_path,
        ref_path=ref_path,
        query_in_ref_path=query_in_ref_path,
        batch_size=64,
        logger=logger,
        test_only_labels=test_only_labels,
        plot_name=plot_name,
        marks=args.marks,
        dist_name=args.dist_name,
        reuse_embeddings=args.reuse_embeddings,
    )


if __name__ == "__main__":
    _main()
