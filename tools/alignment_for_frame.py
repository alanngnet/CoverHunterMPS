#!/usr/bin/env python3
# author: liufeng
# datetime: 2023/7/5 6:08 PM


import argparse
import os
import sys

import torch

from src.Aligner import Aligner
from src.eval_testset import eval_for_map_with_feat
from src.model import Model
from src.utils import create_logger, get_hparams_as_string, load_hparams

torch.backends.cudnn.benchmark = True


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="alignment with coarse trained model"
    )
    parser.add_argument("model_dir", help="coarse trained dir")
    parser.add_argument("data_path", help="input file contains init data")
    parser.add_argument(
        "alignment_path",
        help="output file contains alignment information",
    )

    args = parser.parse_args()
    model_dir = args.model_dir
    data_path = args.data_path
    alignment_path = args.alignment_path
    logger = create_logger()
    hp = load_hparams(os.path.join(model_dir, "config/hparams.yaml"))
    logger.info(f"{get_hparams_as_string(hp)}")

    match hp["device"]:
        case "mps":
            assert (
                torch.backends.mps.is_available()
            ), "You requested 'mps' device in your hyperparameters but you are not running on an Apple M-series chip or have not compiled PyTorch for MPS support."
            device = torch.device("mps")
        case "cuda":
            assert (
                torch.cuda.is_available()
            ), "You requested 'cuda' device in your hyperparameters but you do not have a CUDA-compatible GPU available."
            device = torch.device("cuda")
        case _:
            print(
                "You set device: ",
                hp["device"],
                " in your hyperparameters but that is not a valid option.",
            )
            sys.exit()

    # Note: we need to change chunks to 15s
    hp["chunk_frame"] = [125]
    hp["chunk_s"] = 15  # = 125 / 25 * 3

    model = Model(hp).to(device)
    checkpoint_dir = os.path.join(model_dir, "checkpoint")
    os.makedirs(checkpoint_dir, exist_ok=True)
    epoch = model.load_model_parameters(checkpoint_dir, device)

    # Calculate all chunks embedding and dump.
    embed_dir = os.path.join(model_dir, "embed_{}_{}".format(epoch, "tmp"))
    mean_ap, hit_rate, rank1 = eval_for_map_with_feat(
        hp,
        model,
        embed_dir,
        query_path=data_path,
        ref_path=data_path,
        query_in_ref_path=None,
        batch_size=64,
        logger=logger,
        reuse_embeddings=False,
    )
    logger.info(f"Test, map:{mean_ap}")

    # Calculate shift frames for every-two items with same label
    aligner = Aligner(os.path.join(embed_dir, "query_embed"))
    aligner.align(data_path, alignment_path)
    logger.info(f"Output alignment into {alignment_path}")


if __name__ == "__main__":
    _main()
