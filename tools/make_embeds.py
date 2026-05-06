#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility to generate reference embeddings for all production-use audio data
available, using the production-ready model you trained using, for example,
tools.train_prod.py. Intended for use by tools.identify.py or other
applications you might create that use your fully trained model.

Example invocation:
    python -m tools.make_embeds data/covers80 training/covers80

Add --raw for research use (no normalization, no calibration; embeddings
stored exactly as the model emits them):
    python -m tools.make_embeds data/covers80 training/covers80 --raw

Tune calibration sample size (default 50000):
    python -m tools.make_embeds data/covers80 training/covers80 \\
        --calibration-pairs 100000

Parameters
----------
data_path : string
    Relative path to the project folder containing a full.txt file that
    you generated using tools.extract_csi_features.py, for example the one
    you used to train your model. These will be the recordings that your
    inference solution will "know."
    Example: "data/covers80"

model_path : string
    Relative path to the project folder containing your trained model.
    Example: "training/covers80"
    This script requires reuse of the following files that you used and
    generated during training of your model:
        [model_path]/config/hparams.yaml
        [model_path]/checkpoints/...

--raw : flag
    If set, store embeddings exactly as the model emits them (no L2
    normalization), and skip calibration. Use only for research into
    the radial coordinate; the resulting file is not consumable by
    tools.identify or downstream centroid utilities.

--calibration-pairs : int
    Number of cross-work pairs to sample for calibration. Default 50000.
    Ignored if --raw is set.

Output
------
Pickle file of reference embeddings saved to data_path:
    [data_path]/reference_embeddings.pkl

Schema (default, normalized):
    {
        "embeddings":         dict perf_id -> unit np.ndarray (float32)
        "norms":              dict perf_id -> float (pre-norm magnitude)
        "normalized":         True
        "embed_dim":          int
        "model_fingerprint":  str (16 hex chars)
        "calibration":        dict per src.calibration schema
    }

Schema (--raw):
    {
        "embeddings":         dict perf_id -> raw np.ndarray (float32)
        "norms":              dict perf_id -> float
        "normalized":         False
        "embed_dim":          int
        "model_fingerprint":  str
        "calibration":        None
    }


Complete rewrite 2026-05-03 by alanngnet with Claude Opus 4.7, switching to 
normalized embeddings as default, with calibration data supporting both 
end-user CSI/VI applications and research-grade data science work.

Originally created on Sat Jul 27 10:54:22 2024, generating raw embeddings only.
@author: alanngnet with Claude
"""

import os
import argparse
import pickle
import re
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.calibration import compute_calibration
from src.model import Model
from src.dataset import AudioFeatDataset
from src.utils import (
    compute_model_fingerprint,
    load_hparams,
    line_to_dict,
    read_lines,
)


def generate_embeddings(model, data_loader, device):
    """Generate raw embeddings for all samples in the data loader.

    Returns a dict perf_id -> np.ndarray (float32, un-normalized).
    Normalization, if any, is applied later in main().
    """
    embeddings = {}
    model.eval()
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Generating embeddings"):
            perf, feat, _ = batch
            feat = feat.float().to(device)
            embed, _ = model.inference(feat)
            for i, p in enumerate(perf):
                embeddings[p] = embed[i].cpu().numpy().astype(np.float32)
    return embeddings


def main(data_path, model_path, raw, calibration_pairs):
    # Load hyperparameters
    model_hp = load_hparams(os.path.join(model_path, "config/hparams.yaml"))

    # Set up device
    device = torch.device(model_hp["device"])

    # Initialize model
    model = Model(model_hp).to(device)
    checkpoint_dir = os.path.join(model_path, "checkpoints")
    model.load_model_parameters(checkpoint_dir, device=device)

    # Compute fingerprint from the loaded weights. Doing this after
    # load_model_parameters ensures we hash the actual weights now in
    # memory, not whatever happens to be on disk.
    fingerprint = compute_model_fingerprint(model.state_dict())
    print(f"Model fingerprint: {fingerprint}")

    # Prepare dataset
    # Need to use full.txt because only it has the necessary workid field
    dataset_file = os.path.join(data_path, "full.txt")
    lines = read_lines(dataset_file)
    # Filter out speed-augmented perfs
    lines = [
        line
        for line in lines
        if not re.match(r"sp_[0-9.]+-.+", line_to_dict(line)["perf"])
    ]
    print(f"Loaded {len(lines)} CQT arrays to compute their embeddings.")

    infer_frame = model_hp["chunk_frame"][0] * model_hp["mean_size"]
    dataset = AudioFeatDataset(
        model_hp,
        data_lines=lines,
        train=False,
        mode=model_hp["mode"],
        chunk_len=infer_frame,
    )

    data_loader = DataLoader(
        dataset,
        batch_size=model_hp["batch_size"],
        shuffle=False,
        num_workers=model_hp["num_workers"],
        pin_memory=True,
    )

    raw_embeddings = generate_embeddings(model, data_loader, device)

    # Always extract pre-normalization norms (cheap; preserved as OOD
    # signal even when --raw is not set).
    norms = {p: float(np.linalg.norm(v)) for p, v in raw_embeddings.items()}

    # Sanity check: any zero-norm embedding is a degenerate case that
    # would divide-by-zero on normalization. Fail loud.
    zero_norm = [p for p, n in norms.items() if n == 0.0]
    if zero_norm:
        raise RuntimeError(
            f"{len(zero_norm)} embedding(s) have zero norm: "
            f"{zero_norm[:5]}{'...' if len(zero_norm) > 5 else ''}. "
            f"This indicates a corrupt input or model bug."
        )

    if raw:
        embeddings_out = raw_embeddings
        normalized = False
        calibration = None
        print("Storing raw embeddings (no normalization, no calibration).")
    else:
        embeddings_out = {
            p: (v / norms[p]).astype(np.float32)
            for p, v in raw_embeddings.items()
        }
        normalized = True

        # Build work_to_perfs from perf_id naming convention
        # (perf_id format: "{work_id}.{performance_suffix}")
        work_to_perfs = {}
        for p in embeddings_out:
            work_id = p.split(".")[0]
            work_to_perfs.setdefault(work_id, []).append(p)

        print(
            f"Computing calibration from {calibration_pairs} cross-work "
            f"pairs..."
        )
        calibration = compute_calibration(
            embeddings_out,
            work_to_perfs,
            n_pairs=calibration_pairs,
        )
        print(
            f"Calibration: mu={calibration['mu']:.4f}, "
            f"sigma={calibration['sigma']:.4f}, "
            f"n_pairs={calibration['n_pairs']}"
        )

    # Determine embed_dim from any embedding (all have same shape)
    any_perf = next(iter(embeddings_out))
    embed_dim = int(embeddings_out[any_perf].shape[0])

    record = {
        "embeddings": embeddings_out,
        "norms": norms,
        "normalized": normalized,
        "embed_dim": embed_dim,
        "model_fingerprint": fingerprint,
        "calibration": calibration,
    }

    output_file = os.path.join(data_path, "reference_embeddings.pkl")
    with open(output_file, "wb") as f:
        pickle.dump(record, f)

    print(f"Reference embeddings saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate reference embeddings for production use."
    )
    parser.add_argument(
        "data_path", help="Path to the data folder containing dataset.txt"
    )
    parser.add_argument(
        "model_path", help="Path to the folder containing the trained model"
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Store raw (un-normalized) embeddings, skip calibration. "
        "Research-only; output is not consumable by tools.identify.",
    )
    parser.add_argument(
        "--calibration-pairs",
        type=int,
        default=50_000,
        help="Number of cross-work pairs to sample for calibration "
        "(default: 50000). Ignored if --raw.",
    )
    args = parser.parse_args()

    main(
        args.data_path,
        args.model_path,
        raw=args.raw,
        calibration_pairs=args.calibration_pairs,
    )
