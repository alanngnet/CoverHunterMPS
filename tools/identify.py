#!/usr/bin/env python3
"""
Command-line utility to identify the closest matches to the query audio
from within the trained model's training data.

Assumptions:
    Query audio has a duration < chunk_s (default = 135sec).
    You already trained your model.
    You already ran tools.make_embeds

Example invocation:
python -m tools.identify data/covers80 training/covers80 query.wav -top=10

Or after using the optional tools.make_centroids utility:
    
python -m tools.identify data/covers80 training/covers80 query.wav -centroids

Parameters
----------
data_path : string
    The relative path that must contain:
        1) hparams.yaml file as documented for use with
        tools.extract_csi_features. These hyperparameters are necessary to
        generate a CQT spectrogram for your query that uses the same CQT
        parameters as were used for the audio during training of the model. 
        Most important is that n_bins must match.
        2) The reference_embeddings.pkl file that you generated using
        tools.make_embeds.

model_path : string
    The relative path that must contain: 
        1) a subfolder "checkpoints" containing checkpoint files
        2) The model's hyperparameters as hparams.yaml

query_path : string
    Relative path to the query audio. 

top : integer
    Optional. Return N closest matches to this query where N = top.

save : string
    Relative path to use for saving the embedding of the query audio. 

Created on Sat Mar  2 17:31:23 2024
@author: alanngnet

"""
import os, torch, numpy as np
import argparse
import pickle
import librosa
from heapq import nsmallest
from scipy.spatial.distance import cosine
from nnAudio.features.cqt import CQT, CQT2010v2
from tabulate import tabulate
from src.model import Model
from src.cqt import shorter
from src.utils import (
    load_hparams,
    #    RARE_DELIMITER,
    #    line_to_dict,
    #    read_lines,
)


def _make_feat(wav_path, fmin, max_freq, n_bins, bins_per_octave, device):
    """
    Borrowed CQT logic from tools.extract_csi_features
        _extract_cqt_worker_torchaudio()
    This function should stay synchronized with that one.

    Parameters
    ----------

    wav_path : string
       path to the .wav audio file that you want to identify

    For other parameters see:
        tools.extract_csi_features::_extract_cqt_worker_torchaudio()

    Returns
    -------
    CQT array

    """
    # CQT seems faster on mps, and CQT2010v2 faster on cuda
    if device == "mps":
        transform = CQT
    elif device == "cuda":
        transform = CQT2010v2

    # Load audio using librosa and force sample rate of 16kHz
    signal, sr = librosa.load(wav_path, sr=16000, mono=True)
    
    # Convert to torch tensor and move to device
    signal = torch.from_numpy(signal).float().unsqueeze(0).to(device)

    signal = (
        signal
        / torch.max(
            torch.tensor(0.001).to(device), torch.max(torch.abs(signal))
        )
        * 0.999
    )
    signal = transform(
        16000,
        hop_length=640,
        n_bins=n_bins,
        fmin=fmin,
        bins_per_octave=bins_per_octave,
        verbose=False,
    ).to(device)(signal)
    signal = signal + 1e-9
    signal = signal.squeeze(0)

    # Add contrast
    ref_value_log10 = torch.log10(torch.max(signal))
    signal = 20 * torch.log10(signal) - 20 * ref_value_log10

    signal = torch.swapaxes(signal, 0, 1)
    cqt = signal.numpy(force=True)
    return cqt


def _get_feat(query_path, data_hp, chunk_len, mean_size):
    """
    adapted from dataset.py AudioFeatDataset::__getitem__()
    assumes "mode" = "defined" and start = 0
    """
    if "bins_per_octave" not in data_hp:
        bins_per_octave = 12
    else:
        bins_per_octave = data_hp["bins_per_octave"]
    max_freq = data_hp["fmin"] * (2 ** (data_hp["n_bins"] / bins_per_octave))
    feat = _make_feat(
        query_path,
        data_hp["fmin"],
        max_freq,
        data_hp["n_bins"],
        bins_per_octave,
        data_hp["device"],
    )
    feat = feat[0:chunk_len]
    if len(feat) < chunk_len:
        feat = np.pad(
            feat,
            pad_width=((0, chunk_len - len(feat)), (0, 0)),
            mode="constant",
            constant_values=-100,
        )
    feat = shorter(feat, mean_size)
    return torch.from_numpy(feat)


def _main():
    """
    Parameters
    ----------
    See documentation at top of this file

    Returns
    -------
        list: A list of ranked reference labels, from closest to farthest.
    """
    parser = argparse.ArgumentParser(
        description="Use trained model to rank closest matches to query audio"
    )
    parser.add_argument("data_path")
    parser.add_argument("model_path")
    parser.add_argument("query_path")
    parser.add_argument("-top", default=10, type=int)
    parser.add_argument("-save", help="Path to save the query embedding as .npy file")
    parser.add_argument("-centroids", action="store_true", 
                        help="Use work_centroids.pkl instead of reference_embeddings.pkl")  # ADD THIS
    args = parser.parse_args()
    data_dir = args.data_path
    model_dir = args.model_path
    query_path = args.query_path
    top = args.top
    data_hp = load_hparams(os.path.join(data_dir, "hparams.yaml"))
    model_hp = load_hparams(os.path.join(model_dir, "config/hparams.yaml"))

    match model_hp["device"]:  # noqa requires Python 3.10
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
                model_hp["device"],
                " in your hyperparameters but that is not a valid option.",
            )
            exit()

    ## Get query embedding
    # next logic copied from eval_testset.py eval_for_map_with_feat()
    if isinstance(model_hp["chunk_frame"], list):
        infer_frame = model_hp["chunk_frame"][0] * model_hp["mean_size"]
    else:
        infer_frame = model_hp["chunk_frame"] * model_hp["mean_size"]
    chunk_s = model_hp["chunk_s"]
    # assumes 25 frames per second
    assert (
        infer_frame == chunk_s * 25
    ), f"Error for mismatch of chunk_frame and chunk_s: {infer_frame}!={chunk_s}*25"

    #   query_feat = _get_feat(local_data["feat"],hp,infer_frame)
    query_feat = _get_feat(
        query_path, data_hp, infer_frame, model_hp["mean_size"]
    )

    # unsqueeze to simulate being packaged by DataLoader as expected by the model
    query_feat = query_feat.unsqueeze(0).to(device)  # needs float()?
    with torch.no_grad():
        model = Model(model_hp).to(device)
        model.eval()
        checkpoint_dir = os.path.join(model_dir, "checkpoints")
        _ = model.load_model_parameters(checkpoint_dir, device=device)
        query_embed, _ = model.inference(query_feat)
    query_embed = query_embed.cpu().numpy()[0]

    # Save embedding if requested 
    if args.save:
        np.save(args.save, query_embed)
        print(f"Query embedding saved to: {args.save}")

    # Load reference embeddings or centroids from pickle file
    if args.centroids:
        with open(os.path.join(data_dir, "work_centroids.pkl"), "rb") as f:
            centroid_data = pickle.load(f)
            ref_embeds = centroid_data['centroids']
            radii = centroid_data.get('radii', {})  # Optional radii
    else:
        with open(os.path.join(data_dir, "reference_embeddings.pkl"), "rb") as f:
            ref_embeds = pickle.load(f)
            radii = {}

    # Calculate cosine similarity between query embedding and reference embeddings
    cos_dists = {
        label: round(cosine(query_embed, ref_embed), 6)
        for label, ref_embed in ref_embeds.items()
    }

    # Optional: Filter out matches beyond confidence radius (with floor)
    # Comment out this section if you want unfiltered results when using -centroids
    if radii and args.centroids:
        min_threshold = 0.05  # Tune this radius floor for your model
            # Necessary because single-performance works will have radius=0
        padding = .05 # Tune this "open-mindedness" percentage for your model
            # example: .05 = include candidate works within 5% farther than
            # than the confidence radius associated with that particular work centroid
        cos_dists = {
            label: dist for label, dist in cos_dists.items()
            if dist <= max(radii.get(label, min_threshold) * (1 + padding), min_threshold)
        }
    
    # Handle case where filtering removed all matches
    if not cos_dists:
        print("\nNo matches found within confidence radius thresholds.")
        print(f"Tried radius floor: {min_threshold}, padding: {padding}")
        return [], []

    # Get the top N closest reference embeddings
    top_n = nsmallest(top, cos_dists.items(), key=lambda x: x[1])
    if not top_n:  # Handle empty results
        return [], []

    top_n_labels, top_n_distances = zip(*top_n)

    return top_n_labels, top_n_distances


if __name__ == "__main__":
    labels, distances = _main()
    table = zip(labels, distances)
    print(tabulate(table, headers=["Label", "Distance"], tablefmt="grid"))
    pass
