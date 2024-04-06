#!/usr/bin/env python3
"""
Created on Sat Mar  2 17:31:23 2024
@author: Alan Ng

Command-line utility to identify closest matches to the target audio
from the model's training data. 

MVP proof-of-concept. Assumptions:
    target has a duration < chunk_s (135sec).
    You already ran tools/eval_testset.py to generate reference embeddings 
        in the model's "embed_...tmp/query_embed" folder

Example:
python -m tools.identify egs/covers80 target.txt -top=10

model_path is the relative path that expects a subfolder "pt_model" containing checkpoint files and the abovementioned tmp/query_embed/ folder of reference embeddings.

query_path is a relative path to the metadata description of the audio you want to identify, following the same format as the full.txt file generated by extract_csi_features.py

"""
import os, torch, numpy as np
from src.model import Model
from src.pytorch_utils import get_latest_model
from src.cqt import shorter
from src.utils import (
    load_hparams,
    RARE_DELIMITER,
    line_to_dict,
    read_lines,
)
import argparse
from heapq import nsmallest
from scipy.spatial.distance import cosine
from tabulate import tabulate


def _get_feat(feat_path,hp,chunk_len):
    """
    adapted from dataset.py AudioFeatDataset::__getitem__()
    assumes "mode" = "defined" and start = 0
    """
    feat = np.load(feat_path)
    feat = feat[0:chunk_len]
    if len(feat) < chunk_len:
        feat = np.pad(
            feat,
            pad_width=((0, chunk_len - len(feat)), (0, 0)),
            mode="constant",
            constant_values=-100,
        )
    feat = shorter(feat, hp["mean_size"])
    return torch.from_numpy(feat)


def _load_ref_embeds(ref_lines):
    """
    adapted from src/eval_testset.py _load_chunk_embed_from_dir()
    
    returns dictionary of "label" -> embedding.npy associations
    """
    ref_embeds = {}
    for line in ref_lines:
        local_data=line_to_dict(line)
        utt = local_data["utt"].split(f"-{RARE_DELIMITER}start-")[0]
        label = local_data["song_id"]
        ref_embeds[label] = np.load(local_data["embed"])
    return ref_embeds


def _main():
    """
    Args:
        path_to_model (nnet model): The trained model to use for embeddings.
        query_path (str): Path to the individual query file.
        -top (int): optional. Return how many of the closest matches
        

    Returns:
        list: A list of ranked reference labels, from closest to farthest.
        
    
    """
    parser = argparse.ArgumentParser(
      description="use model to rank closest matches to input CQT")
    parser.add_argument('model_path')
    parser.add_argument('query_path')
    parser.add_argument("-top", default=0, type=int)
    args = parser.parse_args()
    model_dir = args.model_path
    query_path = args.query_path
    top = args.top
    hp = load_hparams(os.path.join(model_dir, "config/hparams.yaml"))

    match hp['device']:  # noqa requires Python 3.10
      case 'mps':
          assert torch.backends.mps.is_available(), "You requested 'mps' device in your hyperparameters but you are not running on an Apple M-series chip or have not compiled PyTorch for MPS support."
          device = torch.device('mps')
      case 'cuda':
          assert torch.cuda.is_available(), "You requested 'cuda' device in your hyperparameters but you do have a CUDA-compatible GPU available."
          device = torch.device('cuda')
      case _:
          print("You set device: ",hp['device']," in your hyperparameters but that is not a valid option.")
          exit();

  
    # Get query embedding
    query_lines = read_lines(query_path)
    # assume only one member of query_lines
    local_data = line_to_dict(query_lines[0])
    # next logic copied from eval_testset.py eval_for_map_with_feat()
    if isinstance(hp["chunk_frame"], list):
      infer_frame = hp["chunk_frame"][0] * hp["mean_size"]
    else:
      infer_frame = hp["chunk_frame"] * hp["mean_size"]
    chunk_s = hp["chunk_s"]
    # assumes 25 frames per second
    assert infer_frame == chunk_s * 25, \
      "Error for mismatch of chunk_frame and chunk_s: {}!={}*25".format(
    infer_frame, chunk_s)
    query_feat = _get_feat(local_data["feat"],hp,infer_frame)

    # unsqueeze to simulate being packaged by DataLoader as expected by the model
    query_feat = query_feat.unsqueeze(0).to(device)  # needs float()?
    with torch.no_grad():
        model = Model(hp).to(device)
        model.eval()
        checkpoint_dir = os.path.join(model_dir, "pt_model")
        epoch = model.load_model_parameters(checkpoint_dir, device=device)
        query_embed, _ = model.inference(query_feat)
    query_embed = query_embed.cpu().numpy()[0]

    # Get ref embeddings. Adapted from _cut_lines_with_dur()
    ref_dict = []
    embed_dir = os.path.join(model_dir, "embed_{}_{}".format(epoch, "tmp"))
    ref_lines = read_lines(os.path.join(embed_dir,"ref.txt"))
    ref_embeds = _load_ref_embeds(ref_lines)
    top = len(ref_embeds) if top == 0 else top

    # Calculate cosine similarity between query embedding and reference embeddings
    cos_dists = {label: round(cosine(query_embed, ref_embed),6) 
                 for label, ref_embed in ref_embeds.items()}

    # Get the top N closest reference embeddings
    top_n = nsmallest(top, cos_dists.items(), key=lambda x: x[1])
    top_n_labels, top_n_distances = zip(*top_n)
    
    return top_n_labels, top_n_distances


if __name__ == '__main__':
    labels, distances = _main()
    table = zip(labels, distances)
    print(tabulate(table, headers=["Label", "Distance"], tablefmt="grid"))
    pass