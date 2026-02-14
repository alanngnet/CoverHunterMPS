#!/usr/bin/env python3
# author:liufeng
# datetime:2023/1/13 11:07 AM
# software: PyCharm

import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.dataset import AudioFeatDataset
from src.map import calc_map
from src.utils import (
    RARE_DELIMITER,
    dict_to_line,
    line_to_dict,
    read_lines,
    write_lines,
)


def _cluster_plot(
    dist_matrix,
    ref_labels,
    output_path,
    test_only_labels=None,
    marks="markers",
    logger=None,
) -> None:
    """
    Generate t-SNE clustering PNG plot.

    Args:
        dist_matrix: must be square array
        test_only_labels: is an optional list of labels (work_ids) that will be
            marked in the plot to visually distinguish samples of work IDs that
            were excluded from the training and validation datasets.
        marks: "markers": cycle through marker-color combinations to reuse them
            as little as possible, and to maximize color differentiation.
            "ids": use work_id numbers as markers
    """
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    if test_only_labels is None:
        test_only_labels = []
    model = TSNE(
        n_components=2,
        init="random",
        random_state=0,
    )  # Adjust parameters as needed
    embedding = model.fit_transform(dist_matrix)

    unique_labels = np.unique(ref_labels)
    cmap_name = "hsv"  # use any pyplot palette you like
    # See https://matplotlib.org/stable/api/markers_api.html for style definitions
    marker_styles = ["o", "s", "^", "p", "x", "D"]
    num_colors = len(unique_labels) // len(marker_styles)
    colors = plt.get_cmap(cmap_name, num_colors)(range(num_colors))
    plt.figure(figsize=(15, 15))

    color_dict = {}  # Dictionary to store color for each label
    marker_dict = {}  # Dictionary to store marker style for each label

    if marks == "ids":
        marker_styles = [f"${label}$" for label in unique_labels]

    for i, label in enumerate(unique_labels):
        # Assign color for label
        color_dict[label] = colors[i % num_colors]
        # Assign marker style for label
        # Assign marker style for label
        if marks == "markers":
            marker_dict[label] = marker_styles[i % len(marker_styles)]
        else:
            marker_dict[label] = marker_styles[i]
    if logger:
        logger.info(f"test_only_labels: {test_only_labels}")

    for i, label in enumerate(unique_labels):
        label_indices = np.where(ref_labels == label)[0]
        color = color_dict[label]
        marker = marker_dict[label]
        if label in test_only_labels:
            logger.info(f"test_only_label: {label}")
            # Loop through each coordinate for this label
            for j in range(len(label_indices)):
                x = embedding[label_indices[j], 0]
                y = embedding[label_indices[j], 1]

                plt.gca().add_artist(
                    plt.Circle(
                        (x, y),
                        radius=0.4,
                        color="black",
                        linewidth=1,
                        fill=False,
                        zorder=2,
                    ),
                )
        plt.scatter(
            embedding[label_indices, 0],
            embedding[label_indices, 1],
            color=color,
            marker=marker,
            label=label,
        )

    plt.title("t-SNE Visualization of Clustering")
    if test_only_labels:
        plt.text(
            1,
            1.02,
            "Circles = work_ids not seen in training",
            ha="right",
            va="bottom",
            transform=plt.gca().transAxes,
        )

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def _calc_embed(model, query_loader, device, saved_dir=None):
    """calculate audio embedding

    Args:
      model: trained model
      query_loader: dataset loader for audio
      device: PyTorch device
      saved_dir: To save embed to disk as npy

    Returns:
      query_perf_label: List[(perf, label), ...],
      query_embed: Dict, key is perf, value is List with embed of every chunk

    """
    query_label = {}
    query_embed = {}
    with torch.no_grad():

        for _j, batch in enumerate(query_loader):
            perf_b, feat_b, label_b = batch
            feat_b = batch[1].float().to(device)
            label_b = batch[2].long().to(device)
            embed_b, _ = model.inference(feat_b)

            embed_b = embed_b.cpu().numpy()
            label_b = label_b.cpu().numpy()
            for idx_embed in range(len(embed_b)):
                perf = perf_b[idx_embed]
                embed = embed_b[idx_embed]
                label = label_b[idx_embed]

                assert np.shape(embed) == (
                    model.get_embed_length(),
                ), np.shape(embed)

                if perf not in query_label:
                    query_label[perf] = label
                else:
                    assert query_label[perf] == label

                if perf not in query_embed:
                    query_embed[perf] = []
                query_embed[perf].append(embed)
                if saved_dir:
                    saved_path = os.path.join(saved_dir, f"{perf}.npy")
                    np.save(saved_path, embed)
    query_perf_label = sorted(query_label.items())
    return query_perf_label, query_embed


def _compute_distance_worker(args):
    perf_query, perf_ref, query_embed, ref_embed = args
    to_choice = []
    for embed_x in query_embed[perf_query]:
        for embed_y in ref_embed[perf_ref]:
            embed_x = embed_x / np.linalg.norm(embed_x)
            embed_y = embed_y / np.linalg.norm(embed_y)
            cos_sim = embed_x.dot(embed_y)
            dist = 1 - cos_sim
            to_choice.append(dist)
    return min(to_choice)


def _generate_dist_matrixMPS(
    query_perf_label,
    query_embed,
    ref_perf_label=None,
    ref_embed=None,
    query_in_ref=None,
):
    """generate distance matrix from query and ref embeddings

    Args:
      query_perf_label: List[(perf, label), ...],
      query_embed: Dict, key is perf, value is List with embedding of every chunk
      ref_perf_label: List[(perf, label), ...]
      ref_embed: Dict, key is perf, value is List with embedding of every chunk
      query_in_ref: List[(idx, idy), ...], means query[idx] is in ref[idy] so
                    we skip that when computing mAP

    Returns:
      dist_matrix: [numpy.ndarray]
      query_label:
      ref_label:

    """
    import multiprocessing

    if ref_perf_label is None and ref_embed is None:
        query_in_ref = [(i, i) for i in range(len(query_perf_label))]
        ref_perf_label = query_perf_label
        ref_embed = query_embed

    dist_matrix = np.zeros([len(query_perf_label), len(ref_perf_label)])
    args_list = [
        (perf_query, perf_ref, query_embed, ref_embed)
        for perf_query, _ in query_perf_label
        for perf_ref, _ in ref_perf_label
    ]

    with multiprocessing.Pool() as pool:
        distances = pool.map(_compute_distance_worker, args_list)

    for (idx, idy), distance in zip(
        [
            (idx, idy)
            for idx in range(len(query_perf_label))
            for idy in range(len(ref_perf_label))
        ],
        distances,
    ):
        dist_matrix[idx, idy] = distance

    if query_in_ref:
        for idx, idy in query_in_ref:
            dist_matrix[idx, idy] = -1  # will be skipped when computing mAP

    query_label = [v for k, v in query_perf_label]
    ref_label = [v for k, v in ref_perf_label]

    return dist_matrix, query_label, ref_label


# =============================================================================
# The original CoverHunter distance matrix function was very slow at scale.
# It did not use multiprocessing (only one core of the CPU).
# During use of the alignment_for_frame script, on a M2 Max chip, the old
# version took 12 minutes vs. 1.5 minutes for this new version.
# =============================================================================


def _load_data_from_dir(query_chunked_lines):
    query_perf_label = []
    query_embed = {}
    perf_set = set()
    for line in query_chunked_lines:
        local_data = line_to_dict(line)
        perf = local_data["perf"].split(f"-{RARE_DELIMITER}start-")[0]
        label = local_data["work_id"]
        if perf not in perf_set:  # exclude duplicate perfs
            query_perf_label.append((perf, label))
            perf_set.add(perf)
            query_embed[perf] = []
        query_embed[perf].append(np.load(local_data["embed"]))
    return query_perf_label, query_embed


def _cut_one_line_with_dur(
    line, window_length_s, window_shift_s, hop_size=0.04
):
    """cut line with window_length_s

    If duration is smaller than window_length_s, return only one chunk.

    Args:
      line:
      window_length_s:
      window_shift_s:

    Returns:
      lines with "start" info

    """
    local_data = line_to_dict(line)
    perf = local_data["perf"]
    if "dur_s" in local_data:
        dur_s = local_data["dur_s"]
    else:
        dur_s = local_data["dur_ms"] / 1000

    short_lines = []
    start_s = 0.0
    while start_s + window_length_s < dur_s or start_s == 0:
        local_data["start_s"] = start_s
        local_data["start"] = int(start_s * 25)
        local_data["perf"] = f"{perf}-{RARE_DELIMITER}start-{int(start_s)}"
        short_lines.append(dict_to_line(local_data))
        start_s += window_shift_s
    return short_lines


def _cut_lines_with_dur(init_lines, chunk_s, embed_dir):
    """
    Cut lines with duration: For every init_line that is longer than chunk_s,
    split into lines of max length chunk_s.
    Add "start" to all lines for chunk starting point relative to original perf.

    Args:
      init_lines: list of all lines
      chunk_s: float, chunk duration in seconds
      embed_dir: embed dir to save embed array

    Returns:
      chunk_lines: init_lines split as needed to respect chunk_s, and then
      modified by _cut_one_line_with_dur
    """
    os.makedirs(embed_dir, exist_ok=True)
    chunk_lines = []
    for line in init_lines:
        short_lines = _cut_one_line_with_dur(
            line,
            window_length_s=chunk_s,
            window_shift_s=chunk_s,
        )
        for short_line in short_lines:
            local_data = line_to_dict(short_line)
            local_data["embed"] = os.path.join(
                embed_dir,
                "{}.npy".format(local_data["perf"]),
            )
            chunk_lines.append(dict_to_line(local_data))
    return chunk_lines


def eval_for_map_with_feat(
    hp,
    model,
    embed_dir,
    query_path,
    ref_path,
    query_in_ref_path=None,
    batch_size=128,
    num_workers=1,
    device="mps",
    logger=None,
    plot_name="",
    marks="markers",
    dist_name="",
    test_only_labels=None,
    reuse_embeddings=False,
    bootstrap=0,
):
    """compute map10 with trained model and query/ref loader(dataset loader
    can speed up process dramatically)

    Args:
      num_workers:
      hp: dict contains hparams
      model: nnet model, should have method 'infer'
      embed_dir: dir for saving embedding, None for not saving anything
      query_path: text file with query perf info
      ref_path: text file with ref perf info
      query_in_ref_path: path to prepared query in ref index. None means that
          query index equals ref index
      batch_size: for nnet infer
      device: "mps" or "cuda" or "cpu"
      logger:
      plot_name: if a path is provided, save t-SNE plot there
      dist_name: if a path is provided, save dist_matrix there
      test_only_labels: see explanation in _cluster_plot()
      marks: see explanation in _cluster_plot()
      bootstrap: int, number of work-stratified bootstrap iterations for
          confidence intervals on mAP, MR1, and hit_rate. 0 disables.
          Resamples works (not performances) with replacement to correctly
          capture inter-work variance. Operates on the pre-computed distance
          matrix, so computational cost is negligible. Typical value: 1000.

    Returns:
      map10
      hitrate
      rank1

    """
    if test_only_labels is None:
        test_only_labels = []
    if logger:
        logger.info("=" * 40)
        logger.info("Start to Eval")
        logger.info(f"query_path: {query_path}")
        logger.info(f"ref_path: {ref_path}")
        logger.info(f"query_in_ref_path: {query_in_ref_path}")
        logger.info(f"using batch-size: {batch_size}")

    os.makedirs(embed_dir, exist_ok=True)
    if not reuse_embeddings and os.path.exists(embed_dir):
        import shutil

        if logger:
            logger.info(f"Clearing existing embeddings directory: {embed_dir}")
        shutil.rmtree(embed_dir)
        os.makedirs(embed_dir, exist_ok=True)
    elif reuse_embeddings and os.path.exists(embed_dir):
        existing_files = [
            f
            for f in os.listdir(embed_dir)
            if f.endswith(".npy") or f.endswith(".txt")
        ]
        if logger and existing_files:
            logger.warning(
                f"REUSING existing embeddings from: {embed_dir}\n"
                f"Found {len(existing_files)} existing files.\n"
                f"This assumes data and CQT parameters haven't changed."
            )

    model.eval()
    model = model.to(device)

    if isinstance(hp["chunk_frame"], list):
        infer_frame = hp["chunk_frame"][0] * hp["mean_size"]
    else:
        infer_frame = hp["chunk_frame"] * hp["mean_size"]

    chunk_s = hp["chunk_s"]
    # assumes resolution of 25 features per second (hop length 0.04s)
    assert (
        infer_frame == chunk_s * 25
    ), f"Error for mismatch of chunk_frame and chunk_s: {infer_frame}!={chunk_s}*25"

    query_lines = read_lines(query_path, log=False)
    ref_lines = read_lines(ref_path, log=False)
    if logger:
        logger.info(f"query lines: {len(query_lines)}")
        logger.info(f"ref lines: {len(ref_lines)}")
        logger.info(f"chunk_frame: {infer_frame} chunk_s:{chunk_s}\n")

    # basic validation of the query_in_ref contents
    if query_in_ref_path:
        firstline = read_lines(query_in_ref_path, log=False)[0]
        query_in_ref = line_to_dict(firstline)["query_in_ref"]
        for idx, idy in query_in_ref:
            assert idx < len(
                query_lines,
            ), f"query idx {idx} must be smaller than max query idx {len(query_lines)}"
            assert idy < len(
                ref_lines,
            ), f"ref idx {idy} must be smaller than max ref idx {len(ref_lines)}"
    else:
        query_in_ref = None

    query_embed_dir = os.path.join(embed_dir, "query_embed")
    query_chunked_lines = _cut_lines_with_dur(
        query_lines, chunk_s, query_embed_dir
    )
    write_lines(
        os.path.join(embed_dir, "query.txt"), query_chunked_lines, False
    )
    # select query perfs for which there is not yet a saved embedding
    to_calc_lines = [
        l
        for l in query_chunked_lines
        if not os.path.exists(line_to_dict(l)["embed"])
    ]
    if logger:
        logger.info(
            f"query chunked lines: {len(query_chunked_lines)}, to compute lines: {len(to_calc_lines)}",
        )
    # generate any missing embeddings
    if len(to_calc_lines) > 0:
        data_loader = DataLoader(
            AudioFeatDataset(
                hp,
                data_lines=to_calc_lines,
                mode="defined",
                chunk_len=infer_frame,
            ),
            num_workers=num_workers,
            shuffle=False,
            sampler=None,
            batch_size=batch_size,
            pin_memory=True,
            collate_fn=None,
            drop_last=False,
        )
        _calc_embed(model, data_loader, device, saved_dir=query_embed_dir)

    ref_embed_dir = os.path.join(embed_dir, "query_embed")
    ref_chunked_lines = _cut_lines_with_dur(ref_lines, chunk_s, ref_embed_dir)
    write_lines(os.path.join(embed_dir, "ref.txt"), ref_chunked_lines, False)
    if ref_path != query_path:
        # select ref perfs for which there is not yet a saved embedding
        to_calc_lines = [
            l
            for l in ref_chunked_lines
            if not os.path.exists(line_to_dict(l)["embed"])
        ]
        if logger:
            logger.info(
                f"ref chunked lines: {len(ref_chunked_lines)}, to compute lines: {len(to_calc_lines)}",
            )
        if len(to_calc_lines) > 0:
            data_loader = DataLoader(
                AudioFeatDataset(
                    hp,
                    data_lines=to_calc_lines,
                    mode="defined",
                    chunk_len=infer_frame,
                ),
                num_workers=num_workers,
                shuffle=True,
                sampler=None,
                batch_size=batch_size,
                pin_memory=True,
                collate_fn=None,
                drop_last=False,
            )
            _calc_embed(model, data_loader, device, saved_dir=ref_embed_dir)
        if logger:
            logger.info(
                f"Finish computing ref embedding, saved at {ref_embed_dir}\n",
            )
    elif logger:
        logger.info(
            "Because query and ref have the same path, "
            "skip computing the ref embeddings",
        )

    query_perf_label, query_embed = _load_data_from_dir(query_chunked_lines)
    if ref_path == query_path:
        ref_perf_label, ref_embed = None, None
    else:
        ref_perf_label, ref_embed = _load_data_from_dir(ref_chunked_lines)
    if logger:
        logger.info(
            "Finished loading embedding. Start to compute distance matrix"
        )

    # parallelized version for CoverHunterMPS
    dist_matrix, query_label, ref_label = _generate_dist_matrixMPS(
        query_perf_label,
        query_embed,
        ref_perf_label,
        ref_embed,
        query_in_ref=query_in_ref,
    )

    if dist_name:
        path = os.path.dirname(dist_name)
        if path != "":
            assert os.path.isdir(path), f"Invalid dist_name path: {plot_name}"
        if not dist_name.endswith(".npy"):
            label_path = dist_name + ".reflabels"
            dist_name += ".npy"
        else:
            label_path = dist_name[:-4] + ".reflabels"
        np.save(dist_name, dist_matrix)
        np.save(label_path, ref_label)
        logger.info(f"distance matrix saved to: {dist_name}")
        logger.info(f"ref labels saved to: {label_path}")

    if plot_name:
        path = os.path.dirname(plot_name)
        if path != "":
            assert os.path.isdir(path), f"Invalid plot path: {plot_name}"

        _cluster_plot(
            dist_matrix, ref_label, plot_name, test_only_labels, marks, logger
        )
        logger.info(f"t-SNE plot saved to: {plot_name}")

    if logger:
        logger.info(
            "Finish computing distance matrix and Start to compute map"
        )
        logger.info(
            f"Inp dist shape: {np.shape(dist_matrix)}, query: {len(query_label)}, ref: {len(ref_label)}",
        )

    metrics = calc_map(
        dist_matrix, query_label, ref_label, topk=10000, verbose=0
    )
    if logger:
        logger.info("map: {}".format(metrics["mean_ap"]))
        logger.info("rank1: {}".format(metrics["rank1"]))
        logger.info("hit_rate: {}\n".format(metrics["hit_rate"]))

    if bootstrap:
        from src.map import bootstrap_metrics

        boot = bootstrap_metrics(
            dist_matrix,
            query_label,
            ref_label,
            n_bootstrap=bootstrap,
            seed=12345,
        )
        if logger:
            for key in ("mean_ap", "rank1", "hit_rate"):
                b = boot[key]

                logger.info(
                    "Bootstrap %s: %.4f [%.4f, %.4f] (±%.4f) n=%d",
                    key,
                    b["point"],
                    b["ci_low"],
                    b["ci_high"],
                    b["std"],
                    bootstrap,
                )

    return metrics["mean_ap"], metrics["hit_rate"], metrics["rank1"]


if __name__ == "__main__":
    pass
