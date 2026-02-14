#!/usr/bin/env python3
# author:liufeng
# datetime:2022/7/15 9:19 AM
# software: PyCharm


import logging
from typing import Dict, List

import numpy as np


LOGGER = logging.getLogger(__name__)


def calc_map(
    array2d: np.ndarray,
    label_query: List,
    label_ref: List,
    topk: int = 10,
    verbose: int = 0,
) -> Dict:
    """calculate map@10, top10, rank1, hit_rate

    Args:
      array2d: matrix for distance. Note: negative value will be excluded.
      label_query: query label
      label_ref: ref label
      topk: k value for map, usually we set topk 10, if topk is set 1,
            map@1 equals precision
      verbose: logging level, 0 for no log, 1 for print ap of every query

    Returns:
      MAP, top10, rank1, hit_rate

    Notes:
      P@k: for a given query and k as a rank# within ranked query results,
           top k results have a total of (P@k) positive samples (correct classifications)
      AP@k: = A(P@k), for given k, if the kth member of ranked results is correct, choose it.
              A(P@k) is to compute average of chosen results
      MAP@k: mean of (Ap@k) across all classes

    References:
      mAP definition: https://towardsdatascience.com/breaking-down-mean-average-precision-map-ae462f623a52#f9ce
    """
    query_num, ref_num = np.shape(array2d)
    new_array2d = []
    for u, row in enumerate(array2d):
        row = [(v, col) for (v, col) in enumerate(row) if col >= 0]
        new_array2d.append(row)

    mean_ap, top10, rank1 = 0, 0, 0
    for u, row in enumerate(new_array2d):
        row = sorted(row, key=lambda x: x[1])
        per_top10, per_rank1, per_map = 0, 0, 0
        version_cnt = 0.0
        for k, (v, _val) in enumerate(row):
            if k >= topk:
                continue
            if label_query[u] == label_ref[v]:
                if k < topk:
                    version_cnt += 1
                    per_map += version_cnt / (k + 1)
                if per_rank1 == 0:
                    per_rank1 = k + 1
                if k < 10:
                    per_top10 += 1

        if per_rank1 == 0:
            for k, (v, _val) in enumerate(row):
                if label_query[u] == label_ref[v]:
                    if per_rank1 == 0:
                        per_rank1 = k + 1

        if version_cnt > 0:
            per_map = per_map / version_cnt

        if verbose > 0:  # added filter to make logging output more readable
            LOGGER.info("XX per_rank1: %d", per_rank1)
        if verbose > 0:
            top5_res = [x for x, _ in row][:5]
            LOGGER.debug(
                "Debug:: %dth, query work: %s, map: %s rank1: %s, top5: %s",
                u,
                label_query[u],
                per_map,
                per_rank1,
                top5_res,
            )

        mean_ap += per_map
        top10 += per_top10
        rank1 += per_rank1

    mean_ap = mean_ap / query_num
    top10 = top10 / query_num / 10
    rank1 = rank1 / query_num

    hit_rate = 0
    for u, row in enumerate(new_array2d):
        row = sorted(row, key=lambda x: x[1])
        if len(row) == 0:
            continue
        v, val = row[0]
        if label_query[u] == label_ref[v]:
            hit_rate += 1
    hit_rate = hit_rate / query_num
    return {"mean_ap": mean_ap, "top10": top10, "rank1": rank1, "hit_rate": hit_rate}


def _fast_calc_metrics(
    dist_matrix: np.ndarray, label_query: List, label_ref: List
) -> tuple:
    """Numpy-vectorized mAP, MR1, and hit_rate for bootstrap inner loop.

    Equivalent to calc_map() but ~100x faster by avoiding Python loops
    over matrix elements. Still loops over queries (unavoidable for
    per-query AP) but uses vectorized numpy operations within each.

    Args:
      dist_matrix: [n_query, n_ref] distance matrix. Negative values excluded.
      label_query: Work labels for query axis.
      label_ref: Work labels for reference axis.

    Returns:
      (mean_ap, rank1, hit_rate) as floats
    """
    n_query = len(label_query)
    ref_arr = np.array(label_ref)

    masked = np.where(dist_matrix >= 0, dist_matrix, np.inf)
    rankings = np.argsort(masked, axis=1)

    sum_ap = 0.0
    sum_rank1 = 0.0
    sum_hit = 0.0

    for i in range(n_query):
        row = rankings[i]
        valid_mask = masked[i, row] < np.inf
        row = row[valid_mask]
        if len(row) == 0:
            continue

        matches = ref_arr[row] == label_query[i]
        n_relevant = matches.sum()
        if n_relevant == 0:
            continue

        cumhits = np.cumsum(matches)
        positions = np.arange(1, len(row) + 1)
        ap = (cumhits[matches] / positions[matches]).sum() / n_relevant

        first_match = np.argmax(matches)
        sum_rank1 += first_match + 1
        sum_hit += 1.0 if matches[0] else 0.0
        sum_ap += ap

    return sum_ap / n_query, sum_rank1 / n_query, sum_hit / n_query


def bootstrap_metrics(
    dist_matrix: np.ndarray,
    label_query: List,
    label_ref: List,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 12345,
    topk: int = 10000,
) -> Dict:
    """Work-stratified bootstrap confidence intervals for mAP, MR1, hit_rate.

    Resamples works (not performances) with replacement to correctly capture
    inter-work variance — the dominant source of metric uncertainty in CSI
    evaluation on small testsets.

    Args:
        dist_matrix: Pre-computed distance matrix (query × ref), as used by calc_map.
                     Negative values are excluded per calc_map convention.
        label_query: Work labels for query axis.
        label_ref: Work labels for reference axis.
        n_bootstrap: Number of bootstrap iterations.
        confidence: Confidence level for intervals (default 0.95 → 95% CI).
        seed: Random seed for reproducibility.
        topk: Passed through to calc_map.

    Returns:
        Dict with keys for each metric (mean_ap, rank1, hit_rate), each containing:
            "point": Point estimate from full dataset
            "mean": Bootstrap mean
            "ci_low": Lower confidence bound
            "ci_high": Upper confidence bound
            "std": Bootstrap standard deviation
    """
    rng = np.random.default_rng(seed)

    # Point estimates from full dataset
    point = calc_map(dist_matrix, label_query, label_ref, topk=topk, verbose=0)

    # Group query indices by work
    work_to_query_idx = {}
    for i, w in enumerate(label_query):
        work_to_query_idx.setdefault(w, []).append(i)

    # All unique works present in query set (ref set remains full)
    works = sorted(work_to_query_idx.keys())
    n_works = len(works)

    boot_results = {"mean_ap": [], "rank1": [], "hit_rate": []}

    for _ in range(n_bootstrap):
        # Resample works with replacement for queries only
        sampled_works = rng.choice(works, size=n_works, replace=True)

        # Gather query indices with original work labels
        query_indices = []
        query_labels = []

        for w in sampled_works:
            for qi in work_to_query_idx[w]:
                query_indices.append(qi)
                query_labels.append(w)  # Use original work label

        # Keep FULL reference set - no subsampling
        # Extract only query rows from distance matrix (all ref columns)
        sub_matrix = dist_matrix[query_indices, :]

        b_map, b_mr1, b_hit = _fast_calc_metrics(sub_matrix, query_labels, label_ref)
        boot_results["mean_ap"].append(b_map)
        boot_results["rank1"].append(b_mr1)
        boot_results["hit_rate"].append(b_hit)



    # Compute intervals
    alpha = (1 - confidence) / 2
    results = {}
    for key in boot_results:
        samples = np.array(boot_results[key])
        results[key] = {
            "point": point[key],
            "mean": float(np.mean(samples)),
            "ci_low": float(np.percentile(samples, 100 * alpha)),
            "ci_high": float(np.percentile(samples, 100 * (1 - alpha))),
            "std": float(np.std(samples)),
        }

    return results