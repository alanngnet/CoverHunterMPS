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
