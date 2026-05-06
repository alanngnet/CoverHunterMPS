#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Score calibration for work identification embeddings.

Given a set of reference embeddings already grouped by work, sample
cross-work (unrelated) performance pairs, compute their cosine distances,
and summarize the resulting "negative distribution." The summary lets us
convert a raw cosine distance into a model-portable score:

    z_score(d, cal) = (d - mu) / sigma
        Standard score under the negative distribution.
        Strongly negative -> match-like distance.

    pvalue(d, cal)   ~ ECDF_neg(d)
        Empirical P(D <= d | unrelated), interpolated between stored
        quantile anchors. Small -> unlikely under null of "unrelated
        tunes" -> strong match.

Calibration is meaningful only in the geometry actually used at query
time. This module assumes unit-normalized embeddings and cosine distance
(1 - dot). Do not call compute_calibration on raw embeddings; the result
would not transfer to inference.

Calibration record schema (dict, pickle-friendly):

    {
        "mu":              float,        # mean of negative-pair distances
        "sigma":           float,        # std of negative-pair distances
        "ecdf_quantiles":  np.ndarray,   # shape (n_quantiles,), float32,
                                         # sorted ascending; the i-th entry
                                         # is the distance value at which
                                         # the empirical CDF equals
                                         # ecdf_probs[i].
        "ecdf_probs":      np.ndarray,   # shape (n_quantiles,), float32,
                                         # ascending probability anchors
                                         # log-spaced toward 0 and 1 for
                                         # tail resolution. Paired with
                                         # ecdf_quantiles for pvalue lookup.
        "n_pairs":         int,          # how many pairs were sampled
        "n_quantiles":     int,          # len(ecdf_quantiles)
    }

@author: alanngnet with Claude Opus 4.7 2026-05-03

"""

import numpy as np
from typing import Dict, List


def _log_spaced_probs(n: int) -> np.ndarray:
    """Probabilities for ECDF anchors, log-spaced toward both 0 and 1.

    Half the points concentrate near 0 (log-spaced from ~1/n_pairs up to
    0.5); the other half mirror them near 1. This puts resolution where
    we need it for confidence reporting (very small p, very large p) and
    saves storage in the bulk.
    """
    half = n // 2
    # Lower tail: log-spaced from 1e-6 (or finer if n_pairs supports it)
    # up to 0.5. The 1e-6 floor matches what we can resolve with the
    # n_pairs=50000 default; finer is reported as 0 by searchsorted.
    lower = np.logspace(-6, np.log10(0.5), half, endpoint=False)
    upper = 1.0 - lower[::-1]
    middle = np.array([0.5])
    return np.concatenate([lower, middle, upper])


def compute_calibration(
    embeddings: Dict[str, np.ndarray],
    work_to_perfs: Dict[str, List[str]],
    n_pairs: int = 50_000,
    n_quantiles: int = 1001,
    seed: int = 12345,
) -> Dict:
    """Sample cross-work performance pairs, compute their cosine distances,
    and return a calibration record.

    Assumes embeddings are already L2-normalized (unit vectors). Cosine
    distance is computed as 1 - dot.
    
    Args:
        embeddings: dict perf_id -> unit-normalized embedding vector.
        work_to_perfs: dict work_id -> list of perf_ids belonging to that
            work. Used to ensure sampled pairs are cross-work.
        n_pairs: target number of cross-work pairs to sample. Sub-second
            at 50k for typical CSI scales; raise for very large libraries
            if the ECDF tails matter.
        n_quantiles: resolution of the stored empirical CDF. 1001 gives
            0.1% resolution, ~4 KB at float32.
        seed: RNG seed for reproducibility. Also impacts model fingerprinting
            strategy in make_embeds, so don't change this within a model's lifetime.

    Returns:
        Calibration record dict per the schema in the module docstring.
    """
    rng = np.random.default_rng(seed)

    # Stack perfs into a contiguous matrix; remember work assignment per row
    perf_ids = list(embeddings.keys())
    n_perfs = len(perf_ids)
    if n_perfs < 2:
        raise ValueError(
            f"Cannot calibrate from {n_perfs} perfs; need at least 2."
        )

    perf_to_work = {}
    for work_id, perfs in work_to_perfs.items():
        for p in perfs:
            perf_to_work[p] = work_id
    works_arr = np.array([perf_to_work[p] for p in perf_ids], dtype=object)

    # If every perf belongs to the same work, calibration is impossible.
    n_distinct_works = len(set(works_arr.tolist()))
    if n_distinct_works < 2:
        raise ValueError(
            "Cannot calibrate: all reference perfs belong to one work."
        )

    # Stack embeddings as (N, D) float32 matrix for fast dot products.
    D = len(embeddings[perf_ids[0]])
    matrix = np.empty((n_perfs, D), dtype=np.float32)
    for i, p in enumerate(perf_ids):
        matrix[i] = embeddings[p]

    # Rejection-sample pairs: draw two indices, keep if cross-work.
    # Oversample to amortize rejection overhead (rejection rate is the
    # fraction of pairs that are same-work, which for a typical CSI
    # library is small).
    #
    # Estimate cross-work yield to choose a strategy. For balanced
    # datasets this is high (~1 - 1/n_works); for severely imbalanced
    # ones it can collapse, in which case rejection sampling is wasteful
    # and we enumerate instead.
    total_pairs_possible = n_perfs * (n_perfs - 1) // 2
    if total_pairs_possible <= max(n_pairs * 4, 100_000):
        # Small dataset: enumerate all cross-work pairs, then sample.
        ii_all, jj_all = np.triu_indices(n_perfs, k=1)
        cross_mask = works_arr[ii_all] != works_arr[jj_all]
        ii_all = ii_all[cross_mask]
        jj_all = jj_all[cross_mask]
        n_available = len(ii_all)
        if n_available == 0:
            raise ValueError(
                "Cannot calibrate: no cross-work pairs exist in the "
                "reference set. Check that work_to_perfs covers more "
                "than one work."
            )
        actual_n = min(n_pairs, n_available)
        if actual_n < n_pairs:
            print(
                f"Warning: requested {n_pairs} calibration pairs but "
                f"only {n_available} cross-work pairs exist; using "
                f"{actual_n}."
            )
        sel = rng.choice(n_available, size=actual_n, replace=False)
        ii = ii_all[sel]
        jj = jj_all[sel]
        dots = np.einsum("nd,nd->n", matrix[ii], matrix[jj], dtype=np.float64)
        dots = np.clip(dots, -1.0, 1.0)
        distances = (1.0 - dots).astype(np.float32)
        n_pairs = actual_n  # update for record-keeping
    else:
        # Large dataset: rejection sampling, with a hard iteration cap
        # to fail loud rather than spin if cross-work yield is pathological.
        distances = np.empty(n_pairs, dtype=np.float32)
        filled = 0
        max_iterations = 100
        iteration = 0
        while filled < n_pairs:
            iteration += 1
            if iteration > max_iterations:
                raise RuntimeError(
                    f"Calibration sampling failed: only {filled} of "
                    f"{n_pairs} cross-work pairs sampled after "
                    f"{max_iterations} iterations. The reference set "
                    f"may be severely imbalanced (one work dominating). "
                    f"Inspect work_to_perfs distribution."
                )
            need = n_pairs - filled
            draw = max(need * 2, 1024)
            i = rng.integers(0, n_perfs, size=draw)
            j = rng.integers(0, n_perfs, size=draw)
            keep_mask = (i != j) & (works_arr[i] != works_arr[j])
            ii = i[keep_mask][:need]
            jj = j[keep_mask][:need]
            if len(ii) == 0:
                continue
            dots = np.einsum(
                "nd,nd->n", matrix[ii], matrix[jj], dtype=np.float64
            )
            dots = np.clip(dots, -1.0, 1.0)
            d = (1.0 - dots).astype(np.float32)
            distances[filled : filled + len(ii)] = d
            filled += len(ii)

    distances = np.empty(n_pairs, dtype=np.float32)
    filled = 0
    # Empirically generous oversample factor; loop will re-draw if short.
    while filled < n_pairs:
        need = n_pairs - filled
        draw = max(need * 2, 1024)
        i = rng.integers(0, n_perfs, size=draw)
        j = rng.integers(0, n_perfs, size=draw)
        # Reject self-pairs and same-work pairs
        keep_mask = (i != j) & (works_arr[i] != works_arr[j])
        ii = i[keep_mask][:need]
        jj = j[keep_mask][:need]
        if len(ii) == 0:
            continue
        # Cosine distance for unit vectors: 1 - dot product
        # Accumulate in float64 for ECDF accuracy at the tails;
        # this runs once per make_embeds and is not on any hot path.
        # Clip guards against floating-point overshoot of [-1, 1] for
        # near-identical or near-antipodal unit vectors.
        dots = np.einsum("nd,nd->n", matrix[ii], matrix[jj], dtype=np.float64)
        dots = np.clip(dots, -1.0, 1.0)
        d = (1.0 - dots).astype(np.float32)
        distances[filled : filled + len(ii)] = d
        filled += len(ii)

    mu = float(distances.mean())
    sigma = float(distances.std(ddof=1))

    # ECDF stored as sorted quantiles with log-spaced probabilities so that
    # the distribution is dense at both tails (high confidence and no-match).
    # Lookup at query time is a binary search.
    sorted_d = np.sort(distances)
    probs = _log_spaced_probs(n_quantiles)
    quantile_idx = np.clip(
        (probs * (n_pairs - 1)).round().astype(np.int64), 0, n_pairs - 1
    )
    ecdf_quantiles = sorted_d[quantile_idx].astype(np.float32)
    ecdf_probs = probs.astype(np.float32)  # store the prob anchors

    return {
        "mu": mu,
        "sigma": sigma,
        "ecdf_quantiles": ecdf_quantiles,
        "ecdf_probs": ecdf_probs,
        "n_pairs": int(n_pairs),
        "n_quantiles": int(n_quantiles),
    }


def z_score(distance, calibration: Dict):
    """Standard score under the negative distribution.

    Strongly negative values mean the distance is much smaller than
    typical for unrelated pairs - i.e. a match-like distance.
    A z-score of 0 means the query looks like a random unrelated pair.

    Accepts a scalar or numpy array of distances; returns the same shape.
    """
    d = np.asarray(distance)
    return (d - calibration["mu"]) / calibration["sigma"]


def pvalue(distance, calibration: Dict):
    """Empirical P(D <= distance | unrelated pair).

    Small p-value means the distance is unlikely under the null of
    unrelated tunes - i.e. a strong match. Bounded in [0, 1].

    Accepts a scalar or numpy array of distances; returns the same shape.
    """
    quantiles = calibration["ecdf_quantiles"]
    probs = calibration["ecdf_probs"]
    d = np.asarray(distance)
    # np.interp does linear interpolation between anchors; both arrays
    # must be ascending, which they are by construction.
    return np.clip(np.interp(d, quantiles, probs), 0.0, 1.0)
