#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze TensorBoard logs produced by tools/train_tune.py to report best epochs by mAP scores across k-fold training.

Usage: python -m tools.analyze_tb_logs <logs_dir> --runid <runid>
       e.g., python -m tools.analyze_tb_logs training/SHS100K/logs

Configuration: use TESTSETS below to define which testset names have the mAP scores.

Created on Mon Dec 8 08:20:39 2025
@author: alanngnet and Claude Opus 4.5

"""

import os
import sys
import argparse
import re
from tensorboard.backend.event_processing.event_accumulator import (
    EventAccumulator,
)

TESTSETS = ["covers80", "SHS100K-TEST"]


def find_fold_dirs(logs_dir, runid):
    """Find all fold directories including 'full' if present for given runid."""
    fold_dirs = []
    for name in os.listdir(logs_dir):
        path = os.path.join(logs_dir, name)
        if os.path.isdir(path):
            if re.match(rf"^{re.escape(runid)}_fold_\d+$", name):
                fold_dirs.append((name, path))
            elif name == f"{runid}_full":
                fold_dirs.append((name, path))
    if not fold_dirs:
        print(f"No directories matching '{runid}_fold_*' found in {logs_dir}")
        sys.exit(1)
    # Sort: runid_fold_1, runid_fold_2, ..., runid_full
    fold_dirs.sort(key=lambda x: (x[0].endswith("_full"), x[0]))
    return fold_dirs


def load_fold_data(fold_name, fold_path):
    """Load mAP data from a single fold's event files."""
    event_files = sorted(
        [f for f in os.listdir(fold_path) if "events.out.tfevents" in f],
        key=lambda x: os.path.getctime(os.path.join(fold_path, x)),
    )
    if not event_files:
        return {}, None, None

    event_file = os.path.join(fold_path, event_files[-1])
    ea = EventAccumulator(event_file)
    ea.Reload()

    data = {}  # testset -> {epoch: (value, wall_time)}
    min_time = None
    max_time = None

    for testset in TESTSETS:
        tag = f"mAP/{testset}"
        try:
            events = ea.Scalars(tag)
            data[testset] = {}
            for e in events:
                if min_time is None or e.wall_time < min_time:
                    min_time = e.wall_time
                if max_time is None or e.wall_time > max_time:
                    max_time = e.wall_time
                data[testset][e.step] = (e.value, e.wall_time)
        except KeyError:
            pass

    return data, min_time, max_time


def load_all_folds(logs_dir, runid):
    """Load mAP data from all folds, adjusting times to be cumulative."""
    fold_dirs = find_fold_dirs(logs_dir, runid)

    if not fold_dirs:
        print(f"No fold directories found in {logs_dir}")
        sys.exit(1)

    print(f"Found folds: {[f[0] for f in fold_dirs]}")

    # First pass: get base time from fold_1
    all_data = []  # list of (fold_name, testset, epoch, value, wall_time)
    global_base_time = None

    for fold_name, fold_path in fold_dirs:
        fold_data, min_time, max_time = load_fold_data(fold_name, fold_path)
        if not fold_data:
            print(f"Warning: No data in {fold_name}")
            continue

        if global_base_time is None:
            global_base_time = min_time

        for testset, epochs in fold_data.items():
            for epoch, (value, wall_time) in epochs.items():
                elapsed = wall_time - global_base_time
                all_data.append((fold_name, testset, epoch, value, elapsed))

    return all_data


def format_time(seconds):
    """Format seconds as HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def get_top_by_testset(all_data, testset, n=3):
    """Get top n entries for a single testset."""
    filtered = [(f, e, v, t) for f, ts, e, v, t in all_data if ts == testset]
    filtered.sort(key=lambda x: x[2], reverse=True)  # sort by value
    return filtered[:n]


def get_top_by_avg(all_data, n=3):
    """Get top n entries by average mAP across all testsets."""
    # Group by (fold, epoch)
    grouped = {}
    for fold, testset, epoch, value, elapsed in all_data:
        key = (fold, epoch)
        if key not in grouped:
            grouped[key] = {"values": {}, "elapsed": elapsed}
        grouped[key]["values"][testset] = value

    # Calculate averages for entries with all testsets
    avg_scores = []
    for (fold, epoch), info in grouped.items():
        if len(info["values"]) == len(TESTSETS):
            avg = sum(info["values"].values()) / len(TESTSETS)
            avg_scores.append(
                (fold, epoch, avg, info["elapsed"], info["values"])
            )

    avg_scores.sort(key=lambda x: x[2], reverse=True)
    return avg_scores[:n]


def main():
    parser = argparse.ArgumentParser(
        description="Find best epochs from TensorBoard logs across k-fold training"
    )
    parser.add_argument(
        "logs_dir", help="Directory containing fold_* subdirectories"
    )
    parser.add_argument(
        "--runid", required=True, help="Run ID used in train_prod/train_tune"
    )
    args = parser.parse_args()

    all_data = load_all_folds(args.logs_dir, args.runid)

    print("\n" + "=" * 70)
    print("BEST EPOCHS BY INDIVIDUAL TESTSET")
    print("=" * 70)

    for testset in TESTSETS:
        print(f"\n{testset}:")
        print(
            f"  {'Rank':<6}{'Fold':<10}{'Epoch':<8}{'mAP':<10}{'Elapsed':<12}"
        )
        print(f"  {'-'*46}")
        top = get_top_by_testset(all_data, testset)
        for i, (fold, epoch, val, elapsed) in enumerate(top, 1):
            print(
                f"  {i:<6}{fold:<10}{epoch:<8}{val:<10.4f}{format_time(elapsed):<12}"
            )

    print("\n" + "=" * 70)
    print("BEST EPOCHS BY AVERAGE mAP (all 3 testsets)")
    print("=" * 70)
    print(
        f"\n  {'Rank':<6}{'Fold':<10}{'Epoch':<8}{'Avg mAP':<10}{'Elapsed':<12}"
    )
    print(f"  {'-'*46}")
    top_avg = get_top_by_avg(all_data)
    for i, (fold, epoch, avg, elapsed, values) in enumerate(top_avg, 1):
        print(
            f"  {i:<6}{fold:<10}{epoch:<8}{avg:<10.4f}{format_time(elapsed):<12}"
        )

    # Detail for best overall epoch
    if top_avg:
        best_fold, best_epoch, _, _, best_values = top_avg[0]
        print(f"\n  Detail for {best_fold} epoch {best_epoch}:")
        for ts in TESTSETS:
            print(f"    {ts}: {best_values[ts]:.4f}")


if __name__ == "__main__":
    main()
