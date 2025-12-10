#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare two CoverHunterMPS dataset.txt files and report statistics.

One use case is to help identify any data leakage between datasets, such as between train and test sets.

Usage: python compare_datasets.py file1.txt file2.txt

Created on Tue Dec 9 2025

@author: alanngnet and Claude Opus 4.5

"""

import json
import sys
from pathlib import Path


def parse_dataset(filepath):
    """Parse a dataset.txt file, returning sets of workIDs and perfIDs."""
    records = []
    work_ids = set()
    perf_ids = set()

    with open(filepath, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                records.append(record)
                work_ids.add(record["work"])
                perf_ids.add(record["perf"])
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Skipping line {line_num} in {filepath}: {e}")

    return records, work_ids, perf_ids


def compare_datasets(file1, file2):
    """Compare two dataset files and print statistics."""

    print(f"\n{'='*60}")
    print("CoverHunterMPS Dataset Comparison")
    print(f"{'='*60}\n")

    # Parse both files
    records1, works1, perfs1 = parse_dataset(file1)
    records2, works2, perfs2 = parse_dataset(file2)

    # File 1 statistics
    print(f"File 1: {Path(file1).name}")
    print(f"  Records:         {len(records1):,}")
    print(f"  Unique workIDs:  {len(works1):,}")
    print(f"  Unique perfIDs:  {len(perfs1):,}")

    # File 2 statistics
    print(f"\nFile 2: {Path(file2).name}")
    print(f"  Records:         {len(records2):,}")
    print(f"  Unique workIDs:  {len(works2):,}")
    print(f"  Unique perfIDs:  {len(perfs2):,}")

    # Overlap statistics
    common_works = works1 & works2
    common_perfs = perfs1 & perfs2

    print(f"\n{'─'*60}")
    print("Overlap")
    print(f"{'─'*60}")
    print(f"  workIDs in both: {len(common_works):,}")
    print(f"  perfIDs in both: {len(common_perfs):,}")

    # Additional context
    print(f"\n{'─'*60}")
    print("Exclusive to each file")
    print(f"{'─'*60}")
    print(f"  workIDs only in File 1: {len(works1 - works2):,}")
    print(f"  workIDs only in File 2: {len(works2 - works1):,}")
    print(f"  perfIDs only in File 1: {len(perfs1 - perfs2):,}")
    print(f"  perfIDs only in File 2: {len(perfs2 - perfs1):,}")

    print()


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <dataset1.txt> <dataset2.txt>")
        sys.exit(1)

    file1, file2 = sys.argv[1], sys.argv[2]

    for f in (file1, file2):
        if not Path(f).exists():
            print(f"Error: File not found: {f}")
            sys.exit(1)

    compare_datasets(file1, file2)


if __name__ == "__main__":
    main()
