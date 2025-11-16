#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare two dataset.txt files and generate query_in_ref describing which perf
ids in one are also present in the other.
Generate query_in_ref file as used by eval_testset.py

Usage from repo root:
    Set query and ref locations at top of script
    python tools/make_query_in_ref.py

Created on Thu Apr 11 21:18:06 2024

@author: alan
"""
reference_file = "data/reels50hard_testset/full.txt"
query_file = "../CHmodels/reels80.6-6/Reel80_6-6.txt"

import json
import argparse


def process_files(query_path, ref_path):
    """
    Processes reference and query metadata to generate "query_in_ref" data,
    which is an index of perfs (performances) that appear in both data sets.
    Typical usage during training is to provide the training dataset as
    the query file and a test set as the reference file. If you use multiple
    test sets, you would want to generate a separate query_in_ref file for each
    test set (assuming there could be overlap of the same performances in both
    training and test data.)
    Note that this is irrelevant when using the same data for both training
    and test, such as in the default covers80 demo setup where covers80 is used
    as the only test set while training on covers80.

    Args:
        ref_file (str): Path to the reference JSON file.
        query_file (str): Path to the query JSON file.

    Returns:
        list: A list of tuples representing the query-reference mappings.
    """

    with open(ref_path, "r") as f:
        ref_data = [json.loads(line) for line in f]

    with open(query_path, "r") as f:
        query_data = [json.loads(line) for line in f]

    # Build a dictionary for lookup of ref works by their 'perf' value
    ref_perfs = {item["perf"]: item for item in ref_data}

    query_in_ref = []
    for q_idx, query_item in enumerate(query_data):
        if query_item["perf"] in ref_perfs:
            ref_idx = ref_data.index(ref_perfs[query_item["work"]])
            query_in_ref.append((q_idx, ref_idx))

    return query_in_ref


parser = argparse.ArgumentParser(
    description="generate query_in_ref mapping of perfs appearing in both datasets"
)
parser.add_argument("query_path")
parser.add_argument("ref_path")
args = parser.parse_args()
query_path = args.query_path
ref_path = args.ref_path
query_in_ref = process_files(query_path, ref_path)

# Output to "query_in_ref.txt"
with open("query_in_ref.txt", "w") as f:
    json.dump({"query_in_ref": query_in_ref}, f)
