#!/usr/bin/env python3
# author:liufeng
# datetime:2022/7/11 3:20 PM
# software: PyCharm

import argparse
import logging
import os
import gc
import random
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

import librosa
from nnAudio.features.cqt import CQT, CQT2010v2
import numpy as np
import torch
import torchaudio

from src.cqt import PyCqt
from src.dataset import SignalAug
from src.utils import (
    RARE_DELIMITER,
    dict_to_line,
    line_to_dict,
    load_hparams,
    read_lines,
    remake_path_for_linux,
    write_lines,
)


def _sort_lines_by_perf(init_path, sorted_path) -> None:
    dump_lines = read_lines(init_path, log=False)
    dump_lines = sorted(dump_lines, key=lambda x: (line_to_dict(x)["perf"]))
    write_lines(sorted_path, dump_lines, log=True)


def _remove_dup_line(init_path, new_path) -> None:
    """Skip lines having the same perf"""
    old_line_num = len(read_lines(init_path, log=False))
    perf_set = set()
    valid_lines = []
    for line in read_lines(init_path, log=False):
        perf = line_to_dict(line)["perf"]
        if perf not in perf_set:
            perf_set.add(perf)
            valid_lines.append(line)
    logging.info(f"Filter stage: {old_line_num}->{len(valid_lines)}")
    write_lines(new_path, valid_lines)


# Unused
# def _remove_invalid_line(init_path, new_path) -> None:
#     old_line_num = len(read_lines(init_path, log=False))
#     dump_lines = []
#     for line in read_lines(init_path, log=False):
#         local_data = line_to_dict(line)
#         if not os.path.exists(local_data["wav"]):
#             logging.info(f"Unvalid data for wav path: {line}")
#             continue
#         dump_lines.append(line)
#     logging.info(f"Filter stage: {old_line_num}->{len(dump_lines)}")
#     write_lines(new_path, dump_lines)


# Unused
# def _remove_line_with_same_dur(init_path, new_path) -> None:
#     """remove line with same work-id and same dur-ms"""
#     old_line_num = len(read_lines(init_path, log=False))
#     dump_lines = []
#     for line in read_lines(init_path, log=False):
#         local_data = line_to_dict(line)
#         if not os.path.exists(local_data["wav"]):
#             logging.info(f"Unvalid data for wav path: {line}")
#             continue
#         dump_lines.append(line)
#     logging.info(f"Filter stage: {old_line_num}->{len(dump_lines)}")
#     write_lines(new_path, dump_lines)


def sox_change_speed(inp_path, out_path, k):
    cmd = (
        "sox -q {} -t wav  -r 16000 -c 1 {} tempo {} "
        "> sox.log 2> sox.log".format(
            remake_path_for_linux(inp_path), remake_path_for_linux(out_path), k
        )
    )

    try:
        subprocess.call(cmd, shell=True)
        success = os.path.exists(out_path)
        if not success:
            logging.info(f"Error for sox: {cmd}")
        return success
    except RuntimeError:
        logging.info(f"RuntimeError: {cmd}")
        return False
    except EOFError:
        logging.info(f"EOFError: {cmd}")
        return False


# instead of original serial function
# leverage multiple CPU cores to run multiple sox instances in parallel
def _speed_aug_worker(args):
    """worker function for _speed_aug_parallel"""
    line, speed, sp_dir = args
    wav_path = line["wav"]

    if abs(speed - 1.0) > 0.01:
        sp_perf = "sp_{}-{}".format(speed, line["perf"])
        sp_wav_path = os.path.join(sp_dir, f"{sp_perf}.wav")
        if not os.path.exists(sp_wav_path):
            sox_change_speed(wav_path, sp_wav_path, speed)
    else:
        sp_perf = line["perf"]
        sp_wav_path = line["wav"]

    # added logic missing in original CoverHunter: modify dur_s
    # so that _cut_one_line_with_dur function slices augmented samples appropriately
    # since speed augmentation also changes duration
    line["dur_s"] = round(line["dur_s"] / speed, 2)
    line["perf"] = sp_perf
    line["wav"] = sp_wav_path
    return line


def _speed_aug_parallel(init_path, aug_speed_lst, aug_path, sp_dir) -> None:
    """add items with speed argument wav"""
    logging.info(f"speed factors: {aug_speed_lst}")
    os.makedirs(sp_dir, exist_ok=True)
    total_lines = read_lines(init_path, log=False)
    dump_lines = []

    with ProcessPoolExecutor() as executor:
        worker_args = [
            (line_to_dict(line), speed, sp_dir)
            for line in total_lines
            for speed in aug_speed_lst
        ]

        for result in executor.map(_speed_aug_worker, worker_args):
            if result == "skip":
                continue
            dump_lines.append(dict_to_line(result))
            if len(dump_lines) % 1000 == 0:
                logging.info(f"{len(dump_lines)}: {dump_lines[-1]}")

    write_lines(aug_path, dump_lines)


# instead of original serial function,
# leverage multiple CPU cores to run multiple CQT extractions in parallel
def _extract_cqt_worker_librosa(args):
    """worker function for _extract_cqt_parallel"""
    line, cqt_dir, fmin, max_freq, bins_per_octave = args
    wav_path = line["wav"]
    py_cqt = PyCqt(
        sample_rate=16000,
        hop_size=0.04,
        octave_resolution=bins_per_octave,
        min_freq=fmin,
        max_freq=max_freq,
    )
    feat_path = os.path.join(cqt_dir, "{}.cqt.npy".format(line["perf"]))

    if not os.path.exists(feat_path):
        y, sr = librosa.load(wav_path, sr=16000)  # y is a npy ndarray
        y = y / max(0.001, np.max(np.abs(y))) * 0.999
        cqt = py_cqt.compute_cqt(signal_float=y, feat_dim_first=False)
        np.save(feat_path, cqt)
        feat_len = len(cqt)
    else:
        feat_len = len(np.load(feat_path))
    line["feat"] = feat_path
    line["feat_len"] = feat_len
    return line


def _extract_cqt_worker_torchaudio(args):
    line, cqt_dir, fmin, max_freq, n_bins, bins_per_octave, device = args
    wav_path = line["wav"]
    feat_path = os.path.join(cqt_dir, "{}.cqt.npy".format(line["perf"]))

    # CQT seems faster on mps, and CQT2010v2 faster on cuda
    if device == "mps":
        transform = CQT
    elif device == "cuda":
        transform = CQT2010v2

    signal, sr = torchaudio.load(wav_path)
    # Force mono by averaging channels if stereo
    if signal.shape[0] > 1:
        signal = torch.mean(signal, dim=0, keepdim=True)
    # Resample if needed, for example, in case input is .mp3
    if sr != 16000:
        signal = torchaudio.functional.resample(signal, sr, 16000)
    signal = signal.to(device)
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

    # does this help prevent MPS hanging between batches?
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()
    np.save(feat_path, cqt)

    feat_len = len(cqt)
    line["feat"] = feat_path
    line["feat_len"] = feat_len
    return line


def worker(args):
    line, cqt_dir, fmin, max_freq, n_bins, bins_per_octave, device = args
    try:
        if device in ("mps", "cuda"):
            return _extract_cqt_worker_torchaudio(args)

        return _extract_cqt_worker_librosa(
            line, cqt_dir, fmin, max_freq, bins_per_octave
        )
    except Exception as e:
        return f"Error processing line: {line}\nError: {str(e)}"


def _extract_cqt_parallel(
    init_path, out_path, cqt_dir, fmin, n_bins, bins_per_octave, device
) -> None:
    os.makedirs(cqt_dir, exist_ok=True)
    dump_lines = []
    # calculate max_freq in case CPU device requires use of the PyCQT function
    max_freq = fmin * (2 ** (n_bins / bins_per_octave))

    # Pre-process to separate existing from missing CQT files
    lines = read_lines(init_path, log=False)
    missing_cqt_lines = []

    for line in lines:
        local_data = line_to_dict(line)
        feat_path = os.path.join(
            cqt_dir, "{}.cqt.npy".format(local_data["perf"])
        )

        if os.path.exists(feat_path):
            # Get file size instead of
            # loading existing CQT files to estimate feat_len
            # Note that feat_len is only used when training with mode="random"
            # and even then it ends up extracting the real length anyway
            # AudioFeatDataset.__getitem__()

            file_size = os.path.getsize(feat_path)
            # Rough estimate: each frame is n_bins * 8 bytes (float64)
            estimated_feat_len = file_size // (n_bins * 8)
            local_data["feat"] = feat_path
            local_data["feat_len"] = estimated_feat_len
            dump_lines.append(dict_to_line(local_data))
        else:
            missing_cqt_lines.append(line)

    logging.info(
        f"Found {len(dump_lines)} existing CQT files, need to process {len(missing_cqt_lines)} files"
    )

    if not missing_cqt_lines:
        write_lines(out_path, dump_lines)
        return

    # Process data in smaller batches as workaround to memory leak
    # encountered in large runs, probably in nnAudio library
    batch_size = 2000
    lines = read_lines(init_path, log=False)
    mp.set_start_method("spawn", force=True)

    for i in range(0, len(missing_cqt_lines), batch_size):
        batch = missing_cqt_lines[i : i + batch_size]
        with ProcessPoolExecutor() as executor:
            worker_args = [
                (
                    line_to_dict(line),
                    cqt_dir,
                    fmin,
                    max_freq,
                    n_bins,
                    bins_per_octave,
                    device,
                )
                for line in batch
            ]

            for result in executor.map(worker, worker_args):
                if isinstance(result, str) and result.startswith(
                    "Error processing line:"
                ):
                    print(result)  # This will print only the error lines
                    continue
                dump_lines.append(dict_to_line(result))
                if len(dump_lines) % 1000 == 0:
                    logging.info(
                        "Extracted CQT for {} items: {}".format(
                            len(dump_lines),
                            result["perf"],
                        ),
                    )
        # clean up context to prevent hanging between batches
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
            torch.mps.synchronize()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
        # Small delay to let system recover
        import time

        time.sleep(0.1)
    write_lines(out_path, dump_lines)


def _extract_cqt_with_noise(init_path, full_path, cqt_dir, hp_noise) -> None:
    logging.info("Extract CQT features with noise argumentation")
    os.makedirs(cqt_dir, exist_ok=True)

    py_cqt = PyCqt(sample_rate=16000, hop_size=0.04)
    sig_aug = SignalAug(hp_noise)
    vol_lst = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    dump_lines = []
    for line in read_lines(init_path, log=False):
        local_data = line_to_dict(line)
        wav_path = local_data["wav"]
        local_data["perf"] = local_data["perf"] + "{}noise_{}".format(
            RARE_DELIMITER,
            hp_noise["name"],
        )
        local_data["feat"] = os.path.join(
            cqt_dir,
            "{}.cqt.npy".format(local_data["perf"]),
        )

        vol = random.choice(vol_lst)
        if not os.path.exists(local_data["feat"]):
            y, sr = librosa.load(wav_path, sr=16000)
            y = sig_aug.augmentation(y)
            y = y / max(0.001, np.max(np.abs(y))) * 0.999 * vol
            cqt = py_cqt.compute_cqt(signal_float=y, feat_dim_first=False)
            np.save(local_data["feat"], cqt)
            local_data["feat_len"] = len(cqt)

        if "feat_len" not in local_data:
            cqt = np.load(local_data["feat"])
            local_data["feat_len"] = len(cqt)

        dump_lines.append(dict_to_line(local_data))

        if len(dump_lines) % 1000 == 0:
            logging.info(
                "Process cqt for {}items: {}, vol:{}".format(
                    len(dump_lines),
                    local_data["perf"],
                    vol,
                ),
            )

    write_lines(full_path, dump_lines)


def _add_work_id(init_path, out_path, map_path=None) -> None:
    """map format:: work_id -> work"""
    work_id_map = {}
    dump_lines = []
    for line in read_lines(init_path, log=False):
        local_data = line_to_dict(line)
        work_name = local_data["work"]
        if work_name not in work_id_map:
            work_id_map[work_name] = len(work_id_map)
        local_data["work_id"] = work_id_map[work_name]
        dump_lines.append(dict_to_line(local_data))
    write_lines(out_path, dump_lines)

    if map_path:
        dump_lines = []
        for k, v in work_id_map.items():
            dump_lines.append(f"{v} {k}")
        write_lines(map_path, dump_lines)


def _split_data_by_work_id(
    input_path,
    train_path,
    val_path,
    test_path,
    test_work_ids_path,
    hp,
) -> None:
    """
    Splits data into train / validation / test sets
        using stratified sampling based on work_id.

    Args:
        input_path: Path to the input data file.
        train_path: Path to write the training data.
        val_path: Path to write the validation data.
        test_path: Path to write the testing data.
        test_work_ids_path: Path to write a list of work_ids reserved for test
    """
    # percent of unique work IDs to include only in the val or test sets
    val_only_percent = hp["val_unseen"]
    test_only_percent = hp["test_data_unseen"]
    val_ratio = hp["val_data_split"]
    test_ratio = hp["test_data_split"]

    # Dictionary to store work ID counts and shuffled sample lists
    work_data = {}
    for line in read_lines(input_path):
        local_data = line_to_dict(line)
        work_id = local_data["work_id"]
        if work_id not in work_data:
            work_data[work_id] = []
        work_data[work_id].append(local_data)
    num_works = len(work_data)
    logging.info("Number of distinct works: %s", num_works)

    # Separate works for test-only and stratified split.
    # Ensure minimum of one work in test only if non-zero requested.
    test_only_count = (
        max(1, int(num_works * test_only_percent))
        if test_only_percent > 0 and test_ratio > 0
        else 0
    )
    # Randomly select works for test only
    test_only_works = random.sample(list(work_data.keys()), test_only_count)
    remaining_works = {
        work_id: samples
        for work_id, samples in work_data.items()
        if work_id not in test_only_works
    }

    # Process works for test only (all samples to test)
    test_data = []
    for work_id in test_only_works:
        test_data.extend(work_data[work_id])

    # Separate works for val-only and stratified split.
    # Ensure minimum of one work in val only if non-zero requested
    val_only_count = (
        max(1, int(num_works * val_only_percent))
        if val_only_percent > 0
        else 0
    )
    val_only_works = random.sample(
        list(remaining_works.keys()),
        val_only_count,
    )  # Randomly select works for val only
    remaining_works = {
        work_id: samples
        for work_id, samples in remaining_works.items()
        if work_id not in val_only_works
    }

    # Process works for val only (all samples to val)
    val_data = []
    for work_id in val_only_works:
        val_data.extend(work_data[work_id])

    train_data, remaining_val_data, remaining_test_data = [], [], []
    if (
        val_ratio > 0 or test_ratio > 0
    ):  # don't bother if 0,0 like for testset CQT generation
        # Stratified split for remaining works
        for work_id, samples in remaining_works.items():
            # Randomly shuffle samples for this work ID
            random.shuffle(samples)

            # Calculate val split points based on train ratio and minimum samples
            min_samples = (
                1  # Ensure at least 1 sample in each set for remaining works
            )
            val_split = int(len(samples) * val_ratio)
            val_split = max(
                min_samples,
                val_split,
            )  # Ensure at least min_samples in val

            # Calculate test split points based on train ratio and minimum samples
            min_samples = (
                (1)  # Ensure at least 1 sample in each set for remaining works
                if test_ratio > 0
                else 0
            )
            test_split = int(len(samples) * test_ratio)
            test_split = max(
                min_samples,
                test_split,
            )  # Ensure at least min_samples in test

            # if exceeding test_ratio then put more in val than test
            if (
                (len(test_data) + len(remaining_test_data)) / len(train_data)
                if len(train_data) != 0
                else 0 > test_ratio
            ):
                test_split = max(min_samples, test_split - 1)

            remaining_val_data.extend(samples[:val_split])
            if test_ratio > 0:
                remaining_test_data.extend(
                    samples[val_split : val_split + test_split]
                )
            train_data.extend(samples[val_split + test_split :])

        val_data.extend(remaining_val_data)
        test_data.extend(remaining_test_data)
    else:
        for samples in remaining_works.values():
            train_data.extend(samples)
    #        train_data.extend(remaining_works.items())

    logging.info("Number of samples in train: %s", len(train_data))
    logging.info("Number of samples in validate: %s", len(val_data))
    logging.info("Number of samples in test: %s", len(test_data))

    write_lines(train_path, [dict_to_line(sample) for sample in train_data])
    if len(val_data) > 0:
        write_lines(val_path, [dict_to_line(sample) for sample in val_data])
    if len(test_data) > 0:
        write_lines(test_path, [dict_to_line(sample) for sample in test_data])
    if len(test_only_works) > 0:
        write_lines(
            test_work_ids_path,
            [dict_to_line(work) for work in test_only_works],
        )


# =============================================================================
# Not needed
#
# def _add_version_id(init_path, out_path):
#   work_version_map = {}
#   for line in read_lines(init_path, log=False):
#     local_data = line_to_dict(line)
#     work_id = local_data["work_id"]
#     if work_id not in work_version_map.keys():
#       work_version_map[work_id] = []
#     work_version_map[work_id].append(local_data)
#
#   dump_lines = []
#   for k, v_lst in work_version_map.items():
#     for version_id, local_data in enumerate(v_lst):
#       local_data["version_id"] = version_id
#       dump_lines.append(dict_to_line(local_data))
#   write_lines(out_path, dump_lines)
#   return
#
#
# def _extract_work_num(full_path, work_name_map_path, work_id_map_path):
#   """add map of work_id:num and work_name:num"""
#   work_id_num = {}
#   max_work_id = 0
#   for line in read_lines(full_path):
#     local_data = line_to_dict(line)
#     work_id = local_data["work_id"]
#     if work_id not in work_id_num.keys():
#       work_id_num[work_id] = 0
#     work_id_num[work_id] += 1
#     if work_id >= max_work_id:
#       max_work_id = work_id
#   logging.info("max_work_id: {}".format(max_work_id))
#
#   dump_data = list(work_id_num.items())
#   dump_data = sorted(dump_data)
#   dump_lines = ["{} {}".format(k, v) for k, v in dump_data]
#   write_lines(work_id_map_path, dump_lines, log=False)
#
#   work_num = {}
#   for line in read_lines(full_path, log=False):
#     local_data = line_to_dict(line)
#     work_id = local_data["work"]
#     if work_id not in work_num.keys():
#       work_num[work_id] = 0
#     work_num[work_id] += 1
#
#   dump_data = list(work_num.items())
#   dump_data = sorted(dump_data)
#   dump_lines = ["{} {}".format(k, v) for k, v in dump_data]
#   write_lines(work_name_map_path, dump_lines, log=False)
#   return
#
#
# def _sort_lines_by_work_id(full_path, sorted_path):
#   dump_lines = read_lines(full_path, log=False)
#   dump_lines = sorted(dump_lines,
#                       key=lambda x: (int(line_to_dict(x)["work_id"]),
#                                      int(line_to_dict(x)["version_id"])))
#   write_lines(sorted_path, dump_lines, log=True)
#   return
#
# =============================================================================


def _clean_lines(full_path, clean_path) -> None:
    dump_lines = []
    for line in read_lines(full_path):
        local_data = line_to_dict(line)
        clean_data = {
            "perf": local_data["perf"],
            "work_id": local_data["work_id"],
            "work": local_data["work"],
            "version_id": local_data["version_id"],
        }
        if "feat" in local_data:
            clean_data.update(
                {
                    "feat_len": local_data["feat_len"],
                    "feat": local_data["feat"],
                },
            )
        else:
            clean_data.update(
                {"dur_ms": local_data["dur_ms"], "wav": local_data["wav"]},
            )
        dump_lines.append(dict_to_line(clean_data))
    write_lines(clean_path, dump_lines)


def _generate_csi_features(hp, feat_dir, start_stage, end_stage) -> None:
    data_path = os.path.join(feat_dir, "dataset.txt")
    assert os.path.exists(data_path), "dataset.txt file not found"

    init_path = os.path.join(feat_dir, "data.init.txt")
    shutil.copy(data_path, init_path)
    if start_stage <= 0 <= end_stage:
        logging.info("Stage 0: data deduping")
        _sort_lines_by_perf(init_path, init_path)
        _remove_dup_line(init_path, init_path)

    # aug_speed_mode is a list like: [0.8, 0.9, 1.0, 1.1, 1.2]
    # do include 1.0 to include original speed.
    # Anything between .99 and 1.01 will be ignored,
    # instead passing along the original file.
    sp_aug_path = os.path.join(feat_dir, "sp_aug.txt")
    if start_stage <= 3 <= end_stage:
        logging.info("Stage 3: speed augmentation")
        if "aug_speed_mode" in hp and not os.path.exists(sp_aug_path):
            sp_dir = os.path.join(feat_dir, "sp_wav")
            #      _speed_aug(init_path, hp["aug_speed_mode"], sp_aug_path, sp_dir)
            _speed_aug_parallel(
                init_path, hp["aug_speed_mode"], sp_aug_path, sp_dir
            )

    new_full = False
    full_path = os.path.join(feat_dir, "full.txt")
    if start_stage <= 4 <= end_stage:
        logging.info("Stage 4: extract CQT features")
        if not os.path.exists(full_path):
            new_full = True
            cqt_dir = os.path.join(feat_dir, "cqt_feat")
            if "fmin" not in hp:
                fmin = 32
            else:
                fmin = hp["fmin"]
            if "n_bins" not in hp:
                n_bins = 96
            else:
                n_bins = hp["n_bins"]
            if "bins_per_octave" not in hp:
                bins_per_octave = 12
            else:
                bins_per_octave = hp["bins_per_octave"]
            _extract_cqt_parallel(
                sp_aug_path,
                full_path,
                cqt_dir,
                fmin,
                n_bins,
                bins_per_octave,
                hp["device"],
            )

    # noise augmentation was default off for CoverHunter
    hp_noise = hp.get("add_noise", None)
    if (
        start_stage <= 5 <= end_stage
        and hp_noise
        and os.path.exists(hp_noise["noise_path"])
    ):
        logging.info("Stage 5: add noise and extract CQT features")
        noise_cqt_dir = os.path.join(feat_dir, "cqt_with_noise")
        _extract_cqt_with_noise(
            full_path,
            full_path,
            noise_cqt_dir,
            hp_noise={"add_noise": hp_noise},
        )

    # Assumes "work" values provided in dataset.txt are unique identifiers
    # for the parent works.
    # work_id.map is a useful reference document, but not used in this project
    if start_stage <= 8 <= end_stage:
        logging.info("Stage 8: add work_id")
        work_id_map_path = os.path.join(feat_dir, "work_id.map")
        if new_full or not os.path.exists(full_path):
            _add_work_id(full_path, full_path, work_id_map_path)

    # =============================================================================
    # CoverHunter doesn't actually do anything with version_id or the .map files
    # if start_stage <= 9 <= end_stage:
    #  logging.info("Stage 9: add version_id")
    #   _add_version_id(full_path, full_path)
    #
    # if start_stage <= 10 <= end_stage:
    #   logging.info("Start stage 10: extract version num")
    #   work_id_map_path = os.path.join(feat_dir, "work_id_num.map")
    #   work_num_map_path = os.path.join(feat_dir, "work_name_num.map")
    #   _extract_work_num(full_path, work_num_map_path, work_id_map_path)
    #
    # if start_stage <= 11 <= end_stage:
    #   logging.info("Stage 11: clean for unused keys")
    #   _sort_lines_by_work_id(full_path, full_path)
    # =============================================================================

    if start_stage <= 13 <= end_stage:
        logging.info("Stage 13: split data into train / validate / test sets")
        train_path = os.path.join(feat_dir, "train.txt")
        val_path = os.path.join(feat_dir, "val.txt")
        test_path = os.path.join(feat_dir, "test.txt")
        test_work_ids_path = os.path.join(feat_dir, "test-only-work-ids.txt")
        _split_data_by_work_id(
            full_path,
            train_path,
            val_path,
            test_path,
            test_work_ids_path,
            hp,
        )


def _cmd() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("feat_dir", help="feat_dir")
    parser.add_argument("--start_stage", type=int, default=0)
    parser.add_argument("--end_stage", type=int, default=100)
    args = parser.parse_args()
    hp_path = os.path.join(args.feat_dir, "hparams.yaml")
    hp = load_hparams(hp_path)
    print(hp)
    _generate_csi_features(hp, args.feat_dir, args.start_stage, args.end_stage)


if __name__ == "__main__":
    _cmd()
