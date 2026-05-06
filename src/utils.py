#!/usr/bin/env python3
# author:liufeng
# datetime:2021/11/23 4:44 PM
# software: PyCharm


import json
import logging
import os
import hashlib
import pickle
# import shutil

# import numpy as np
import yaml
# from scipy.io import wavfile

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

RARE_DELIMITER = b"\xe2\x96\x81".decode("utf-8")  # A rare used delimiter as "▁"


# ==================== data utils ====================
def read_lines(data_path, log=True):
    """read from txt file, and return all lines as list, and
    remove \t or \n in end of every line

    Args:
      data_path: txt path
      log: True to output log

    Returns:
      List

    """
    lines = []
    with open(data_path, encoding="utf-8") as fr:
        for line in fr.readlines():
            if len(line.strip().replace(" ", "")):
                lines.append(line.strip())
    if log:
        print(f"read {len(lines)} lines from {data_path}")
        print(f"example(last) {lines[-1]}\n")
    return lines


def write_lines(data_path, lines, log=True) -> None:
    """write lines to txt path.

    Args:
      data_path: txt path
      lines: List
      log: True for output logs

    Returns:

    """

    if log:
        print(f"write {len(lines)} lines to {data_path}")
        print(f"example(last line): {lines[-1]}\n")

    if len(lines) == 0:
        msg = f"Fails to write blank to {data_path}"
        raise Exception(msg)

    with open(data_path, "w", encoding="utf-8") as fw:
        for line in lines:
            fw.write(f"{line}\n")


def line_to_dict(line):
    """change line to dict.
    line: "key1:value1\t key2:value2\t ..."

    Args:
      line: str

    Returns:
      Dict

    """
    if line.startswith("{"):
        return json.loads(line)
    info = line.strip().split("\t")
    data_dict = {}
    for data in info:
        name = data.split(":")[0]
        value = ":".join(data.split(":")[1:])
        data_dict[name] = value
    if "len" in data_dict:
        data_dict["len"] = int(data_dict["len"])
    return data_dict


def dict_to_line(data_dict):
    """change dict to line, inverse of function "line_to_dict"

    Args:
      data_dict: Dict

    Returns:
      List

    """
    return json.dumps(data_dict, ensure_ascii=False)


# Not used
# def read_bin_to_numpy(data_path, data_type="float"):
#     """read matrix from bin file to numpy

#     Args:
#       data_path: bin file path
#       data_type: type, must be "int" or "float"

#     Returns:
#       numpy

#     """
#     with open(data_path) as fr:
#         frms = np.fromfile(fr, dtype=np.int32, count=1)[0]
#         dim = np.fromfile(fr, dtype=np.int32, count=1)[0]
#         if data_type == "float":
#             mat = np.fromfile(fr, dtype=np.float32)
#         elif data_type == "int":
#             mat = np.fromfile(fr, dtype=np.int32)
#         else:
#             msg = "Unvalid type"
#             raise Exception(msg)

#     assert np.size(mat) == frms * dim, f"{np.size(mat)}!={frms}x{dim}\ndata path:{data_path}"
#    return np.reshape(mat, [frms, dim])


# Not used
# def dump_numpy_to_bin(data, data_path, data_type="float") -> None:
#     """dump numpy to bin file, and data will be changed to float32

#     Args:
#       data: numpy
#       data_path: bin path
#       data_type: type must be "int" or "float"

#     Returns:

#     """
#     if data_type == "float":
#         data = data.astype(np.float32)
#     elif data_type == "int":
#         data = data.astype(np.int32)
#     else:
#         msg = "Unvalid type"
#         raise Exception(msg)

#     if np.ndim(data) == 1:
#         data = np.resize(data, [1, np.size(data)])
#     m, n = np.shape(data)
#     bin_shape = np.array([m, n], dtype=np.int32)
#     with open(data_path, "wb") as fw:
#         bin_shape.tofile(fw)
#         data.tofile(fw)


# Not used
# def load_map(map_path, key_as_int=False, value_as_int=False, split_label=" "):
#     """load map from file, file format maybe
#     a 1
#     b 2

#     Args:
#       map_path:
#       key_as_int:
#       value_as_int:

#     Returns:

#     """
#     map_dict = {}
#     for line in read_lines(map_path, log=False):
#         assert len(line.split(split_label)) >= 2, f"line:{line}"
#         if len(line.split(split_label)) == 2:
#             k, v = line.split(split_label)
#         else:
#             v = line.split(split_label)[-1]
#             k = " ".join(line.split(split_label)[:-1])

#         if key_as_int:
#             k = int(k)
#         if value_as_int:
#             v = int(v)
#         map_dict[k] = v
#     return map_dict


# ==================== file utils ====================
# Not used
# def clean_and_new_dir(data_dir) -> None:
#     """clean all files in dir

#     Args:
#       data_dir: str

#     Returns:

#     """
#     if os.path.exists(data_dir):
#         shutil.rmtree(data_dir)
#     os.makedirs(data_dir)


# Not used
# def generate_dir_tree(dir_name, sub_name_list, del_old=False):
#     """generate new folder with sub folder

#     Args:
#       dir_name: father dir
#       sub_name_list: sub dir names
#       del_old: True to del old files

#     Returns:

#     """
#     os.makedirs(dir_name, exist_ok=True)
#     dir_path_list = []
#     if del_old:
#         shutil.rmtree(dir_name, ignore_errors=True)
#     for name in sub_name_list:
#         dir_path = os.path.join(dir_name, name)
#         dir_path_list.append(dir_path)
#         os.makedirs(dir_path, exist_ok=True)
#     return dir_path_list


def get_name_from_path(abs_path):
    """get name from path. eg: /xxx/yyy/zzz/name.prefix -> name

    Args:
      abs_path: str

    Returns:
      str

    """
    return ".".join(os.path.basename(abs_path).split(".")[:-1])


# Not used
# def get_wav_path_from_dir(wav_dir):
#     """get all wav path in the input-dir"""
#     path_list = []
#     for root, _dirs, files in os.walk(wav_dir):
#         for name in files:
#             wav_path = os.path.join(root, name)
#             if (
#                 wav_path.endswith(("mp3", "wav", "mp4"))
#             ):
#                 path_list.append(wav_path)
#     return sorted(path_list)


def remake_path_for_linux(path):
    """change file path for linux"""
    return (
        path.replace(" ", r"\ ")
        .replace("(", r"\(")
        .replace(")", r"\)")
        .replace("&", r"\&")
        .replace(";", r"\;")
        .replace("'", "\\'")
    )


def create_logger():
    return logging.getLogger()


# ==================== hparams utils ====================
def load_hparams(yaml_path):
    """load hparams from yaml path as dict"""
    with open(yaml_path, encoding="utf-8") as yaml_file:
        return yaml.safe_load(yaml_file)


# Not used
# def dump_hparams(yaml_path, hparams) -> None:
#     """dump dict to yaml path

#     Args:
#       yaml_path: str
#       hparams: dict

#     Returns:

#     """
#     with open(yaml_path, "w") as fw:
#         yaml.dump(hparams, fw)
#     print(f"save hparams to {yaml_path}")


def get_hparams_as_string(hparams):
    """get hparams as format string

    Args:
      hparams: dict

    Returns:
      string

    """
    return json.dumps(hparams, indent=2, separators=(",", ": "))


# ==================== wave utils ====================
# Not used
# def save_wav(wav, path, sr, k=None) -> None:
#     norm_wav = wav * 32767 / max(0.01, np.max(np.abs(wav))) * k if k else wav * 32767
#     wavfile.write(path, sr, norm_wav.astype(np.int16))


def compute_model_fingerprint(state_dict) -> str:
    """Deterministic 16-hex-char fingerprint of a model's weights.

    Hashes tensor data in sorted-key order so the same trained weights
    always yield the same fingerprint, regardless of save timestamp,
    PyTorch version, or filesystem metadata. Used to verify that
    reference embeddings, centroids, and the inference-time model are
    mutually consistent.

    Args:
        state_dict: a PyTorch state_dict (OrderedDict of name -> tensor),
            typically obtained from model.state_dict() or torch.load(...).

    Returns:
        16-character lowercase hex string.
    """
    h = hashlib.sha256()
    for key in sorted(state_dict.keys()):
        tensor = state_dict[key]
        # Include the parameter name so renaming a layer changes the hash.
        h.update(key.encode("utf-8"))
        # Include shape and dtype so a same-name parameter with different
        # geometry doesn't collide.
        h.update(str(tuple(tensor.shape)).encode("utf-8"))
        h.update(str(tensor.dtype).encode("utf-8"))
        # Move to CPU and contiguous before extracting bytes; detach in
        # case any param has grad attached.
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()[:16]


def load_reference_embeddings(path: str) -> dict:
    """Load a reference_embeddings.pkl file with format validation.

    Returns the record as written by tools.make_embeds, with these keys:
        embeddings       dict perf_id -> unit-normalized np.ndarray
        norms            dict perf_id -> float (pre-normalization norm)
        normalized       bool (always True for v2 format)
        embed_dim        int
        model_fingerprint  str
        calibration      dict per src.calibration schema, or None if --raw

    Raises a clear error if the file is in legacy flat-dict format,
    directing the user to regenerate.
    """
    with open(path, "rb") as f:
        data = pickle.load(f)

    if not isinstance(data, dict) or "embeddings" not in data:
        raise ValueError(
            f"{path} appears to be in the legacy flat-dict format "
            f"(perf_id -> embedding). Regenerate with the current "
            f"tools.make_embeds; the storage schema has changed to "
            f"include normalization, calibration, and a model "
            f"fingerprint."
        )

    required = {
        "embeddings",
        "norms",
        "normalized",
        "embed_dim",
        "model_fingerprint",
        "calibration",
    }
    missing = required - set(data.keys())
    if missing:
        raise ValueError(
            f"{path} is missing required keys: {sorted(missing)}. "
            f"Regenerate with the current tools.make_embeds."
        )

    return data


def load_centroids(path: str) -> dict:
    """Load a work_centroids.pkl file with format validation.

    Returns the record as written by tools.compute_centroids, with these
    keys:
        centroids          dict work_id -> unit-normalized np.ndarray
        radii              dict work_id -> float (angular spread)
        coherence          dict work_id -> float (pre-renorm magnitude
                           of the work-mean; rough within-work agreement)
        work_to_perfs      dict work_id -> list of perf_ids
        normalized         bool (always True)
        embed_dim          int
        model_fingerprint  str
        calibration        dict per src.calibration schema, or None

    Raises a clear error if the file is in legacy format, directing the
    user to regenerate.
    """
    with open(path, "rb") as f:
        data = pickle.load(f)

    # Legacy format detection: has centroids+radii+work_to_perfs but
    # lacks any of the v2-only keys.
    if not isinstance(data, dict) or "centroids" not in data:
        raise ValueError(
            f"{path} is not a recognizable centroids file. "
            f"Regenerate with the current tools.compute_centroids."
        )

    required = {
        "centroids",
        "radii",
        "coherence",
        "work_to_perfs",
        "normalized",
        "embed_dim",
        "model_fingerprint",
        "calibration",
    }
    missing = required - set(data.keys())
    if missing:
        raise ValueError(
            f"{path} is missing required keys: {sorted(missing)}. "
            f"This may be a legacy v1 centroids file. Regenerate with "
            f"the current tools.compute_centroids."
        )

    return data


if __name__ == "__main__":
    pass
