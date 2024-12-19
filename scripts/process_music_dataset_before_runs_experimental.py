"""
This script intends to preprocess music dataset:


- calculates statistits for features
- scales features
- imputes features
- imputes targets (they are already imputed by zeros)


"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import multiprocessing as mp
import datetime


LOCK = mp.Lock()


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=Path, default=Path("/mnt/ar_hdd/fvelikon/graph-time-series/datasets/music"), help="Root for raw loaded data")
    parser.add_argument("--output_root", type=Path, required=True, help="Root for processed data")
    parser.add_argument("--processes", type=int, required=True, default=12, help="Number of processes")


    return parser

# def read_offset_memmap_file()

def calculate_features_statistics_slice(memmap_path, memmap_shape, slice_start_idx, slice_end_idx, first_val_index):

    slice_shape = (slice_end_idx - slice_start_idx, memmap_shape[1], memmap_shape[-1])
    offset = memmap_shape[2] * memmap_shape[1] * (slice_end_idx - slice_start_idx) * 4

    memmap_slice = np.memmap(memmap_path, mode="r", dtype="float32", shape=slice_shape, offset=offset).copy()

    empty_features_mask = np.isclose(memmap_slice, 0).sum(-1) == memmap_shape[-1]
    # empty_features_mask = np.isclose(memmap_slice, 0).sum(-1) == memmap_shape[-1]


    if slice_start_idx >= first_val_index:
        nonempty_features_amounts_train = 0
        first_momentum_features = second_momentum_features = maxes_features = mins_features = np.zeros(memmap_shape[-1])
    else:
        slice_end_idx = min(slice_end_idx, first_val_index)
        train_features_slice = memmap_slice[slice_start_idx:slice_end_idx, ...]
        first_momentum_features = train_features_slice.sum(0)
        second_momentum_features = (train_features_slice ** 2).sum(0)
        nonempty_features_amounts_train = empty_features_mask[:slice_end_idx-slice_start_idx].sum()
        try:
            maxes_features = train_features_slice[~empty_features_mask].max(0)
            mins_features = train_features_slice[~empty_features_mask].min(0)
        except ValueError:
            maxes_features = -np.inf * np.ones_like(first_momentum_features)
            mins_features = np.inf * np.ones_like(first_momentum_features)

    print(f"Calculated statistics for slice [{slice_start_idx}, {slice_end_idx}) {empty_features_mask.sum()=} {(~empty_features_mask).sum()=} {empty_features_mask[:100]} {memmap_slice[slice_start_idx:slice_end_idx, ...][:100]}")
    return (
        empty_features_mask, nonempty_features_amounts_train, first_momentum_features, second_momentum_features, maxes_features, mins_features
    )


def process_features_based_on_statistics(minmax_scaled_memmap_path, standard_scaled_memmap_path, original_features_memmap_path, empty_features_slice_mask,
                                         slice_start_idx, slice_end_idx, means_features_wise, stds_fetures_wise, maxes_features_wise, mins_features_wise,
                                         shape
                                         ):
    global LOCK

    minmax_memmap = np.memmap(minmax_scaled_memmap_path, dtype='float32', mode='r+', shape=shape)
    std_scaler_memmap = np.memmap(standard_scaled_memmap_path, dtype='float32', mode='r+', shape=shape)
    original_memmap = np.memmap(original_features_memmap_path, mode="r", dtype="float32", shape=shape)

    raw_features_slice_nonempty = original_memmap[slice_start_idx:slice_end_idx]

    standards_scaled = (raw_features_slice_nonempty - means_features_wise) / stds_fetures_wise
    minmax_scaled = (raw_features_slice_nonempty - mins_features_wise) / maxes_features_wise

    standards_scaled[empty_features_slice_mask] = 0
    minmax_scaled[empty_features_slice_mask] = 0

    std_scaler_memmap[slice_start_idx:slice_end_idx, ...] = standards_scaled
    minmax_memmap[slice_start_idx:slice_end_idx, ...] = minmax_scaled
    print(f"Computed scaled slices [{slice_start_idx}, {slice_end_idx}) - MinMax scaled and Standard-scaled")
    with LOCK:
        std_scaler_memmap.flush()
        minmax_memmap.flush()
    print(f"Processed features slice [{slice_start_idx}, {slice_end_idx}) - MinMax scaled and Standard-scaled")


def main():
    args = get_parser().parse_args()

    initial_spatiotemporal_features_path = str(args.dataset_root / "spatiotemporal_features.memmap")
    initial_metainfo_path = args.dataset_root / "music_dataset_except_spatiotemporal_features.npz"

    initial_metainfo = np.load(initial_metainfo_path, allow_pickle=True)
    train_timestamps = initial_metainfo["train_timestamps"]
    val_timestamps = initial_metainfo["val_timestamps"]
    test_timestamps = initial_metainfo["test_timestamps"]


    number_of_nodes = initial_metainfo["num_nodes"].item()
    number_of_timestamps = initial_metainfo["num_timestamps"].item()
    spt_features_number = 207

    memmap_shape = (number_of_timestamps, number_of_nodes, spt_features_number)

    _segment_size = number_of_timestamps // args.processes
    _arguments = []
    _first_val_index = len(train_timestamps)
    for i in range(args.processes):
        start_index = i * _segment_size
        end_index = (i + 1) * _segment_size if i != args.processes - 1 else number_of_timestamps

        _arguments.append([initial_spatiotemporal_features_path, memmap_shape, start_index, end_index, _first_val_index])


    with mp.Pool(processes=args.processes) as pool:
        results_job_wise = pool.starmap(calculate_features_statistics_slice, _arguments)

    (empty_features_mask,
     nonempty_features_amounts_train,
     first_momentum_features,
     second_momentum_features,
     maxes_features,
     mins_features) = zip(*results_job_wise)


    empty_features_mask = np.concatenate(empty_features_mask, 0)
    nonempty_features_amounts_train = sum(nonempty_features_amounts_train)
    first_momentum_features = np.stack(first_momentum_features).sum(0)
    second_momentum_features = np.stack(second_momentum_features).sum(0)
    maxes_features = np.stack(maxes_features).max(0)
    mins_features = np.stack(mins_features).min(0)

    _features_first_momentum = first_momentum_features / nonempty_features_amounts_train
    _features_second_momentum = second_momentum_features / nonempty_features_amounts_train

    means_features_wise = _features_first_momentum
    stds_fetures_wise = _features_second_momentum - (_features_first_momentum) ** 2

    output_root_path: Path = args.output_root
    output_root_path.mkdir(parents=True, exist_ok=True)

    std_scaler_memmap_path = str(output_root_path / "standard_scaled.memmap")
    # std_scaler_memmap = np.memmap(std_scaler_memmap_path, dtype='float32', mode='w+', shape=memmap_shape)
    # std_scaler_memmap[:, :, :] = 0
    # std_scaler_memmap.flush()

    min_max_scaler_memmap_path = str(output_root_path / "min_max_scaled.memmap")
    # min_max_scaler_memmap = np.memmap(min_max_scaler_memmap_path, dtype='float32', mode='w+', shape=memmap_shape)
    # min_max_scaler_memmap[:, :, :] = 0
    # min_max_scaler_memmap.flush()

    empty_features_mask_memmap_path = str(args.dataset_root / "empty_features_mask.memmap")
    # empty_features_mask_memmap = np.memmap(empty_features_mask_memmap_path, dtype='float32', mode='w+', shape=(memmap_shape[0], memmap_shape[1]))
    # empty_features_mask_memmap[:, :] = empty_features_mask
    # empty_features_mask_memmap.flush()

    _scaling_arguments = []
    for i in range(args.processes):
        start_index = i * _segment_size
        end_index = (i + 1) * _segment_size if i != args.processes - 1 else number_of_timestamps

        _scaling_arguments.append(
            [min_max_scaler_memmap_path, std_scaler_memmap_path, initial_spatiotemporal_features_path,
             empty_features_mask[start_index:end_index, ...], start_index, end_index,
             means_features_wise, stds_fetures_wise, maxes_features, mins_features, memmap_shape]
        )

    with mp.Pool(processes=args.processes) as pool:
        results_job_wise = pool.starmap(process_features_based_on_statistics, _scaling_arguments)


if __name__ == '__main__':
    main()
