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
import gc


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=Path, default=Path("/mnt/ar_hdd/fvelikon/graph-time-series/datasets/music"), help="Root for raw loaded data")
    parser.add_argument("--output_root", type=Path, required=True, help="Root for processed data")
    parser.add_argument("--processes", type=int, required=False, default=1, help="Number of processes")


    return parser


def calculate_features_statistics_slice(raw_features: np.ndarray):

    empty_features_mask = np.isclose(raw_features, 0).sum(-1) == raw_features.shape[-1]

    # nonempty_features = raw_features[~empty_features_mask]

    # breakpoint()

    means_features_wise = raw_features.mean((0, 1), keepdims=True)
    stds_fetures_wise = raw_features.std((0, 1), keepdims=True)
    maxes_features = raw_features.max((0, 1), keepdims=True)
    mins_features  = raw_features.min((0, 1), keepdims=True)

    print(f"{means_features_wise.shape=}")
    print(f"{stds_fetures_wise.shape=}")
    print(f"{maxes_features.shape=}")
    print(f"{mins_features.shape=}")



    print(f"{means_features_wise=}")
    print(f"{stds_fetures_wise=}")
    print(f"{maxes_features=}")
    print(f"{mins_features=}")


    # return (
    #     empty_features_mask, nonempty_features, means_features_wise, stds_fetures_wise, maxes_features,mins_features
    # )
    return (
        empty_features_mask, means_features_wise, stds_fetures_wise, maxes_features, mins_features
    )


def main():
    args = get_parser().parse_args()

    initial_spatiotemporal_features_path = str(args.dataset_root / "spatiotemporal_features.memmap")
    initial_metainfo_path = args.dataset_root / "music_dataset_except_spatiotemporal_features.npz"

    initial_metainfo = np.load(initial_metainfo_path, allow_pickle=True)


    number_of_nodes = initial_metainfo["num_nodes"].item()
    number_of_timestamps = initial_metainfo["num_timestamps"].item()
    spt_features_number = 207

    memmap_shape = (number_of_timestamps, number_of_nodes, spt_features_number)


    raw_features = np.array(np.memmap(initial_spatiotemporal_features_path, mode="r", dtype="float32", shape=memmap_shape))


    output_root_path: Path = args.output_root
    output_root_path.mkdir(parents=True, exist_ok=True)

    std_scaler_memmap_path = str(output_root_path / "standard_scaled.memmap")
    std_scaler_memmap = np.memmap(std_scaler_memmap_path, dtype='float32', mode='r+', shape=memmap_shape)
    # std_scaler_memmap[:, :, :] = raw_features
    # std_scaler_memmap.flush()
    gc.collect()

    min_max_scaler_memmap_path = str(output_root_path / "min_max_scaled.memmap")
    min_max_scaler_memmap = np.memmap(min_max_scaler_memmap_path, dtype='float32', mode='r+', shape=memmap_shape)
    # min_max_scaler_memmap[:, :, :] = raw_features
    # min_max_scaler_memmap.flush()
    gc.collect()



    (empty_features_mask,
    #  nonempty_features,
     means_features_wise,
     stds_fetures_wise,
     maxes_features,
     mins_features) = calculate_features_statistics_slice(raw_features)
    del raw_features
    gc.collect()


    std_scaler_memmap[:, :, :] -= means_features_wise
    std_scaler_memmap[:, :, :] /= (stds_fetures_wise + 1e-6)
    std_scaler_memmap.flush()
    del std_scaler_memmap
    gc.collect()

    min_max_scaler_memmap[:, :, :] -= mins_features
    min_max_scaler_memmap[:, :, :] /= (maxes_features - mins_features)
    min_max_scaler_memmap.flush()
    del min_max_scaler_memmap
    gc.collect()

    empty_features_mask_memmap_path = str(args.dataset_root / "empty_features_mask")
    np.save(empty_features_mask_memmap_path, empty_features_mask)



if __name__ == '__main__':
    main()
