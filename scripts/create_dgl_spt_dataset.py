import argparse

from pathlib import Path


import torch
import numpy as np
import pandas as pd
import typing as tp

from datetime import timedelta




road_numerical_feature_names = [
    'length',
    'num_segments',
    'x_coordinate_start',
    'y_coordinate_start',
    'x_coordinate_end',
    'y_coordinate_end',
]

road_categorical_feature_names = [
    'category',
    'edge_type',
    'speed_mode',
    'speed_limit',
    'region_id',
]

road_binary_feature_names = [
    'can_bind_to_reverse_edge',
    'dismount_bike',
    'has_masstransit_lane',
    'ends_with_crosswalk',
    'ends_with_toll_post',
    'is_in_poor_condition',
    'is_paved',
    'is_restricted_for_trucks',
    'is_toll',
    'access_0',
    'access_1',
    'access_2',
    'access_3',
    'access_4',
    'access_5',
]

def create_split_indices(num_timestamps: int, split_type: tp.Literal['train', 'val', 'test']) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_indices = np.arange(num_timestamps)
    return all_indices, all_indices, all_indices # BUG dgl-spt wants all indices to be nonempty, we will limit the splits in the framework

    empty = np.arange(0)
    empty2 = empty.copy()

    if split_type == 'train':
        return all_indices, empty, empty2
    if split_type == 'val':
        return empty, all_indices, empty2
    if split_type == 'test':
        return empty, empty2, all_indices

    raise ValueError(f"{split_type=} isn't supported!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--input-dir", "-i", type=Path, help="Dir containing the ALL results of `load_traffic_componnent.py` script")
    parser.add_argument("--split", choices=['train', 'val', 'test'], required=True, help="Temporal split of current data shard [Train/Val/Test]")

    parser.add_argument("--out", "-o", help="Name (and path) to the output_file")

    args = parser.parse_args()

    input_dir: Path = args.input_dir
    out_name: str = args.out

    static_features_df = pd.read_csv(input_dir / 'features_processed.csv', index_col=0)
    unix_timestamps: np.ndarray = np.load(input_dir / 'timestamps.npy')
    edges: np.ndarray = torch.load(input_dir / 'edge_index.pt').numpy()
    travel_speed_tensor: np.ndarray = torch.load(input_dir / 'travel_speed_tensor.pt').float().numpy()
    traverses_tensor: np.ndarray = torch.load(input_dir / 'traverses_tensor.pt').float().numpy()



    N = static_features_df.shape[0]
    T = len(unix_timestamps)


    train_indices, val_indices, test_indices = create_split_indices(T, split_type=args.split)


    static_features_ordered = road_binary_feature_names + road_categorical_feature_names + road_numerical_feature_names
    static_features = static_features_df.loc[:, static_features_ordered].values[None, ...]


    # make spatial features from dataframe:

    # breakpoint()


    dataset = dict(
        unix_timestamps=unix_timestamps,

        targets=travel_speed_tensor.T,

        train_timestamps=train_indices,
        val_timestamps=val_indices,
        test_timestamps=test_indices,

        spatial_node_features=static_features,
        spatial_node_feature_names=np.array(static_features_ordered),
        temporal_node_features=np.empty(shape=(T, 1, 0)),
        temporal_node_feature_names=np.array([]),
        spatiotemporal_node_features=traverses_tensor.T[..., None],
        spatiotemporal_node_feature_names=np.array(['traverses']),

        num_feature_names=np.array(road_numerical_feature_names + ['traverses']),
        bin_feature_names=np.array(road_binary_feature_names),
        cat_feature_names=np.array(road_categorical_feature_names),

        edges=edges,
        num_timestamps=T,
        num_nodes=N,
        timestamp_frequency=timedelta(minutes=5),
    )

    np.savez(out_name, **dataset)
    print(f"Dataset is saved at `{out_name}`")
