__doc__ = """
This script loads time-series component of the dataset. Returns processed spatiotemporal tensor
"""


import os

from tqdm import tqdm
import numpy as np
import torch
import yt.wrapper as yt
import pandas as pd
import typing as tp

from pathlib import Path

import gc


from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder


import argparse



GRANULARITY_MINS = 5


yt.config.config['token'] = os.environ.get("YT_TOKEN")
yt.config.config['proxy']['url'] = "hahn"



client = yt.YtClient(proxy="hahn", token=os.environ.get("YT_TOKEN"))



def convert_nan(x, value=0):
    x = np.nan_to_num(x, value)
    x = x if x is not None else value
    return x


def get_coordinate_start(geometry_feature):
    return geometry_feature[0]

def get_coordinate_end(geometry_feature):
    return geometry_feature[-1]

def process_graph_features(item):
    x_coordinate_start, y_coordinate_start = get_coordinate_start(item['geometry'])
    x_coordinate_end, y_coordinate_end = get_coordinate_end(item['geometry'])
    return {
        'access_mask': str(bin(int(item['access_mask'])))[2:], # skip '0b' from the start
        'category': np.int64(item['category']),
        'edge_type': str(item['edge_type']),

        'can_bind_to_reverse_edge': bool(item['can_bind_to_reverse_edge']),
        'dismount_bike': bool(item['dismount_bike']),
        'has_masstransit_lane': bool(item['has_masstransit_lane']),
        
        'ends_with_crosswalk': bool(item['ends_with_crosswalk']),
        'ends_with_railroad_crossing': bool(item['ends_with_railroad_crossing']),
        'ends_with_toll_post': bool(item['ends_with_toll_post']),
        'ends_with_traffic_light': bool(item['ends_with_traffic_light']),
        
        'is_in_poor_condition': bool(item['is_in_poor_condition']),
        'is_paved': bool(item['is_paved']),
        'is_residential': bool(item['is_residential']),
        'is_restricted_for_trucks': bool(item['is_restricted_for_trucks']),
        'is_toll': bool(item['is_toll']),
        
        'length': np.float64(item['edge_length']), # np.float64(item['length']),
        'speed_mode': np.float64(item['speed']),
        'speed_limit': np.float64(convert_nan(item['speed_limit'], value=np.nan)),
        
        'road_id': item['persistent_id'],
        'region_id': str(item['region_id']),
        'num_segments': np.int64(item['edge_segments_number']), # np.int64(item['segments']),
        'source': np.int64(item['source_vertex_id']), # np.int64(item['source']),
        'target': np.int64(item['target_vertex_id']), # np.int64(item['target']),

        ### v2
        'x_coordinate_start': x_coordinate_start,
        'y_coordinate_start': y_coordinate_start,
        'x_coordinate_end': x_coordinate_end,
        'y_coordinate_end': y_coordinate_end,
    }


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--time-series-table-yt-path", help='Input table with time-series component in YT')
    parser.add_argument("--static-features-yt-path", help='Input table with time-series component in YT')
    parser.add_argument("--outdir", type=Path, required=True)

    return parser.parse_args()




graph_feature_name_to_dtype = {
    'access_mask': object,
    'category': np.int64,
    'edge_type': object,

    'can_bind_to_reverse_edge': bool,
    'dismount_bike': bool,
    'has_masstransit_lane': bool,

    'ends_with_crosswalk': bool,
    'ends_with_railroad_crossing': bool,
    'ends_with_toll_post': bool,
    'ends_with_traffic_light': bool,

    'is_in_poor_condition': bool,
    'is_paved': bool,
    'is_residential': bool,
    'is_restricted_for_trucks': bool,
    'is_toll': bool,

    'length': np.float64,
    'speed_mode': np.float64,
    'speed_limit': np.float64,

    'road_id': object,
    'region_id': object,
    'num_segments': np.int64,
    'source': np.int64,
    'target': np.int64,

    ### v2
    'x_coordinate_start': np.float64,
    'y_coordinate_start': np.float64,
    'x_coordinate_end': np.float64,
    'y_coordinate_end': np.float64,
}



def prepare_static_features(static_features_path: str) -> pd.DataFrame:
    NUM_ROWS = client.get_attribute(static_features_path, 'row_count')

    yt_table_iterator = yt.read_table(
        table=yt.TablePath(static_features_path), 
        format=yt.YsonFormat(), 
        raw=False, 
        enable_read_parallel=True
    )


    # TODO MAKE TRANSFORMATION FROM GLEB CODE
    df = pd.DataFrame(list(map(lambda row: process_graph_features(row), tqdm(yt_table_iterator, total=NUM_ROWS, desc="Loading static features raw table"))))


    def f(entry, idx):
        try:
            return bool(int(entry[idx]))
        except IndexError:
            return False

    for idx in range(6):
        df[f"access_{idx}"] = df["access_mask"].apply(
            lambda entry: f(entry, idx)
        )
        print(f"Processed access {idx=}")

    df = df.drop(columns="access_mask")
    print("Loaded& gransformed raw table")

    print("Transformed raw table")


    road_numerical_feature_names = [
        "length",
        "num_segments",
        # 'travel_velocity',
        "x_coordinate_start",
        "y_coordinate_start",
        "x_coordinate_end",
        "y_coordinate_end",
    ]

    road_categorical_feature_names = [
        "category",
        "edge_type",
        "speed_mode",
        "speed_limit",
        "region_id",
    ]

    road_binary_feature_names = [
        "can_bind_to_reverse_edge",
        "dismount_bike",
        "has_masstransit_lane",
        "ends_with_crosswalk",
        "ends_with_toll_post",
        "is_in_poor_condition",
        "is_paved",
        # 'is_residential',
        "is_restricted_for_trucks",
        "is_toll",
        "access_0",
        "access_1",
        "access_2",
        "access_3",
        "access_4",
        "access_5",
    ]

    road_non_numerical_feature_names = (
        road_categorical_feature_names + road_binary_feature_names
    )
    road_feature_names = road_non_numerical_feature_names + road_numerical_feature_names

    processed_df = df[road_feature_names].copy()

    # for some reason, ordinal encoder becomes very slow when encounters numerous NaNs
    # so we process categorical features in another way
    processed_df["speed_limit"] = (
        processed_df["speed_limit"].fillna(value=-1.0).astype(np.int64)
    )   

    encoder = OrdinalEncoder(encoded_missing_value=-1.0)
    processed_df.loc[:, road_categorical_feature_names] = (
        encoder.fit_transform(
            processed_df[road_categorical_feature_names].values
        )
    )


    processed_df[road_binary_feature_names] = processed_df[
        road_binary_feature_names
    ].astype(np.float64)


    processed_df["road_id"] = df["road_id"]


    edge_index = create_edge_list(df)

    return processed_df, edge_index



def create_edge_list(static_features_df: pd.DataFrame):

    road_segments_df = static_features_df.copy()
    unique_road_ids = set(road_segments_df["road_id"].unique().tolist())
    print(f"Number of unique road_ids (nodes) = {len(unique_road_ids)/1e6:.3f}K")


    road_segments_df = road_segments_df.reset_index().rename({"index": "node_idx"}, axis="columns")

    crossing_ids = set(road_segments_df["source"].values.tolist()) | set(road_segments_df["target"].values.tolist())
    print(f"Number of unique crossings (road endpoints) = {len(crossing_ids)/1e6:.3f}K")


    crossing_id_to_index_mapping = dict(map(reversed, enumerate(crossing_ids)))  # crossind_id --> index
    crossing_index_to_id_mapping = dict(enumerate(crossing_ids))                 # index --> crossing_id


    road_segments_df["source_idx"] = road_segments_df["source"].map(crossing_id_to_index_mapping)
    road_segments_df["target_idx"] = road_segments_df["target"].map(crossing_id_to_index_mapping)


    road_segments_truncated = road_segments_df.loc[:, ["node_idx", "source_idx", "target_idx"]]

    road_segments_connected_by_crossings_df = road_segments_truncated.merge(
        road_segments_truncated,
        left_on="target_idx",
        right_on="source_idx",
        suffixes=["__source", "__target"]
)


    edges_df = road_segments_connected_by_crossings_df.loc[:, ["node_idx__source", "node_idx__target"]]


    edge_list = torch.from_numpy(edges_df.values)
    
    # edges: pd.DataFrame = None

    # sources = torch.tensor(edges["source"].values)
    # targets  = torch.tensor(edges["target"].values)

    # edge_index = torch.vstack([sources, targets])

    return edge_list

def load_time_series_table(time_series_table: str) -> tuple[int, int, dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    NUM_ROWS = client.get_attribute(time_series_table, 'row_count')

    yt_table_iterator = yt.read_table(
        table=yt.TablePath(time_series_raw_table_path), 
        format=yt.YsonFormat(), 
        raw=False, 
        enable_read_parallel=True
    )


    min_timestamp = torch.inf
    max_timestamp = -torch.inf


    # loop 1 -- get max and min timestamps:
    for i, row in tqdm(enumerate(yt_table_iterator, 1), total=NUM_ROWS, desc="Loading table"):
        
        timestamp = row['utc_timestamp'] // GRANULARITY_MINS // 60
        
        min_timestamp = min(min_timestamp, timestamp)
        max_timestamp = max(max_timestamp, timestamp)

        # if i > 100_000: break  # BUG

    print(f"Maximal timestamp is {max_timestamp}, minimal is {min_timestamp}")

    # ids: torch.Tensor = torch.tensor(ids)
    num_timestamps = max_timestamp - min_timestamp + 1

    print(f"Number of timestamps: {num_timestamps}")


    # loop 2 -- get time series for each persistent id

    yt_table_iterator = yt.read_table(
        table=yt.TablePath(time_series_raw_table_path), 
        format=yt.YsonFormat(), 
        raw=False, 
        enable_read_parallel=True
    )



    persistent_id_to_travel_time: dict[int, torch.Tensor] = {}
    persistent_id_to_traverses: dict[int, torch.Tensor] = {}

    for i, row in tqdm(enumerate(yt_table_iterator, 1), total=NUM_ROWS, desc="Loading table"):
        id_ = row["persistent_id"]

        utc_timestamp = row["utc_timestamp"]
        timestamp_index = (utc_timestamp // GRANULARITY_MINS // 60 - min_timestamp)


        travel_time = row["avg_travel_time"]
        traverses = row["total_traverses_count"]

        if id_ not in persistent_id_to_travel_time:
            empty_travel_time_tensor = torch.empty(size=(num_timestamps, ))
            empty_travel_time_tensor.fill_(torch.nan)

            persistent_id_to_travel_time[id_] = empty_travel_time_tensor

        if id_ not in persistent_id_to_traverses:
            empty_traverses_tensor = torch.zeros(size=(num_timestamps, ))

            persistent_id_to_traverses[id_] = empty_traverses_tensor

        persistent_id_to_travel_time[id_][timestamp_index] = travel_time
        persistent_id_to_traverses[id_][timestamp_index] = traverses

        # if i > 100_000: break  # BUG


    return min_timestamp, max_timestamp, persistent_id_to_travel_time, persistent_id_to_traverses

if __name__ == '__main__':
    args = get_args()

    time_series_raw_table_path: str = args.time_series_table_yt_path
    static_features_raw_table_path: str = args.static_features_yt_path
    outdir: Path = args.outdir

    min_timestamp, max_timestamp, persistent_id_to_travel_time, persistent_id_to_traverses = load_time_series_table(time_series_raw_table_path)

    print(f"Road indices with at least one timestamp: {len(persistent_id_to_travel_time)}")

    # process static features table -- expand coordinates, expand access category feature, etc. # TODO wait Gleb

    static_features_df_processed, edge_index = prepare_static_features(static_features_raw_table_path)

    print(f"{static_features_df_processed=}")

    road_ids_with_features_array = static_features_df_processed['road_id'] # TODO

    # create a bijection between index and __road_id
    index_to_road_id = dict(enumerate(road_ids_with_features_array))
    road_id_to_index = dict(map(reversed, index_to_road_id.items()))

    road_id_to_length = dict(zip(static_features_df_processed['road_id'].values, static_features_df_processed['length'].values))

    print("Processed static features")

    # create edge list with this mapping

    print("Loaded features, edges.")


    num_timestamps = max_timestamp - min_timestamp + 1
    num_nodes = len(road_id_to_index)

    traverses_tensor = torch.zeros((num_nodes, num_timestamps))
    travel_speed_tensor = torch.empty((num_nodes, num_timestamps))
    travel_speed_tensor[:, :] = torch.nan



    # create spatiotemporal tensors:
    skipped = 0
    for persistent_id in tqdm(persistent_id_to_travel_time, desc="Creating dense spatiotemporal tensors", total=len(persistent_id_to_travel_time)):
        if persistent_id not in road_id_to_index:
            print(f"ID {persistent_id} doesn't have static features. Skipping")
            skipped += 1
            continue

        index = road_id_to_index[persistent_id]
        length = road_id_to_length[persistent_id]

        id_travel_times = persistent_id_to_travel_time[persistent_id]
        id_traverses = persistent_id_to_traverses[persistent_id]


        # fill traverses:
        traverses_tensor[index, :] = id_traverses
        travel_speed_tensor[index, :] = length / id_travel_times


    print(f"Skipped roads due to absense of static features: {skipped}")

    assert static_features_df_processed.shape[0] == travel_speed_tensor.shape[0] == traverses_tensor.shape[0]
    assert travel_speed_tensor.shape[1] == traverses_tensor.shape[1]



    # save all data:
    timestamps = (np.arange(num_timestamps) + min_timestamp) * GRANULARITY_MINS * 60

    static_features_df_processed.to_csv(outdir / "features_processed.csv")

    with open(outdir / "timestamps.npy", "wb") as f:
        np.save(f, timestamps)

    torch.save(traverses_tensor, outdir / "traverses_tensor.pt")
    torch.save(travel_speed_tensor, outdir / "travel_speed_tensor.pt")
    torch.save(edge_index, outdir / "edge_index.pt")


    print("DONE!!!!!!!!!")
