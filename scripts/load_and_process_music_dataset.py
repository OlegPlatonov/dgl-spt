import yt.wrapper as yt
import os
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, Lock
import typing as tp
import datetime


LOCK = Lock()

os.environ["YT_ALLOW_HTTP_REQUESTS_TO_YT_FROM_JOB"] = "1"


DATE_START = datetime.datetime(year=2024, month=7, day=1)
DATE_FINISH = datetime.datetime(year=2024, month=10, day=20)

ONE_DAY_DELTA = datetime.timedelta(days=1)

TARGETS_GRANULARITY_SECONDS = 60 * 60  # 1hr


FEATURES_TABLES_YT_INPUT_DIR = '//home/yr/ostroumova-la/music/processed-features-daily-hourly/filtered-users-from-01-07-2024-to-20-10-2024'
TARGETS_YT_INPUT_TABLE = '//home/yr/fvelikon/nirvana/5d24d125-130c-4c67-bbeb-6f3043f59c3a/output1___ydpv-QHT7KILqYEBAqwcw'
GRAPH_EDGELIST_YT_INPUT_TABLE = '//home/yr/fvelikon/nirvana/64aba77b-64d1-4cce-b4bb-2fe890da730e/output1__cl9h3u-AQvSWYMJaWMwfDQ'
NODE_ID_TO_INDEX_INPUT_TABLE = '//home/yr/fvelikon/nirvana/287639fc-0e5e-4d29-b2c7-820c610f22fe/output1__Upq8HTKJTBSMdsaHZbZpJQ'



LOCAL_FILES_DIR = Path("/mnt/ar_hdd/fvelikon/graph-time-series/datasets/music")
LOCAL_FILES_ST_FEATURES_DIR = LOCAL_FILES_DIR / "spatiotemporal_features"
LOCAL_FILES_ST_FEATURES_DIR.mkdir(exist_ok=True, parents=True)


DATASET_OUTPUT_YT_DIR = '//home/yr/fvelikon/graph_time_series/datasets/music'
DATASET_FEATURES_OUTPUT_YT_DIR = DATASET_OUTPUT_YT_DIR + "/spatiotemporal_features"


N_PARALLEL_PROCESSES = 32
YT_TOKEN = os.environ["YT_TOKEN"]
DATE_STR_FORMAT = r"%Y-%m-%d"

YT_CONFIG  = {
            "read_retries": {"enable": True},
            "allow_receive_token_by_current_ssh_session": True,
            "table_writer": {"desired_chunk_size": 1024 * 1024 * 500},
            "concatenate_retries": {
                "enable": True,
                "total_timeout": datetime.timedelta(minutes=128),
            },
            "write_retries": {"enable": True, "count": 30},
            "read_parallel": {
                "enable": True,
                "max_thread_count": 64,
            },
            "write_parallel" : {
                "enable": True,
                "max_thread_count": 64,
            }
        }


def make_client():
    client = yt.YtClient(proxy="hahn", token=YT_TOKEN, config=YT_CONFIG)
    return client


def load_table(table: Path, client: yt.YtClient | None = None) -> pd.DataFrame:
    client: yt.YtClient = client or make_client()
    num_rows = client.get_attribute(table, "row_count")
    rows_iterator = client.read_table(table, unordered=True)

    df = pd.DataFrame(tqdm(list(rows_iterator), total=num_rows, desc=f"Reading {table}"))
    return df


def upload_object_to_file(object_local_path: Path, yt_destination_path: Path, client: yt.YtClient | None = None) -> None:
    client: yt.YtClient = client or make_client()
    source_path: str = str(object_local_path)
    dest_path: str = str(yt_destination_path)
    client.smart_upload_file(filename=source_path, destination=dest_path, placement_strategy="replace")




def _mp_read_write_upload_function(source_remote_yt_path: str,
                                   destination_local_path: str,
                                   destination_remote_path: str,
                                   node_id_to_index_mapping: dict[int, int],
                                   timestamp_utc_seconds_to_timestamp_index_mapping: dict[int, int],
                                   num_tries: int = 10,
                                   ) -> str:
    # load file
    job_client: yt.YtClient = make_client()
    table_df = load_table(source_remote_yt_path, client=job_client)

    # transform it
    # user id to index
    # timestamp utc seconds to index

    table_df["user_id"] = table_df["user_id"].map(lambda x: node_id_to_index_mapping[x])
    table_df["timestamp_utc_seconds"] = table_df["timestamp_utc_seconds"].map(lambda x: timestamp_utc_seconds_to_timestamp_index_mapping[x])
    table_df["timestamp_utc_seconds"] -= table_df["timestamp_utc_seconds"].min()

    features_flattened = table_df["features"].values
    features_flattened = np.stack(features_flattened)
    
    # breakpoint()

    features_array = np.zeros((24, # number of timestamps during one day
                               len(node_id_to_index_mapping),
                               len(features_flattened[0]))
                              )
    

    features_array[table_df["timestamp_utc_seconds"].values, table_df["user_id"].values, :] = features_flattened.squeeze(-1)
    # breakpoint()

    # save locally
    np.save(destination_local_path, arr=features_array)
    print(f"Downloaded and saved\t[{source_remote_yt_path} -----> {destination_local_path}]")

    # # # load back transformed data
    # with open(destination_local_path + ".npy", "rb") as reader:
    #     job_client.write_file(destination_remote_path, reader, force_create=True)

    # print(f"Uploaded\t[{destination_local_path} -----> {destination_remote_path}]")
    return destination_local_path, np.array(features_array.shape)


def get_date_range() -> list[str]:
    days_range = []

    _date: datetime.datetime = DATE_START
    while _date <= DATE_FINISH:
        date_str = _date.strftime(DATE_STR_FORMAT)
        days_range.append(date_str)
        _date += ONE_DAY_DELTA

    print(f"{days_range=}")
    print(f"Number of days: {len(days_range)}")

    return days_range


def main():
    days_range = get_date_range()

    global_client = make_client()

    edgelist_table: pd.DataFrame = load_table(table=GRAPH_EDGELIST_YT_INPUT_TABLE, client=global_client)
    print(f"{edgelist_table=}")
    _node2index_table: pd.DataFrame = load_table(table=NODE_ID_TO_INDEX_INPUT_TABLE, client=global_client)
    print(f"{_node2index_table=}")

    node_id_to_index_mapping = dict(zip(_node2index_table["user_id"].values, _node2index_table["index"].values))

    edgelist_table["source"] = edgelist_table["source"].map(lambda x: node_id_to_index_mapping[x])
    edgelist_table["target"] = edgelist_table["target"].map(lambda x: node_id_to_index_mapping[x])

    print("Encoded edge list")
    print(f"{edgelist_table=}")

    targets = load_table(table=TARGETS_YT_INPUT_TABLE, client=global_client)
    print("Loaded targets")
    utc_timestamps = targets["timestamp_utc_seconds"].values.copy()

    targets["user_id"] = targets["user_id"].map(lambda x: node_id_to_index_mapping[x])
    targets["timestamp"] = targets["timestamp_utc_seconds"] // TARGETS_GRANULARITY_SECONDS
    targets["timestamp"] = targets["timestamp"] - targets["timestamp"].min()

    number_of_timestamps = targets["timestamp"].nunique() + 1
    number_of_nodes = len(node_id_to_index_mapping)
    print(f"There are total {number_of_timestamps} timestamps and {number_of_nodes} nodes in the dataset")

    timestamp_utc_seconds_to_timestamp_index_mapping: dict[int, int] = dict(zip(utc_timestamps, targets["timestamp"].values))

    print("Encoded user ids in targets, encoded timestamps in targets")

    # process features
    with Pool(processes=N_PARALLEL_PROCESSES) as workers_pool:
        paths_to_download = [
            FEATURES_TABLES_YT_INPUT_DIR + f"/{date}"
            for date in days_range
        ]

        spatiotemporal_features_local_files = [
            str(LOCAL_FILES_ST_FEATURES_DIR / date)
            for date in days_range
        ]
    
        spatiotemporal_transformed_features_remote_files = [
            DATASET_FEATURES_OUTPUT_YT_DIR + f"/{date}.npy"
            for date in days_range
        ]
        
        arguments = zip(paths_to_download,
                        spatiotemporal_features_local_files,
                        spatiotemporal_transformed_features_remote_files,
                        [node_id_to_index_mapping] * len(days_range),
                        [timestamp_utc_seconds_to_timestamp_index_mapping] * len(days_range),
                        )

        pass 
        # dynamic_features_logs = workers_pool.starmap(_mp_read_write_upload_function, arguments)


    # TODO SAVE Local
    # TODO LOAD
    # TODO UPLOAD all remaining attributes
    # _mp_read_write_upload_function(*next(arguments))


    edges = edgelist_table.values

    # transform targets array
    targets_array = np.zeros(shape=(number_of_timestamps, number_of_nodes))
    targets_array[targets["timestamp"].values, targets["user_id"].values] = targets["total_played_seconds_target"].values
    train_timestamps = ...
    val_timestamps = ...
    test_timestamps = ...

    spatial_node_features = np.empty(shape=(1, number_of_nodes, 0))
    spatial_node_feature_names = np.array([])

    temporal_node_features = ...
    temporal_node_feature_names = np.array(["timestamp_utc_seconds"])

    spatiotemporal_node_features = ... # already produces in the separate files
    spatiotemporal_node_feature_names = ... # ND should be tprch memmap files already???

    num_feature_names = ... # all features are numerical

    bin_feature_names = cat_feature_names = np.array([])
    

    num_timestamps = number_of_timestamps
    num_nodes = number_of_nodes



    first_timestamp_datetime: datetime.datetime = datetime.datetime.fromtimestamp(utc_timestamps.min())
    last_timestamp_datetime: datetime.datetime = datetime.datetime.fromtimestamp(utc_timestamps.max())
    timestamp_frequency: datetime.timedelta = datetime.timedelta(hours=TARGETS_GRANULARITY_SECONDS // 3600)




    x = iter(arguments).__next__()
    _ = _mp_read_write_upload_function(*x)

if __name__ == '__main__':
    main()