import yt.wrapper as yt
import os
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, Lock
import multiprocessing as mp
import typing as tp
import datetime
from time import mktime

os.environ["YT_ALLOW_HTTP_REQUESTS_TO_YT_FROM_JOB"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

# mp.set_start_method('spawn', force=True)  # Ensure compatibility on all platforms

# TODO UPLOAD all remaining attributes



LOCK = Lock()



DATE_START = datetime.datetime(year=2024, month=7, day=1)
DATE_FINISH = datetime.datetime(year=2024, month=10, day=20)

ONE_DAY_DELTA = datetime.timedelta(days=1)

TARGETS_GRANULARITY_SECONDS = 60 * 60  # 1hr

SPATIOTEMPORAL_FEATURES_DIM = 207

FEATURES_TABLES_YT_INPUT_DIR = '//home/yr/ostroumova-la/music/processed-features-daily-hourly/filtered-users-from-01-07-2024-to-20-10-2024-truncated'
TARGETS_YT_INPUT_TABLE = '//home/yr/fvelikon/nirvana/90c1acbc-f1c3-4472-a3df-4af66c798723/output1__uJAeBCA2RN-_kUPg3gevLA'
GRAPH_EDGELIST_YT_INPUT_TABLE = '//home/yr/fvelikon/nirvana/f1939781-c6bb-4b00-8995-898d7c4977cb/output1__ENc_Ztp7RnO0FXmLpLQMWQ'
NODE_ID_TO_INDEX_INPUT_TABLE = '//home/yr/fvelikon/nirvana/60ddc2e7-8ee2-4335-8405-a5460163c134/output1__1T_bNwvwTp6yNxd2miW8uQ'



LOCAL_FILES_DIR = Path("/mnt/ar_hdd/fvelikon/graph-time-series/datasets/music_truncated")


N_PARALLEL_PROCESSES = 16
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


def make_client(parallel_reading=True):
    if parallel_reading:
        client = yt.YtClient(proxy="hahn", token=YT_TOKEN, config=YT_CONFIG)
    else:
        config = YT_CONFIG.copy()
        config["read_parallel"]["enable"] = False
        config["write_parallel"]["enable"] = False

        client = yt.YtClient(proxy="hahn", token=YT_TOKEN, config=config)
    return client


def load_table(table: Path, client: yt.YtClient | None = None, read_parallel: bool = True) -> pd.DataFrame:
    client: yt.YtClient = client or make_client()
    num_rows = client.get_attribute(table, "row_count")
    rows_iterator = client.read_table(table, unordered=read_parallel, enable_read_parallel=read_parallel)

    df = pd.DataFrame(tqdm(list(rows_iterator), total=num_rows, desc=f"Reading {table}"))
    return df


def upload_object_to_file(object_local_path: Path, yt_destination_path: Path, client: yt.YtClient | None = None) -> None:
    client: yt.YtClient = client or make_client()
    source_path: str = str(object_local_path)
    dest_path: str = str(yt_destination_path)
    client.smart_upload_file(filename=source_path, destination=dest_path, placement_strategy="replace")




def _mp_read_write_function(remote_table_path: str,
                            node_id_to_index_mapping: dict[int, int],
                            timestamp_utc_seconds_to_timestamp_index_mapping: dict[int, int],
                            memmap_file_path: str,
                            memmap_file_shape: tuple[int, int, int],
                            ) -> str:
    global LOCK

    # 1) loads table
    # 2) transforms it
    #       - maps user id to index
    #       - maps timestamp utc seconds to index

    job_client = make_client(parallel_reading=False)
    table_df =  load_table(table=remote_table_path, client=job_client, read_parallel=False)

    table_df["user_id"] = table_df["user_id"].map(lambda x: node_id_to_index_mapping[x])
    table_df["timestamp_utc_seconds"] = table_df["timestamp_utc_seconds"].map(lambda x: timestamp_utc_seconds_to_timestamp_index_mapping[x])

    features_flattened = table_df["features"].values
    features_flattened = np.stack(features_flattened).squeeze(-1)  # dimensions are [(timestamp index, node index), features]

    min_timestamp = table_df["timestamp_utc_seconds"].min()
    max_timestamp = table_df["timestamp_utc_seconds"].max()


    print(f"Preprocessed timestamps [{min_timestamp}, {max_timestamp}]")

    memmap = np.memmap(memmap_file_path, dtype='float32', mode='r+', shape=memmap_file_shape)

    timestamp_indices_list = table_df["timestamp_utc_seconds"].values
    node_ids_indices_list = table_df["user_id"].values
    memmap[timestamp_indices_list, node_ids_indices_list, :] = features_flattened



    with LOCK:
        # flush changes
        memmap.flush()
    # memmap.flush()  # PRONE TO RACE CONDITIONS!!!
    print(f'Flushed timestamps [{min_timestamp}, {max_timestamp}] into memmap object')

    # num_observations = table_df.shape[0]
    # features_first_momentum = features_flattened.mean((0,))
    # features_second_momentum = (features_flattened ** 2).mean((0,))

    # return features_first_momentum, features_second_momentum, num_observations
    


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

    number_of_timestamps = targets["timestamp"].nunique()
    number_of_nodes = len(node_id_to_index_mapping)
    print(f"There are total {number_of_timestamps} timestamps and {number_of_nodes} nodes in the dataset")

    timestamp_utc_seconds_to_timestamp_index_mapping: dict[int, int] = dict(zip(utc_timestamps, targets["timestamp"].values))

    print("Encoded user ids in targets, encoded timestamps in targets")

    # # transform targets array
    targets_array = np.zeros(shape=(number_of_timestamps, number_of_nodes))
    targets_array[targets["timestamp"].values, targets["user_id"].values] = targets["total_played_seconds_target"].values
    # np.save("./targets", targets_array)

    memmap_path = str(LOCAL_FILES_DIR / "spatiotemporal_features.memmap")
    memmap_shape = (number_of_timestamps, number_of_nodes, SPATIOTEMPORAL_FEATURES_DIM)
    memmap = np.memmap(memmap_path, dtype='float32', mode='w+', shape=memmap_shape)
    memmap[:, :, :] = 0
    memmap.flush()
    # process features

    with Pool(processes=N_PARALLEL_PROCESSES) as workers_pool:
        paths_to_download = [
            FEATURES_TABLES_YT_INPUT_DIR + f"/{date}"
            for date in days_range
        ]

        arguments = zip(paths_to_download,
                        [node_id_to_index_mapping] * len(days_range),
                        [timestamp_utc_seconds_to_timestamp_index_mapping] * len(days_range),
                        [memmap_path] * len(days_range),
                        [memmap_shape] * len(days_range),
                        )
 
        _ = workers_pool.starmap(_mp_read_write_function, arguments)
        memmap.flush()


    edges = edgelist_table.values


    spatial_node_features = np.empty(shape=(1, number_of_nodes, 0))
    spatial_node_feature_names = np.array([])

    temporal_node_feature_names = np.array(["timestamp_utc_seconds"])

    spatiotemporal_node_features = None # already produced in the separate files
    spatiotemporal_node_feature_names = [f"spatiotemporal_feature_{i}" for i in range(SPATIOTEMPORAL_FEATURES_DIM)] # ND should be tprch memmap files already??? --> it will be numpy 

    num_feature_names = spatiotemporal_node_feature_names + ["timestamp_utc_seconds"]  # all features are numerical and in spatiotemporal

    bin_feature_names = cat_feature_names = np.array([])
    

    num_timestamps = number_of_timestamps
    num_nodes = number_of_nodes


    first_timestamp_datetime: datetime.datetime = datetime.datetime.fromtimestamp(utc_timestamps.min())
    last_timestamp_datetime: datetime.datetime = datetime.datetime.fromtimestamp(utc_timestamps.max())
    timestamp_frequency: datetime.timedelta = datetime.timedelta(hours=TARGETS_GRANULARITY_SECONDS // 3600)

    datetime_indices = []
    current = first_timestamp_datetime
    while current <= last_timestamp_datetime:
        datetime_indices.append(current)
        current += timestamp_frequency

    val_start_datetime: datetime.datetime =  first_timestamp_datetime + 12 * datetime.timedelta(weeks=1)
    test_start_datetime: datetime.datetime = first_timestamp_datetime + 14 * datetime.timedelta(weeks=1)

    val_start_index = datetime_indices.index(val_start_datetime)
    test_start_index = datetime_indices.index(test_start_datetime)

    train_timestamps = np.arange(0, val_start_index)
    val_timestamps = np.arange(val_start_index, test_start_index)
    test_timestamps = np.arange(test_start_index, number_of_timestamps)

    music_dict_except_spatiotemporal_features = dict(
        edges=edges,
        spatial_node_features=spatial_node_features,
        spatial_node_feature_names=spatial_node_feature_names,
        temporal_node_features=np.array([
                int(mktime(dt.timetuple())) for dt in datetime_indices
            ]).reshape(-1, 1, 1),
        temporal_node_feature_names=temporal_node_feature_names,
        spatiotemporal_node_feature_names=spatiotemporal_node_feature_names,
        spatiotemporal_node_features=spatiotemporal_node_features,
        num_feature_names=num_feature_names,
        bin_feature_names=bin_feature_names,
        cat_feature_names=cat_feature_names,
        num_timestamps=num_timestamps,
        num_nodes=num_nodes,
        first_timestamp_datetime=first_timestamp_datetime,
        last_timestamp_datetime=last_timestamp_datetime,
        timestamp_frequency=timestamp_frequency,
        train_timestamps=train_timestamps,
        val_timestamps=val_timestamps,
        test_timestamps=test_timestamps,
        val_start_datetime=val_start_datetime,
        test_start_datetime=test_start_datetime,
        targets=targets_array
    )
    np.savez_compressed(file=str(LOCAL_FILES_DIR / "music_dataset_except_spatiotemporal_features"), **music_dict_except_spatiotemporal_features)

if __name__ == '__main__':
    main()
