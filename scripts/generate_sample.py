import numpy as np
import pandas as pd
from pathlib import Path


def save_with_fallback(df: pd.DataFrame, out_base: Path):
    """
    Try Parquet first (smaller, preserves dtypes).
    If pyarrow/fastparquet isn't installed, fall back to CSV.
    Returns the final file path.
    """
    try:
        fp = out_base.with_suffix(".parquet")
        # Keep index (timestamps, etc.)
        df.to_parquet(fp)
        return fp
    except Exception as e:
        print(f"Caught exception: {e}")
        fp = out_base.with_suffix(".csv")
        df.to_csv(fp)
        return fp


# symbolic links to the full files:
CITY_TRAFFIC_L_SPEED_PATH = "city_traffic_L_speed_link"
CITY_TRAFFIC_L_VOLUME_PATH = "city_traffic_L_volume_link"

TIMESTAMPS_OFFSET = 12 * 24 * 7 # get the second week of the dataset

NUM_SAMPLE_TIMESTAMPS = 12 * 24 * 14 # timestamps in an hour x hours in a day x 2 weeks

NUM_TRAIN_TIMESTAMPS = 12 * 24 * 7
NUM_VAL_TIMESTAMPS = 12 * 24 * 4
NUM_TEST_TIMESTAMPS = NUM_SAMPLE_TIMESTAMPS - NUM_TRAIN_TIMESTAMPS - NUM_VAL_TIMESTAMPS


timestamps_slice = slice(TIMESTAMPS_OFFSET, TIMESTAMPS_OFFSET + NUM_SAMPLE_TIMESTAMPS)

dataset_full_speed = dict(**np.load(CITY_TRAFFIC_L_SPEED_PATH, allow_pickle=True))

dataset_sample = {}

 
# copy dataset keys not requiring any slices:
keys_without_slices = [
    'spatial_node_features',
    'spatial_node_feature_names',
    'temporal_node_feature_names',
    'spatiotemporal_node_feature_names',
    'num_feature_names',
    'bin_feature_names',
    'cat_feature_names',
    'edges',
]

for k in keys_without_slices:
    dataset_sample[k] = dataset_full_speed[k]


# copy dataset keys requiring slices across temporal dimension:
# temporal dimension is the first one, see the specification iof the npz data file
keys_requiring_slicing = [
    'targets',
    'temporal_node_features',
    'spatiotemporal_node_features',
    'unix_timestamps',
]

for k in keys_requiring_slicing:
    dataset_sample[k] = dataset_full_speed[k][timestamps_slice, ...]
    # check that the shape of other dimensions is preserved:
    assert dataset_sample[k].shape[0] == NUM_SAMPLE_TIMESTAMPS
    assert dataset_sample[k].shape[1:] == dataset_full_speed[k].shape[1:], f"Shape mismatch of non-temporal dimensions for `{k}` key: Full tensor shape is {dataset_full[k].shape} while subset tensor shape is {dataset_sample[k].shape}"

# handle specific keys defining train/val/test split:
train_indices = np.arange(0, NUM_TRAIN_TIMESTAMPS)
val_indices = np.arange(NUM_TRAIN_TIMESTAMPS, NUM_TRAIN_TIMESTAMPS + NUM_VAL_TIMESTAMPS)
test_indices = np.arange(NUM_TRAIN_TIMESTAMPS + NUM_VAL_TIMESTAMPS, NUM_SAMPLE_TIMESTAMPS)

assert len(train_indices) + len(val_indices) + len(test_indices) == NUM_SAMPLE_TIMESTAMPS

dataset_sample['train_timestamps'] = train_indices
dataset_sample['val_timestamps'] = val_indices
dataset_sample['test_timestamps'] = test_indices



assert len(dataset_sample.keys()) == len(dataset_full_speed.keys())

np.savez_compressed("city-traffic-L-speed-SAMPLE.npz", **dataset_sample)

## PART 2 -- save raw files for further analysis

dataset_full_volume = np.load(CITY_TRAFFIC_L_VOLUME_PATH, allow_pickle=True) # we need volume to extract raw traffic volume targets


speed_targets = dataset_sample["targets"]
volume_targets = dataset_full_volume["targets"][timestamps_slice, ...]

spatial_features = dataset_sample["spatial_node_features"][0]

timestamps_datetime = dataset_sample["unix_timestamps"].astype("datetime64[s]")

static_df = pd.DataFrame(spatial_features, columns=dataset_sample["spatial_node_feature_names"])
categorical_columns_mask = [col in dataset_sample["cat_feature_names"] for col in static_df.columns]
binary_columns_mask = [col in dataset_sample["bin_feature_names"] for col in static_df.columns]

static_df.loc[:, categorical_columns_mask] = static_df.loc[:, categorical_columns_mask].astype(int)
static_df.loc[:, binary_columns_mask] = static_df.loc[:, binary_columns_mask].astype(bool)


print("Created static features df")

edges_df = pd.DataFrame(dataset_sample["edges"], columns=["source", "target"])

print("Created edges df")

speed_df = pd.DataFrame(speed_targets, index=timestamps_datetime)
speed_df.columns = [f"node_{i}" for i in range(speed_df.shape[1])]
speed_df.index.name = "timestamp"

print("created speed df")
volume_df = pd.DataFrame(volume_targets, index=timestamps_datetime)
volume_df.index.name = "timestamp"
volume_df.columns = [f"node_{i}" for i in range(volume_df.shape[1])]

print("created volume df")

print("DataFrames ready: static_df, edges_df, speed_df, volume_df")

outdir = Path().cwd()
out_static = save_with_fallback(static_df, outdir / "city-traffic-L-raw-static-features-SAMPLE")
out_edges  = save_with_fallback(edges_df,  outdir / "city-traffic-L-raw-edges-SAMPLE")
out_speed  = save_with_fallback(speed_df,  outdir / "city-traffic-L-raw-speed-SAMPLE")
out_volume = save_with_fallback(volume_df, outdir / "city-traffic-L-raw-volume-SAMPLE")
