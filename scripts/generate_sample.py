import numpy as np

CITY_TRAFFIC_L_SPEED_PATH = "city_traffic_L_speed_link"

TIMESTAMPS_OFFSET = 12 * 24 * 7 # get the second week of the dataset

NUM_SAMPLE_TIMESTAMPS = 12 * 24 * 14 # timestamps in an hour x hours in a day x 2 weeks

NUM_TRAIN_TIMESTAMPS = 12 * 24 * 7
NUM_VAL_TIMESTAMPS = 12 * 24 * 4
NUM_TEST_TIMESTAMPS = NUM_SAMPLE_TIMESTAMPS - NUM_TRAIN_TIMESTAMPS - NUM_VAL_TIMESTAMPS


timestamps_slice = slice(TIMESTAMPS_OFFSET, TIMESTAMPS_OFFSET + NUM_SAMPLE_TIMESTAMPS)

dataset_full = dict(**np.load(CITY_TRAFFIC_L_SPEED_PATH, allow_pickle=True))

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
    dataset_sample[k] = dataset_full[k]


# copy dataset keys requiring slices across temporal dimension:
# temporal dimension is the first one, see the specification iof the npz data file
keys_requiring_slicing = [
    'targets',
    'temporal_node_features',
    'spatiotemporal_node_features',
    'unix_timestamps',
]

for k in keys_requiring_slicing:
    dataset_sample[k] = dataset_full[k][timestamps_slice, ...]
    # check that the shape of other dimensions is preserved:
    assert dataset_sample[k].shape[0] == NUM_SAMPLE_TIMESTAMPS
    assert dataset_sample[k].shape[1:] == dataset_full[k].shape[1:], f"Shape mismatch of non-temporal dimensions for `{k}` key: Full tensor shape is {dataset_full[k].shape} while subset tensor shape is {dataset_sample[k].shape}"

# handle specific keys defining train/val/test split:
train_indices = np.arange(0, NUM_TRAIN_TIMESTAMPS)
val_indices = np.arange(NUM_TRAIN_TIMESTAMPS, NUM_TRAIN_TIMESTAMPS + NUM_VAL_TIMESTAMPS)
test_indices = np.arange(NUM_TRAIN_TIMESTAMPS + NUM_VAL_TIMESTAMPS, NUM_SAMPLE_TIMESTAMPS)

assert len(train_indices) + len(val_indices) + len(test_indices) == NUM_SAMPLE_TIMESTAMPS

dataset_sample['train_timestamps'] = train_indices
dataset_sample['val_timestamps'] = val_indices
dataset_sample['test_timestamps'] = test_indices



assert len(dataset_sample.keys()) == len(dataset_full.keys())

np.savez_compressed("city-traffic-L-speed-SAMPLE.npz", **dataset_sample)
