import random
import numpy as np
import pandas as pd
import torch
import dgl
from sklearn.preprocessing import (FunctionTransformer, StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer,
                                   QuantileTransformer, OneHotEncoder)
from sklearn.impute import SimpleImputer


class Dataset:
    transforms = {
        'none': FunctionTransformer(func=lambda x: x, inverse_func=lambda x: x),
        'standard-scaler': StandardScaler(),
        'min-max-scaler': MinMaxScaler(),
        'robust-scaler': RobustScaler(unit_variance=True),
        'power-transform-yeo-johnson': PowerTransformer(method='yeo-johnson', standardize=True),
        'quantile-transform-normal': QuantileTransformer(output_distribution='normal', subsample=1_000_000_000,
                                                         random_state=0),
        'quantile-transform-uniform': QuantileTransformer(output_distribution='uniform', subsample=1_000_000_000,
                                                          random_state=0)
    }

    def __init__(self, name, prediction_horizon=12, only_predict_at_end_of_horizon=False,
                 direct_lookback_num_steps=48, reverse_edges=False, to_undirected=False, target_transform='none',
                 transform_targets_for_each_node_separately=False, imputation_startegy_for_nan_targets='prev',
                 add_features_for_nan_targets=False, use_deepwalk_node_embeddings=False,
                 do_not_use_temporal_features=False, do_not_use_spatial_features=False,
                 do_not_use_spatiotemporal_features=False, imputation_strategy_for_num_features='most_frequent',
                 num_features_transform='none', plr_apply_to_past_targets=False, device='cpu', seed=0):
        print('Preparing data...')
        data = np.load(f'data/{name.replace("-", "_")}.npz', allow_pickle=True)

        num_timestamps = data['num_timestamps'].item()
        num_nodes = data['num_nodes'].item()

        first_timestamp_datetime = data['first_timestamp_datetime'].item()
        last_timestamp_datetime = data['last_timestamp_datetime'].item()
        timestamp_frequency = data['timestamp_frequency'].item()

        # Timestamp indices of all targets available for a particular split. Note that the number of timestamps
        # at which predictions can be made for a particular split will be different because it also depends on
        # prediction horizon (and, for train split, it also depends on lookback horizon).
        all_train_targets_timestamps = data['train_timestamps']
        all_val_targets_timestamps = data['val_timestamps']
        all_test_targets_timestamps = data['test_timestamps']

        # PREPARE TARGETS

        targets = data['targets'].astype(np.float32)

        targets_nan_mask = np.isnan(targets)
        if imputation_startegy_for_nan_targets == 'prev':
            targets_df = pd.DataFrame(targets)
            targets_df.ffill(axis=0, inplace=True)
            targets_df.bfill(axis=0, inplace=True)
            targets = targets_df.values
        elif imputation_startegy_for_nan_targets == 'zero':
            targets[targets_nan_mask] = 0
        else:
            raise ValueError(f'Unsupported value for imputation_strategy_for_nan_targets: '
                             f'{imputation_startegy_for_nan_targets}.')

        # Transform targets for training, but keep original targets for computing metrics.
        targets_orig = targets.copy()
        targets_transform = self.transforms[target_transform]
        if transform_targets_for_each_node_separately:
            targets_transform.fit(targets[all_train_targets_timestamps])
            targets = targets_transform.transform(targets)
        else:
            targets_transform.fit(targets[all_train_targets_timestamps].reshape(-1, 1))
            targets = targets_transform.transform(targets.reshape(-1, 1)).reshape(num_timestamps, num_nodes)

        # PREPARE FEATURES

        if do_not_use_temporal_features:
            temporal_features = np.empty((num_timestamps, 1, 0), dtype=np.float32)
            temporal_feature_names = []
        else:
            temporal_features = data['temporal_node_features'].astype(np.float32)
            temporal_feature_names = data['temporal_node_feature_names'].tolist()

        if do_not_use_spatial_features:
            spatial_features = np.empty((1, num_nodes, 0), dtype=np.float32)
            spatial_feature_names = []
        else:
            spatial_features = data['spatial_node_features'].astype(np.float32)
            spatial_feature_names = data['spatial_node_feature_names'].tolist()

        if do_not_use_spatiotemporal_features:
            spatiotemporal_features = np.empty((num_timestamps, num_nodes, 0), dtype=np.float32)
            spatiotemporal_feature_names = []
        else:
            spatiotemporal_features = data['spatiotemporal_node_features'].astype(np.float32)
            spatiotemporal_feature_names = data['spatiotemporal_node_feature_names'].tolist()

        if use_deepwalk_node_embeddings:
            if 'deepwalk_node_embeddings' not in data.keys():
                raise ValueError('DeepWalk node embeddings are not provided for this dataset.')

            deepwalk_embeddings = data['deepwalk_node_embeddings'].astype(np.float32)
        else:
            deepwalk_embeddings = np.empty((1, num_nodes, 0), dtype=np.float32)

        num_feature_names_set = set(data['num_feature_names'])
        bin_feature_names_set = set(data['bin_feature_names'])
        cat_feature_names_set = set(data['cat_feature_names'])

        features_groups = [temporal_features, spatial_features, spatiotemporal_features]
        feature_names_groups = [temporal_feature_names, spatial_feature_names, spatiotemporal_feature_names]
        num_features_masks_by_group = [None, None, None]
        for features_group_idx, (features, feature_names) in enumerate(zip(features_groups, feature_names_groups)):
            num_features_mask = np.zeros(features.shape[2], dtype=bool)
            cat_features_mask = np.zeros(features.shape[2], dtype=bool)
            for i, feature_name in enumerate(feature_names):
                if feature_name in num_feature_names_set:
                    num_features_mask[i] = True
                elif feature_name in cat_feature_names_set:
                    cat_features_mask[i] = True

            # Impute NaNs in numerical features and transform numerical features.
            if num_features_mask.any():
                num_features = features[:, :, num_features_mask]
                num_features_orig_shape = num_features.shape
                num_features = num_features.reshape(-1, num_features.shape[2])

                if np.isnan(features[:, :, num_features_mask]).sum() > 0:
                    imputer = SimpleImputer(strategy=imputation_strategy_for_num_features).fit(num_features)
                    num_features = imputer.transform(num_features)

                num_features = self.transforms[num_features_transform].fit_transform(num_features)

                num_features = num_features.reshape(*num_features_orig_shape)

                # Put transformed numerical features back into features array.
                slow_idx = 0
                for fast_idx in range(features.shape[2]):
                    if num_features_mask[fast_idx]:
                        features[:, :, fast_idx] = num_features[:, :, slow_idx]
                        slow_idx += 1

            # Apply one-hot encoding to categorical features.
            if cat_features_mask.any():
                cat_features = features[:, :, cat_features_mask]
                cat_features_orig_shape = cat_features.shape
                cat_features = cat_features.reshape(-1, cat_features.shape[2])
                one_hot_encoder = OneHotEncoder(sparse_output=False, dtype=np.float32).fit(cat_features)
                cat_features_encoded = one_hot_encoder.transform(cat_features)
                cat_features_encoded = cat_features_encoded.reshape(*cat_features_orig_shape[:2],
                                                                    cat_features_encoded.shape[1])

                one_hot_encoder_output_feature_names = one_hot_encoder.get_feature_names_out(
                    input_features=np.array(feature_names, dtype=object)[cat_features_mask]
                ).tolist()

                # Change features array, feature names, and numerical features mask to include one-hot encoded
                # categorical features.
                features_new = []
                feature_names_new = []
                num_features_mask_new = []
                start_idx = 0
                for i in range(features.shape[2]):
                    feature = features[:, :, i, None]
                    if not cat_features_mask[i]:
                        features_new.append(feature)
                        feature_names_new.append(feature_names[i])
                        num_features_mask_new.append(num_features_mask[i])
                    else:
                        num_categories = len(one_hot_encoder.categories_[i])
                        feature = cat_features_encoded[:, :, start_idx:start_idx + num_categories]
                        features_new.append(feature)
                        feature_names_new += one_hot_encoder_output_feature_names[start_idx:start_idx + num_categories]
                        num_features_mask_new += [0 for _ in range(num_categories)]
                        start_idx += num_categories

                features = np.concatenate(features_new, axis=2)
                feature_names = feature_names_new
                num_features_mask = np.array(num_features_mask_new)

            features_groups[features_group_idx] = features
            feature_names_groups[features_group_idx] = feature_names
            num_features_masks_by_group[features_group_idx] = num_features_mask

        temporal_features, spatial_features, spatiotemporal_features = features_groups
        temporal_feature_names, spatial_feature_names, spatiotemporal_feature_names = feature_names_groups
        num_features_mask = np.concatenate(num_features_masks_by_group, axis=0)

        # PREPARE GRAPH

        edges = torch.from_numpy(data['edges'])
        graph = dgl.graph((edges[:, 0], edges[:, 1]), num_nodes=num_nodes, idtype=torch.int32)
        if reverse_edges:
            graph = dgl.reverse(graph)
        if to_undirected:
            graph = dgl.to_bidirected(graph)

        # PREPARE INDICES OF PAST TARGETS THAT WILL BE USED AS FEATURES, FUTURE TARGETS THAT WILL BE PREDICTED,
        # TIMESTAMPS AT WHICH PREDICTIONS WILL BE MADE

        past_timestamp_indices_for_features = - np.arange(start=0, stop=direct_lookback_num_steps)

        past_targets_features_dim = len(past_timestamp_indices_for_features) * (1 + add_features_for_nan_targets)
        num_features_mask_extentsion = np.zeros(past_targets_features_dim, dtype=bool)
        if plr_apply_to_past_targets:
            num_features_mask_extentsion[:len(past_timestamp_indices_for_features)] = 1

        num_features_mask = np.concatenate([num_features_mask_extentsion, num_features_mask], axis=0)

        if only_predict_at_end_of_horizon:
            future_timestamp_indices_for_prediction = np.array([prediction_horizon])
        else:
            future_timestamp_indices_for_prediction = np.arange(start=1, stop=prediction_horizon + 1)

        # Indices of timestamps at which predictions can be made for a particular split.
        max_prediction_index = future_timestamp_indices_for_prediction.max()

        val_timestamps = np.concatenate(
            [np.array([all_train_targets_timestamps[-1]]), all_val_targets_timestamps[:-max_prediction_index]], axis=0
        )

        test_timestamps = np.concatenate(
            [np.array([all_val_targets_timestamps[-1]]), all_test_targets_timestamps[:-max_prediction_index]], axis=0
        )

        first_train_timestamp = direct_lookback_num_steps - 1 if direct_lookback_num_steps > 0 else 0
        train_timestamps = all_train_targets_timestamps[first_train_timestamp:-max_prediction_index]

        # Remove train timestamps for which all targets are NaNs.
        train_targets_nan_mask = targets_nan_mask[all_train_targets_timestamps[first_train_timestamp + 1:]]
        if only_predict_at_end_of_horizon or prediction_horizon == 1:
            # In this case max_prediction_index is the only prediction index.
            train_targets_nan_mask = train_targets_nan_mask[max_prediction_index - 1:]
            # train_targets_nan_mask shape is now [len(train_timestamps) x num_nodes]
            drop_train_timestamps_mask = train_targets_nan_mask.all(axis=1)
        else:
            train_targets_nan_mask = torch.from_numpy(train_targets_nan_mask)
            train_targets_nan_mask = train_targets_nan_mask.unfold(dimension=0, size=prediction_horizon, step=1)
            train_targets_nan_mask = train_targets_nan_mask.numpy()
            # train_targets_nan_mask shape is now [len(train_timestamps) x num_nodes x prediction_horizon]
            drop_train_timestamps_mask = train_targets_nan_mask.all(axis=(1, 2))

        train_timestamps = train_timestamps[~drop_train_timestamps_mask]

        # STORE EVERYTHING WE MIGHT NEED

        self.name = name
        self.device = device

        self.num_timestamps = num_timestamps
        self.num_nodes = num_nodes

        self.first_timestamp_datetime = first_timestamp_datetime
        self.last_timestamp_datetime = last_timestamp_datetime
        self.timestamp_frequency = timestamp_frequency

        self.all_train_targets_timestamps = all_train_targets_timestamps
        self.all_val_targets_timestamps = all_val_targets_timestamps
        self.all_test_targets_timestamps = all_test_targets_timestamps

        self.train_timestamps = train_timestamps
        self.val_timestamps = val_timestamps
        self.test_timestamps = test_timestamps

        self.targets_orig = torch.from_numpy(targets_orig)
        self.targets = torch.from_numpy(targets)
        self.targets_nan_mask = torch.from_numpy(targets_nan_mask)
        self.targets_transform = targets_transform
        self.transform_targets_for_each_node_separately = transform_targets_for_each_node_separately

        self.temporal_features = torch.from_numpy(temporal_features)
        self.temporal_feature_names = temporal_feature_names
        self.spatial_features = torch.from_numpy(spatial_features)
        self.spatial_feature_names = spatial_feature_names
        self.spatiotemporal_features = torch.from_numpy(spatiotemporal_features)
        self.spatiotemporal_feature_names = spatiotemporal_feature_names
        self.deepwalk_embeddings = torch.from_numpy(deepwalk_embeddings)
        # Might be used for applying numerical feature embeddings.
        self.num_features_mask = torch.from_numpy(num_features_mask)

        self.graph = graph.to(device)

        self.future_timestamp_indices_for_prediction = torch.from_numpy(future_timestamp_indices_for_prediction)
        self.past_timestamp_indices_for_features = torch.from_numpy(past_timestamp_indices_for_features)

        self.add_features_for_nan_targets = add_features_for_nan_targets
        self.plr_apply_to_past_targets = plr_apply_to_past_targets

        self.targets_dim = 1 if only_predict_at_end_of_horizon else prediction_horizon
        self.features_dim = (past_targets_features_dim + temporal_features.shape[2] + spatial_features.shape[2] +
                             spatiotemporal_features.shape[2] + deepwalk_embeddings.shape[2])

        self.cur_epoch_train_timestamps_left = train_timestamps.tolist()
        random.seed(seed)
        random.shuffle(self.cur_epoch_train_timestamps_left)
        self.end_of_epoch = False

    def get_train_timestamp(self):
        try:
            timestamp = self.cur_epoch_train_timestamps_left.pop()
        except IndexError:
            raise IndexError('There are no training timestamps left. Call start_new_epoch method of the dataset '
                             'to start a new epoch.')

        if not self.cur_epoch_train_timestamps_left:
            self.end_of_epoch = True

        return timestamp

    def start_new_epoch(self):
        self.cur_epoch_train_timestamps_left = self.train_timestamps.tolist()
        random.shuffle(self.cur_epoch_train_timestamps_left)
        self.end_of_epoch = False

    def get_timestamp_data(self, timestamp):
        targets = self.targets[timestamp + self.future_timestamp_indices_for_prediction].T.squeeze(1)
        targets_nan_mask = self.targets_nan_mask[timestamp + self.future_timestamp_indices_for_prediction].T.squeeze(1)

        temporal_features = self.temporal_features[timestamp].expand(self.num_nodes, -1)
        spatial_features = self.spatial_features.squeeze(0)
        spatiotemporal_features = self.spatiotemporal_features[timestamp]
        deepwalk_embeddings = self.deepwalk_embeddings.squeeze(0)

        past_targets = self.targets[timestamp + self.past_timestamp_indices_for_features].T

        if self.add_features_for_nan_targets:
            past_targets_nan_mask = self.targets_nan_mask[timestamp + self.past_timestamp_indices_for_features].T
        else:
            past_targets_nan_mask = torch.empty(self.num_nodes, 0)

        features = torch.cat([past_targets, past_targets_nan_mask, temporal_features, spatial_features,
                              spatiotemporal_features, deepwalk_embeddings], axis=1)

        return features.to(self.device), targets.to(self.device), targets_nan_mask.to(self.device)

    def transform_preds_to_orig(self, preds):
        device = preds.device

        if self.transform_targets_for_each_node_separately:
            preds = preds.transpose(1, 2)
            shape = preds.shape
            preds = preds.reshape(-1, self.num_nodes)
            preds = preds.cpu().numpy()
            preds_orig = self.targets_transform.inverse_transform(preds)
            preds_orig = torch.tensor(preds_orig, device=device)
            preds_orig = preds_orig.reshape(*shape)
            preds_orig = preds_orig.transpose(1, 2)
        else:
            shape = preds.shape
            preds = preds.reshape(-1, 1)
            preds = preds.cpu().numpy()
            preds_orig = self.targets_transform.inverse_transform(preds)
            preds_orig = torch.tensor(preds_orig, device=device)
            preds_orig = preds_orig.reshape(*shape)

        return preds_orig

    def get_val_targets_orig(self):
        targets = self.targets_orig[self.all_val_targets_timestamps]
        targets_nan_mask = self.targets_nan_mask[self.all_val_targets_timestamps]

        targets, targets_nan_mask = self.prepare_targets_for_evaluation(targets, targets_nan_mask)

        return targets.to(self.device), targets_nan_mask.to(self.device)

    def get_test_targets_orig(self):
        targets = self.targets_orig[self.all_test_targets_timestamps]
        targets_nan_mask = self.targets_nan_mask[self.all_test_targets_timestamps]

        targets, targets_nan_mask = self.prepare_targets_for_evaluation(targets, targets_nan_mask)

        return targets.to(self.device), targets_nan_mask.to(self.device)

    def prepare_targets_for_evaluation(self, targets, targets_nan_mask):
        if self.targets_dim == 1:
            # If we only predict target for a single timestamp (which is the case if only_predict_at_end_of_horizon
            # is True or if prediction_horizon is 1), we need to shift all targets by the number of timestamps between
            # the observed timestamp and the timestamp of prediction.
            future_timestamp_idx = self.future_timestamp_indices_for_prediction[0]
            targets = targets[future_timestamp_idx - 1:]
            targets_nan_mask = targets_nan_mask[future_timestamp_idx - 1:]
        else:
            # If we predict targets for multiple timestamps, we need to go over the targets with a sliding window
            # of length num_predictions and create a tensor of shape (num_timestamps, num_nodes, num_predictions).
            targets = targets.unfold(dimension=0, size=self.targets_dim, step=1)
            targets_nan_mask = targets_nan_mask.unfold(dimension=0, size=self.targets_dim, step=1)

        return targets, targets_nan_mask
