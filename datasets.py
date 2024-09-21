import os
from functools import cache
import numpy as np
import pandas as pd
import torch
import dgl
from sklearn.preprocessing import (FunctionTransformer, StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer,
                                   QuantileTransformer, OneHotEncoder)
from sklearn.impute import SimpleImputer

# try:
#     import nirvana_dl as ndl
# except ImportError:
#     ndl = None


class NirvanaDatasetWrapper:
    """
    Mimics default numpy npz dictionary, as Nirvana automatically unpacks it to separate arrays.
    """
    def __init__(self, root_path: str):
        self.root_path = root_path
    
    def get_array_path(self, array_name: str):
        return os.path.join(self.root_path, f"{array_name}.npy")
    
    # @cache
    def __getitem__(self, array_name: str):
        array_path = self.get_array_path(array_name)
        
        print(f"Accessing `{array_name}` array at {array_path}")
        array = np.load(array_path, allow_pickle=True)
        
        return array
    
    def __contains__(self, array_name: str):
        return os.path.exists(self.get_array_path(array_name))


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

    def __init__(self, name_or_path, prediction_horizon=12, only_predict_at_end_of_horizon=False,
                 provide_sequnce_inputs=False, direct_lookback_num_steps=48,
                 seasonal_lookback_periods=None, seasonal_lookback_num_steps=None,
                 drop_early_train_timestamps='direct',
                 reverse_edges=False, to_undirected=False, use_forward_and_reverse_edges_as_different_edge_types=False,
                 add_self_loops=False,
                 targets_for_loss_transform='none', transform_targets_for_loss_for_each_node_separately=False,
                 targets_for_features_transform='none', transform_targets_for_features_for_each_node_separately=False,
                 imputation_startegy_for_nan_targets_for_features='prev',
                 add_indicators_of_nan_targets_to_features=False,
                 do_not_use_temporal_features=False, do_not_use_spatial_features=False,
                 do_not_use_spatiotemporal_features=False, use_deepwalk_node_embeddings=False,
                 initialize_learnable_node_embeddings_with_deepwalk=False,
                 num_features_transform='none', imputation_strategy_for_num_features='most_frequent',
                 plr_apply_to_past_targets=False, train_batch_size=1, eval_batch_size=None,
                 device='cpu', in_nirvana=False):
        if name_or_path.endswith('.npz'):
            name = os.path.splitext(os.path.basename(name_or_path))[0].replace('_', '-')
            path = name_or_path
        else:
            name = name_or_path
            path = f'data/{name.replace("-", "_")}.npz'

        print('Preparing data...')
        data = NirvanaDatasetWrapper(root_path="data/") if in_nirvana else np.load(path, allow_pickle=True)        

        # GET TIME SPLITS

        # Timestamp indices of all targets available for a particular split. Note that the number of timestamps
        # at which predictions can be made for a particular split will be different because it also depends on
        # prediction horizon (and, for train split, it also depends on lookback horizon).
        all_train_targets_timestamps = data['train_timestamps']
        all_val_targets_timestamps = data['val_timestamps']
        all_test_targets_timestamps = data['test_timestamps']

        # PREPARE TARGETS

        targets = data['targets'].astype(np.float32)

        num_timestamps = data['num_timestamps'].item() if 'num_timestamps' in data else targets.shape[0]
        num_nodes = data['num_nodes'].item() if 'num_nodes' in data else targets.shape[1]

        targets_nan_mask = np.isnan(targets)

        # We will be using three arrays of targets with possibly different preprocessing transformations applied
        # to them:
        # targets_for_metrics: these targets will be used for computing metrics during evaluation (targets from future
        # timestamps that will be predicted during evaluation), no transformation can be applied to them;
        # targets_for_loss: these targets will be used for computing loss during training (targets from future
        # timestamps that will be predicted during training), some transformation can be applied to them;
        # targets_for_features: these targets will be used as additional features for the model (targets from past
        # timestamps and current timestamp), some transformation can be applied to them and NaN values in them will
        # be imputed.
        targets_for_metrics = targets
        targets_for_loss = targets.copy()
        targets_for_features = targets.copy()

        # Transform targets that will be used for computing loss.
        targets_for_loss_transform = self.transforms[targets_for_loss_transform]
        if transform_targets_for_loss_for_each_node_separately:
            targets_for_loss_transform.fit(targets_for_loss[all_train_targets_timestamps])
            targets_for_loss = targets_for_loss_transform.transform(targets_for_loss)
        else:
            targets_for_loss_transform.fit(targets_for_loss[all_train_targets_timestamps].reshape(-1, 1))
            targets_for_loss = targets_for_loss_transform.transform(targets.reshape(-1, 1))\
                .reshape(num_timestamps, num_nodes)

        # Transform targets that will be used as features for the model.
        targets_for_features_transform = self.transforms[targets_for_features_transform]
        if transform_targets_for_features_for_each_node_separately:
            targets_for_features_transform.fit(targets_for_features[all_train_targets_timestamps])
            targets_for_features = targets_for_features_transform.transform(targets_for_features)
        else:
            targets_for_features_transform.fit(targets_for_features[all_train_targets_timestamps].reshape(-1, 1))
            targets_for_features = targets_for_features_transform.transform(targets_for_features.reshape(-1, 1))\
                .reshape(num_timestamps, num_nodes)

        # Impute NaNs in targets.
        if imputation_startegy_for_nan_targets_for_features == 'prev':
            if targets_nan_mask[all_train_targets_timestamps].all(axis=0).any():
                raise RuntimeError(
                    'There are nodes in the dataset for which all train targets are NaN. "prev" imputation strategy '
                    'for NaN targets cannot be applied in this case. Modify the dataset (e.g., by removing these '
                    'nodes) or set imputation_startegy_for_nan_targets argument to "zero".'
                )

            # First, impute NaN targets with forward fill.
            targets_for_features_df = pd.DataFrame(targets_for_features)
            targets_for_features_df.ffill(axis=0, inplace=True)

            # If some nodes have NaN targets starting from the very beginning of the time series, these NaN values are
            # still left unimputed after forward fill. So, we now apply backward fill to them. Note that we have
            # already verified that there are at least some train target values that are not NaN for each node, and
            # thus it is guaranteed that this will not lead to future targets leakage from val and test timestamps.
            if np.isnan(targets_for_features_df.values).any():
                targets_for_features_df.bfill(axis=0, inplace=True)

            targets_for_features = targets_for_features_df.values

        elif imputation_startegy_for_nan_targets_for_features == 'zero':
            targets_for_features[targets_nan_mask] = 0

        else:
            raise ValueError(f'Unsupported value for imputation_strategy_for_nan_targets: '
                             f'{imputation_startegy_for_nan_targets_for_features}.')

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

        if use_deepwalk_node_embeddings or initialize_learnable_node_embeddings_with_deepwalk:
            if 'deepwalk_node_embeddings' not in data.keys():
                raise ValueError('DeepWalk node embeddings are not provided for this dataset.')

        if use_deepwalk_node_embeddings:
            deepwalk_embeddings = data['deepwalk_node_embeddings'].astype(np.float32)
        else:
            deepwalk_embeddings = np.empty((1, num_nodes, 0), dtype=np.float32)

        if initialize_learnable_node_embeddings_with_deepwalk:
            deepwalk_embeddings_for_initializing_learnable_embeddings = data['deepwalk_node_embeddings']\
                .astype(np.float32)

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

            # Transform numerical features and impute NaNs in numerical features.
            if num_features_mask.any():
                num_features = features[:, :, num_features_mask]
                num_features_orig_shape = num_features.shape
                num_features = num_features.reshape(-1, num_features.shape[2])

                # Transform numerical features.
                num_features = self.transforms[num_features_transform].fit_transform(num_features)

                # breakpoint()
                # Impute NaNs in numerical features.
                if np.isnan(features[:, :, num_features_mask]).any():
                    imputer = SimpleImputer(strategy=imputation_strategy_for_num_features, keep_empty_features=True)
                    imputer.fit(num_features)
                    num_features = imputer.transform(num_features)
                    # Some features could be removed by imputer.
                    num_features_orig_shape = (
                        num_features_orig_shape[0], num_features_orig_shape[1], num_features.shape[-1]
                    )
                # breakpoint()

                num_features = num_features.reshape(*num_features_orig_shape)

                # Put transformed numerical features back into features array.
                # breakpoint()
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
                        num_features_mask_new += [False for _ in range(num_categories)]
                        start_idx += num_categories

                features = np.concatenate(features_new, axis=2)
                feature_names = feature_names_new
                num_features_mask = np.array(num_features_mask_new, dtype=bool)

            features_groups[features_group_idx] = features
            feature_names_groups[features_group_idx] = feature_names
            num_features_masks_by_group[features_group_idx] = num_features_mask

        temporal_features, spatial_features, spatiotemporal_features = features_groups
        temporal_feature_names, spatial_feature_names, spatiotemporal_feature_names = feature_names_groups
        num_features_mask = np.concatenate(num_features_masks_by_group, axis=0)

        # PREPARE GRAPH

        if sum([reverse_edges, to_undirected, use_forward_and_reverse_edges_as_different_edge_types]) > 1:
            raise ValueError('At most one of the graph edge processing arguments reverse_edges, to_undirected, '
                             'use_forward_and_reverse_edges_as_different_edge_types can be True.')

        edges = torch.from_numpy(data['edges'])

        if use_forward_and_reverse_edges_as_different_edge_types:
            graph = dgl.heterograph(
                {
                    ('node', 'forward_edge', 'node'): (edges[:, 0], edges[:, 1]),
                    ('node', 'reverse_edge', 'node'): (edges[:, 1], edges[:, 0])
                },
                num_nodes_dict={'node': num_nodes},
                idtype=torch.int32
            )

        else:
            graph = dgl.graph((edges[:, 0], edges[:, 1]), num_nodes=num_nodes, idtype=torch.int32)
            if to_undirected:
                graph = dgl.to_bidirected(graph)
            elif reverse_edges:
                graph = dgl.reverse(graph)

        if add_self_loops:
            for cur_edge_type in graph.etypes:
                graph = dgl.add_self_loop(graph, etype=cur_edge_type)

        train_batched_graph = dgl.batch([graph for _ in range(train_batch_size)])
        if eval_batch_size is not None and eval_batch_size != train_batch_size:
            eval_batched_graph = dgl.batch([graph for _ in range(eval_batch_size)])

        # PREPARE INDEX SHIFTS FROM THE CURRENT TIMESTAMP TO PAST TARGETS THAT WILL BE USED AS FEATURES

        # Check validity of seasonal lookback arguments.
        if (provide_sequnce_inputs and
                (seasonal_lookback_periods is not None or seasonal_lookback_num_steps is not None)):
            raise ValueError(
                'Seasonal lookback is not meant to be used with sequence input, as sequential timestamps are required '
                'for sequence input to a model. Either set sequence input to False and use a single input model or '
                'disable seasonal lookback by setting seasonal_lookback_periods and seasonal_lookback_num_steps '
                'arguments to None.'
            )
        elif seasonal_lookback_periods is not None and seasonal_lookback_num_steps is None:
            raise ValueError(
                'If argument seasonal_lookback_periods is provided, then argument seasonal_lookback_num_steps should '
                'also be provided.'
            )
        elif seasonal_lookback_periods is None and seasonal_lookback_num_steps is not None:
            raise ValueError(
                'If argument seasonal_lookback_num_steps is provided, then argument seasonal_lookback_periods should '
                'also be provided.'
            )
        elif (seasonal_lookback_periods is not None and
              len(seasonal_lookback_periods) != len(seasonal_lookback_num_steps)):
            raise ValueError(
                'Arguments seasonal_lookback_periods and seasonal_lookback_num_steps should be provided with the same '
                'number of values.'
            )

        past_timestamp_shifts_for_features = - np.arange(start=0, stop=direct_lookback_num_steps)

        if seasonal_lookback_periods is not None:
            # For each predicted target at timestamp t we add to features target values from the past at the following
            # timestamps: t - period, t - period * 2, ..., t - period * num_steps.
            all_seasonal_shifts = []
            for period, num_steps in zip(seasonal_lookback_periods, seasonal_lookback_num_steps):
                start_shifts = - period * np.arange(start=1, stop=num_steps + 1)
                for start_shift in start_shifts:
                    if only_predict_at_end_of_horizon:
                        cur_shifts = np.array([start_shift + prediction_horizon])
                    else:
                        cur_shifts = np.arange(start=start_shift + 1, stop=start_shift + prediction_horizon + 1)

                    all_seasonal_shifts.append(cur_shifts)

            all_seasonal_shifts = np.hstack(all_seasonal_shifts)
            all_seasonal_shifts = all_seasonal_shifts.clip(max=0)

            past_timestamp_shifts_for_features = np.hstack([past_timestamp_shifts_for_features, all_seasonal_shifts])
            past_timestamp_shifts_for_features = np.unique(past_timestamp_shifts_for_features)
            past_timestamp_shifts_for_features.sort()
            # A copy is made after reversing the array because otherwise the stride in the array will be negative,
            # and because of this it will not be possible to load it into a torch tensor using torch.from_numpy.
            past_timestamp_shifts_for_features = past_timestamp_shifts_for_features[::-1].copy()

        past_targets_features_dim = 1 if provide_sequnce_inputs else len(past_timestamp_shifts_for_features)
        past_targets_nan_mask_features_dim = past_targets_features_dim * add_indicators_of_nan_targets_to_features
        features_dim = (past_targets_features_dim + past_targets_nan_mask_features_dim +
                        temporal_features.shape[2] + spatial_features.shape[2] + spatiotemporal_features.shape[2] +
                        deepwalk_embeddings.shape[2])

        past_targets_mask = np.zeros(features_dim, dtype=bool)
        past_targets_mask[:past_targets_features_dim] = True

        num_features_mask_extentsion = np.zeros(past_targets_features_dim + past_targets_nan_mask_features_dim,
                                                dtype=bool)
        num_features_mask = np.concatenate([num_features_mask_extentsion, num_features_mask], axis=0)

        # PREPARE INDEX SHIFTS FROM THE CURRENT TIMESTAMP TO FUTURE TARGETS THAT WILL BE PREDICTED

        if only_predict_at_end_of_horizon:
            future_timestamp_shifts_for_prediction = np.array([prediction_horizon])
        else:
            future_timestamp_shifts_for_prediction = np.arange(start=1, stop=prediction_horizon + 1)

        # PREPARE INDICES OF TIMESTAMPS AT WHICH PREDICTIONS WILL BE MADE FOR EACH SPLIT

        if provide_sequnce_inputs and drop_early_train_timestamps != 'direct':
            raise ValueError(
                f'If provide_sequnce_inputs argument is True, than only "direct" is a valid value for '
                f'drop_early_train_timestamps argument, but value {drop_early_train_timestamps} was provided instead.'
            )

        if drop_early_train_timestamps == 'all':
            first_train_timestamp = - past_timestamp_shifts_for_features.min()
        elif drop_early_train_timestamps == 'direct':
            first_train_timestamp = direct_lookback_num_steps - 1
        elif drop_early_train_timestamps == 'none':
            first_train_timestamp = 0
        else:
            raise ValueError(f'Unknown value for drop_early_train_timestamps argument: {drop_early_train_timestamps}.')

        first_train_timestamp = max(first_train_timestamp, 0)

        max_prediction_shift = future_timestamp_shifts_for_prediction[-1]

        train_timestamps = all_train_targets_timestamps[first_train_timestamp:-max_prediction_shift]

        val_timestamps = np.concatenate(
            [np.array([all_train_targets_timestamps[-1]]), all_val_targets_timestamps[:-max_prediction_shift]], axis=0
        )

        test_timestamps = np.concatenate(
            [np.array([all_val_targets_timestamps[-1]]), all_test_targets_timestamps[:-max_prediction_shift]], axis=0
        )

        # Remove train timestamps for which all targets are NaNs.
        train_targets_nan_mask = targets_nan_mask[all_train_targets_timestamps[first_train_timestamp + 1:]]
        if only_predict_at_end_of_horizon or prediction_horizon == 1:
            # In this case max_prediction_shift is the only prediction shift.
            # Transform train_targets_nan_mask to shape [len(train_timestamps), num_nodes].
            train_targets_nan_mask = train_targets_nan_mask[max_prediction_shift - 1:]
            drop_train_timestamps_mask = train_targets_nan_mask.all(axis=1)
        else:
            # Transform train_targets_nan_mask to shape [len(train_timestamps), num_nodes, prediction_horizon].
            train_targets_nan_mask = torch.from_numpy(train_targets_nan_mask)
            train_targets_nan_mask = train_targets_nan_mask.unfold(dimension=0, size=prediction_horizon, step=1)
            train_targets_nan_mask = train_targets_nan_mask.numpy()
            drop_train_timestamps_mask = train_targets_nan_mask.all(axis=(1, 2))

        train_timestamps = train_timestamps[~drop_train_timestamps_mask]

        # STORE EVERYTHING WE MIGHT NEED

        self.name = name
        self.provide_sequence_inputs = provide_sequnce_inputs
        self.device = device

        self.num_timestamps = num_timestamps
        self.num_nodes = num_nodes

        self.all_train_targets_timestamps = torch.from_numpy(all_train_targets_timestamps)
        self.all_val_targets_timestamps = torch.from_numpy(all_val_targets_timestamps)
        self.all_test_targets_timestamps = torch.from_numpy(all_test_targets_timestamps)

        self.train_timestamps = torch.from_numpy(train_timestamps)
        self.val_timestamps = torch.from_numpy(val_timestamps)
        self.test_timestamps = torch.from_numpy(test_timestamps)

        self.targets_for_metrics = torch.from_numpy(targets_for_metrics)
        self.targets_for_loss = torch.from_numpy(targets_for_loss)
        self.targets_for_features = torch.from_numpy(targets_for_features)
        self.targets_nan_mask = torch.from_numpy(targets_nan_mask)
        self.targets_for_loss_transform = targets_for_loss_transform
        self.transform_targets_for_loss_for_each_node_separately = transform_targets_for_loss_for_each_node_separately

        self.temporal_features = torch.from_numpy(temporal_features)
        self.temporal_feature_names = temporal_feature_names
        self.spatial_features = torch.from_numpy(spatial_features)
        self.spatial_feature_names = spatial_feature_names
        self.spatiotemporal_features = torch.from_numpy(spatiotemporal_features)
        self.spatiotemporal_feature_names = spatiotemporal_feature_names
        self.deepwalk_embeddings = torch.from_numpy(deepwalk_embeddings)

        self.spatial_features_batched_train = self.spatial_features.repeat(1, train_batch_size, 1).to(device)
        self.deepwalk_embeddings_batched_train = self.deepwalk_embeddings.repeat(1, train_batch_size, 1).to(device)
        if eval_batch_size is None or eval_batch_size == train_batch_size:
            self.spatial_features_batched_eval = self.spatial_features_batched_train
            self.deepwalk_embeddings_batched_eval = self.deepwalk_embeddings_batched_train
        else:
            self.spatial_features_batched_eval = self.spatial_features.repeat(1, eval_batch_size, 1).to(device)
            self.deepwalk_embeddings_batched_eval = self.deepwalk_embeddings.repeat(1, eval_batch_size, 1).to(device)

        # Might be used for applying numerical feature embeddings.
        self.num_features_mask = torch.from_numpy(num_features_mask)
        self.past_targets_mask = torch.from_numpy(past_targets_mask)

        if initialize_learnable_node_embeddings_with_deepwalk:
            self.deepwalk_embeddings_for_initializing_learnable_embeddings = torch.from_numpy(
                deepwalk_embeddings_for_initializing_learnable_embeddings
            )
        else:
            self.deepwalk_embeddings_for_initializing_learnable_embeddings = None

        self.train_batch_size = train_batch_size
        self.eval_batch_size = train_batch_size if eval_batch_size is None else eval_batch_size

        self.graph = graph
        self.train_batched_graph = train_batched_graph.to(device)
        if eval_batch_size is None or eval_batch_size == train_batch_size:
            self.eval_batched_graph = self.train_batched_graph
        else:
            self.eval_batched_graph = eval_batched_graph.to(device)

        self.future_timestamp_shifts_for_prediction = torch.from_numpy(future_timestamp_shifts_for_prediction)
        self.past_timestamp_shifts_for_features = torch.from_numpy(past_timestamp_shifts_for_features)

        self.add_indicators_of_nan_targets_to_features = add_indicators_of_nan_targets_to_features
        self.plr_apply_to_past_targets = plr_apply_to_past_targets

        self.targets_dim = 1 if only_predict_at_end_of_horizon else prediction_horizon
        self.past_targets_features_dim = past_targets_features_dim
        self.features_dim = features_dim
        self.seq_len = direct_lookback_num_steps if provide_sequnce_inputs else None

        del data

    def get_timestamp_features_as_single_input(self, timestamp):
        past_timestamps = timestamp + self.past_timestamp_shifts_for_features
        negative_mask = (past_timestamps < 0)
        past_timestamps[negative_mask] = 0

        past_targets = self.targets_for_features[past_timestamps].to(self.device).T

        if self.add_indicators_of_nan_targets_to_features:
            past_targets_nan_mask = self.targets_nan_mask[past_timestamps].to(self.device)
            past_targets_nan_mask[negative_mask] = 1
            past_targets_nan_mask = past_targets_nan_mask.T
        else:
            past_targets_nan_mask = torch.empty(self.num_nodes, 0, device=self.device)

        temporal_features = self.temporal_features[timestamp].to(self.device).expand(self.num_nodes, -1)
        spatiotemporal_features = self.spatiotemporal_features[timestamp].to(self.device)
        spatial_features = self.spatial_features.to(self.device).squeeze(0)
        deepwalk_embeddings = self.deepwalk_embeddings.to(self.device).squeeze(0)

        features = torch.cat([past_targets, past_targets_nan_mask, temporal_features, spatial_features,
                              spatiotemporal_features, deepwalk_embeddings], axis=1)

        return features

    def get_timestamp_features_as_sequence_input(self, timestamp):
        past_timestamps = timestamp + self.past_timestamp_shifts_for_features

        past_targets = self.targets_for_features[past_timestamps].to(self.device).T.unsqueeze(-1)

        if self.add_indicators_of_nan_targets_to_features:
            past_targets_nan_mask = self.targets_nan_mask[past_timestamps].to(self.device).T.unsqueeze(-1)
        else:
            past_targets_nan_mask = torch.empty(self.num_nodes, self.seq_len, 0, device=self.device)

        temporal_features = self.temporal_features[past_timestamps].to(self.device).transpose(0, 1)\
            .expand(self.num_nodes, -1, -1)
        spatiotemporal_features = self.spatiotemporal_features[past_timestamps].to(self.device).transpose(0, 1)
        spatial_features = self.spatial_features.to(self.device).squeeze(0).unsqueeze(1).expand(-1, self.seq_len, -1)
        deepwalk_embeddings = self.deepwalk_embeddings.to(self.device).squeeze(0).unsqueeze(1).expand(-1, self.seq_len, -1)

        features = torch.cat([past_targets, past_targets_nan_mask, temporal_features, spatial_features,
                              spatiotemporal_features, deepwalk_embeddings], axis=2)

        return features

    def get_timestamp_features(self, timestamp):
        if self.provide_sequence_inputs:
            return self.get_timestamp_features_as_sequence_input(timestamp)
        else:
            return self.get_timestamp_features_as_single_input(timestamp)

    def get_timestamp_targets(self, timestamp):
        future_timestamps = timestamp + self.future_timestamp_shifts_for_prediction
        targets = self.targets_for_loss[future_timestamps].to(self.device).T.squeeze(1)
        targets_nan_mask = self.targets_nan_mask[future_timestamps].to(self.device).T.squeeze(1)

        return targets, targets_nan_mask

    def get_timestamp_features_and_targets(self, timestamp):
        features = self.get_timestamp_features(timestamp)
        targets, targets_nan_mask = self.get_timestamp_targets(timestamp)

        return features, targets, targets_nan_mask

    def get_timestamps_batch_features_as_single_input(self, timestamps_batch):
        batch_size = len(timestamps_batch)

        # The shape of past_timestamps is [batch_size, past_targets_features_dim].
        past_timestamps = timestamps_batch[:, None] + self.past_timestamp_shifts_for_features[None, :]
        negative_mask = (past_timestamps < 0)
        past_timestamps[negative_mask] = 0

        # The shape of past targets and past_targets_nan_mask changes from
        # [batch_size, past_targets_features_dim, num_nodes] to
        # [batch_size, num_nodes, past_targets_features_dim] to
        # [num_nodes * batch_size, past_targets_features_dim].
        past_targets = self.targets_for_features[past_timestamps].to(self.device).transpose(1, 2)\
            .flatten(start_dim=0, end_dim=1)

        if self.add_indicators_of_nan_targets_to_features:
            past_targets_nan_mask = self.targets_nan_mask[past_timestamps].to(self.device)
            past_targets_nan_mask[negative_mask] = 1
            past_targets_nan_mask = past_targets_nan_mask.transpose(1, 2).flatten(start_dim=0, end_dim=1)
        else:
            past_targets_nan_mask = torch.empty(self.num_nodes * batch_size, 0, device=self.device)

        temporal_features = self.temporal_features[timestamps_batch].to(self.device).squeeze(1)\
            .repeat_interleave(repeats=self.num_nodes, dim=0)
        spatiotemporal_features = self.spatiotemporal_features[timestamps_batch].to(self.device)\
            .flatten(start_dim=0, end_dim=1)

        if batch_size == self.train_batch_size:
            spatial_features = self.spatial_features_batched_train.squeeze(0)
            deepwalk_embeddings = self.deepwalk_embeddings_batched_train.squeeze(0)
        elif batch_size == self.eval_batch_size:
            spatial_features = self.spatial_features_batched_eval.squeeze(0)
            deepwalk_embeddings = self.deepwalk_embeddings_batched_eval.squeeze(0)
        else:
            spatial_features = self.spatial_features.to(self.device).squeeze(0).repeat(batch_size, 1)
            deepwalk_embeddings = self.deepwalk_embeddings.to(self.device).squeeze(0).repeat(batch_size, 1)

        features = torch.cat([past_targets, past_targets_nan_mask, temporal_features, spatial_features,
                              spatiotemporal_features, deepwalk_embeddings], axis=1)

        return features

    def get_timestamps_batch_features_as_sequence_input(self, timestamps_batch):
        batch_size = len(timestamps_batch)

        # The shape of past_timestamps is [batch_size, seq_len].
        past_timestamps = timestamps_batch[:, None] + self.past_timestamp_shifts_for_features[None, :]

        # The shape of past targets and past_targets_nan_mask changes from
        # [batch_size, seq_len, num_nodes] to
        # [batch_size, num_nodes, seq_len] to
        # [num_nodes * batch_size, seq_len] to
        # [num_nodes * batch_size, seq_len, 1].
        past_targets = self.targets_for_features[past_timestamps].to(self.device).transpose(1, 2)\
            .flatten(start_dim=0, end_dim=1).unsqueeze(-1)

        if self.add_indicators_of_nan_targets_to_features:
            past_targets_nan_mask = self.targets_nan_mask[past_timestamps].to(self.device).transpose(1, 2)\
                .flatten(start_dim=0, end_dim=1).unsqueeze(-1)
        else:
            past_targets_nan_mask = torch.empty(self.num_nodes * batch_size, self.seq_len, 0, device=self.device)

        temporal_features = self.temporal_features[past_timestamps].to(self.device).squeeze(2)\
            .repeat_interleave(repeats=self.num_nodes, dim=0)
        spatiotemporal_features = self.spatiotemporal_features[past_timestamps].to(self.device).transpose(1, 2)\
            .flatten(start_dim=0, end_dim=1)

        if batch_size == self.train_batch_size:
            spatial_features = self.spatial_features_batched_train
            deepwalk_embeddings = self.deepwalk_embeddings_batched_train
        elif batch_size == self.eval_batch_size:
            spatial_features = self.spatial_features_batched_eval
            deepwalk_embeddings = self.deepwalk_embeddings_batched_eval
        else:
            spatial_features = self.spatial_features.to(self.device).repeat(1, batch_size, 1)
            deepwalk_embeddings = self.deepwalk_embeddings.to(self.device).repeat(1, batch_size, 1)

        spatial_features = spatial_features.squeeze(0).unsqueeze(1).expand(-1, self.seq_len, -1)
        deepwalk_embeddings = deepwalk_embeddings.squeeze(0).unsqueeze(1).expand(-1, self.seq_len, -1)

        features = torch.cat([past_targets, past_targets_nan_mask, temporal_features, spatial_features,
                              spatiotemporal_features, deepwalk_embeddings], axis=2)

        return features

    def get_timestamps_batch_features(self, timestamps_batch):
        if self.provide_sequence_inputs:
            return self.get_timestamps_batch_features_as_sequence_input(timestamps_batch)
        else:
            return self.get_timestamps_batch_features_as_single_input(timestamps_batch)

    def get_timestamps_batch_targets(self, timestamps_batch):
        # The shape of future_timestamps is [batch_size, targets_dim].
        future_timestamps = timestamps_batch[:, None] + self.future_timestamp_shifts_for_prediction[None, :]
        # The shape of targets and targets_nan_mask changes from
        # [batch_size, targets_dim, num_nodes] to
        # [batch_size, num_nodes, targets_dim] to
        # [num_nodes * batch_size, targets_dim],
        # and if targets_dim is 1, then the last dimension is squeezed.
        targets = self.targets_for_loss[future_timestamps].to(self.device).transpose(1, 2)\
            .flatten(start_dim=0, end_dim=1).squeeze(1)
        targets_nan_mask = self.targets_nan_mask[future_timestamps].to(self.device).transpose(1, 2)\
            .flatten(start_dim=0, end_dim=1).squeeze(1)

        return targets, targets_nan_mask

    def get_timestamps_batch_features_and_targets(self, timestamps_batch):
        features = self.get_timestamps_batch_features(timestamps_batch)
        targets, targets_nan_mask = self.get_timestamps_batch_targets(timestamps_batch)

        return features, targets, targets_nan_mask

    def transform_preds_for_metrics(self, preds):
        device = preds.device

        if self.transform_targets_for_loss_for_each_node_separately:
            if self.targets_dim == 1:
                preds = preds.cpu().numpy()
                preds_orig = self.targets_for_loss_transform.inverse_transform(preds)
                preds_orig = torch.tensor(preds_orig, device=device)
            else:
                preds = preds.transpose(1, 2)
                shape = preds.shape
                preds = preds.reshape(-1, self.num_nodes)
                preds = preds.cpu().numpy()
                preds_orig = self.targets_for_loss_transform.inverse_transform(preds)
                preds_orig = torch.tensor(preds_orig, device=device)
                preds_orig = preds_orig.reshape(*shape)
                preds_orig = preds_orig.transpose(1, 2)
        else:
            shape = preds.shape
            preds = preds.reshape(-1, 1)
            preds = preds.cpu().numpy()
            preds_orig = self.targets_for_loss_transform.inverse_transform(preds)
            preds_orig = torch.tensor(preds_orig, device=device)
            preds_orig = preds_orig.reshape(*shape)

        return preds_orig

    def get_val_targets_for_metrics(self):
        targets = self.targets_for_metrics[self.all_val_targets_timestamps]
        targets_nan_mask = self.targets_nan_mask[self.all_val_targets_timestamps]

        targets, targets_nan_mask = self.prepare_targets_for_evaluation(targets, targets_nan_mask)

        return targets.to(self.device), targets_nan_mask.to(self.device)

    def get_test_targets_for_metrics(self):
        targets = self.targets_for_metrics[self.all_test_targets_timestamps]
        targets_nan_mask = self.targets_nan_mask[self.all_test_targets_timestamps]

        targets, targets_nan_mask = self.prepare_targets_for_evaluation(targets, targets_nan_mask)

        return targets.to(self.device), targets_nan_mask.to(self.device)

    def prepare_targets_for_evaluation(self, targets, targets_nan_mask):
        if self.targets_dim == 1:
            # If we only predict target for a single timestamp (which is the case if only_predict_at_end_of_horizon
            # is True or if prediction_horizon is 1), we need to shift all targets by the number of timestamps between
            # the observed timestamp and the timestamp of prediction.
            future_timestamp_shift = self.future_timestamp_shifts_for_prediction[0]
            targets = targets[future_timestamp_shift - 1:]
            targets_nan_mask = targets_nan_mask[future_timestamp_shift - 1:]
        else:
            # If we predict targets for multiple timestamps, we need to go over the targets with a sliding window
            # of length num_predictions and create a tensor of shape [num_timestamps, num_nodes, num_predictions].
            targets = targets.unfold(dimension=0, size=self.targets_dim, step=1)
            targets_nan_mask = targets_nan_mask.unfold(dimension=0, size=self.targets_dim, step=1)

        return targets, targets_nan_mask
