import argparse
import json

parser = argparse.ArgumentParser()

parser.add_argument('--name', type=str, help='Experiment name.', default="Name")
parser.add_argument('--save_dir', type=str, default='experiments', help='Base directory for saving information.')
parser.add_argument('--dataset', type=str, default='pems-bay',
                    help='Dataset name (for an existing dataset in the data directory) or a path to a .npz file '
                            'with data. Possible dataset names: metr-la, pems-bay, largest, largest-2019.')
parser.add_argument('--metric', type=str, default='RMSE', choices=['RMSE', 'MAE'])

# Select future timestamps targets from which will be predicted by the model.
parser.add_argument('--prediction_horizon', type=int, default=12)
parser.add_argument('--only_predict_at_end_of_horizon', default=False, action='store_true')

# Select past timestamps targets from which will be used as node features and passed as input to the model.
parser.add_argument('--direct_lookback_num_steps', type=int, default=48)
parser.add_argument('--seasonal_lookback_periods', nargs='+', type=int, default=None,
                    help='Should have the same number of values as seasonal_lookback_num_steps argument.')
parser.add_argument('--seasonal_lookback_num_steps', nargs='+', type=int, default=None,
                    help='Should have the same number of values as seasonal_lookback_periods argument.')
parser.add_argument('--drop_early_train_timestamps', type=str, default='direct', choices=['all', 'direct', 'none'])

# Only for directed graphs: select which edge directions in the graph will be used.
# Use at most one of these three arguments.
parser.add_argument('--reverse_edges', default=False, action='store_true',
                    help='Reverse all edges in the graph.')
parser.add_argument('--to_undirected', default=False, action='store_true',
                    help='Transform the graph to undirected by converting each directed edge into '
                            'an undirected one.')
parser.add_argument('--use_forward_and_reverse_edges_as_different_edge_types', default=False, action='store_true',
                    help='The graph will be transformed to a heterogeneous graph with two edge types: '
                            'forward (original) and reverse edges. Graph neighborhood aggregation wiil be run for '
                            'each edge type separately and its results will be concatenated before being passed '
                            'to the following MLP module in the model.')

# Target preprocessing.
parser.add_argument('--target_transform', type=str, default='standard-scaler',
                    choices=['none', 'standard-scaler', 'min-max-scaler', 'robust-scaler',
                                'power-transform-yeo-johnson', 'quantile-transform-normal',
                                'quantile-transform-uniform'])
parser.add_argument('--transform_targets_for_each_node_separately', default=False, action='store_true')

# Target imputation.
parser.add_argument('--imputation_startegy_for_nan_targets', type=str, default='prev', choices=['prev', 'zero'])
parser.add_argument('--add_features_for_nan_targets', default=False, action='store_true')

# Drop unwanted node features.
parser.add_argument('--do_not_use_temporal_features', default=False, action='store_true')
parser.add_argument('--do_not_use_spatial_features', default=False, action='store_true')
parser.add_argument('--do_not_use_spatiotemporal_features', default=False, action='store_true')

# Add additional node features.
parser.add_argument('--use_deepwalk_node_embeddings', default=False, action='store_true')
parser.add_argument('--use_learnable_node_embeddings', default=False, action='store_true',
                    help='Not used if model_class is Linear.')
parser.add_argument('--learnable_node_embeddings_dim', type=int, default=128,
                    help='Only used if use_learnable_node_embeddings is True.')
parser.add_argument('--initialize_learnable_node_embeddings_with_deepwalk', default=False, action='store_true',
                    help='Initializes learnable node embeddings with DeepWalk node embeddings. '
                            'This can be used instead of or in addition to using fixed (non-trainable) DeepWalk node '
                            'embeddings (which are controlled by use_deepwalk_node_embeddings arguments). '
                            'Only used if use_learnable_node_embeddings is True.')

# Numerical features preprocessing.
parser.add_argument('--imputation_strategy_for_numerical_features', type=str, default='most_frequent',
                    choices=['mean', 'median', 'most_frequent'],
                    help='Only used for datasets that have NaNs in spatial or spatiotemporal numerical features.')
parser.add_argument('--numerical_features_transform', type=str, default='quantile-transform-normal',
                    choices=['none', 'standard-scaler', 'min-max-scaler', 'robust-scaler',
                                'power-transform-yeo-johnson', 'quantile-transform-normal',
                                'quantile-transform-uniform'])

# PLR embeddings for numerical features. Not used if model_class is Linear.
parser.add_argument('--use_plr_for_num_features', default=False, action='store_true',
                    help='Apply PLR embeddings to numerical features.')
parser.add_argument('--plr_num_features_frequencies_dim', type=int, default=48,
                    help='Only used if plr_num_features is True.')
parser.add_argument('--plr_num_features_frequencies_scale', type=float, default=0.01,
                    help='Only used if plr_num_features is True.')
parser.add_argument('--plr_num_features_embedding_dim', type=int, default=16,
                    help='Only used if plr_num_features is True.')
parser.add_argument('--plr_num_features_shared_linear', default=False, action='store_true',
                    help='Only used if plr_num_features is True.')
parser.add_argument('--plr_num_features_shared_frequencies', default=False, action='store_true',
                    help='Only used if plr_num_features is True.')

# PLR embeddings for past targets. Not used if model_class is Linear.
parser.add_argument('--use_plr_for_past_targets', default=False, action='store_true',
                    help='Apply PLR embeddings to past targets.')
parser.add_argument('--plr_past_targets_frequencies_dim', type=int, default=48,
                    help='Only used if plr_past_targets is True.')
parser.add_argument('--plr_past_targets_frequencies_scale', type=float, default=0.01,
                    help='Only used if plr_past_targets is True.')
parser.add_argument('--plr_past_targets_embedding_dim', type=int, default=16,
                    help='Only used if plr_past_targets is True.')
parser.add_argument('--plr_past_targets_shared_linear', default=False, action='store_true',
                    help='Only used if plr_past_targets is True.')
parser.add_argument('--plr_past_targets_shared_frequencies', default=False, action='store_true',
                    help='Only used if plr_past_targets is True.')

# Model type selection.
parser.add_argument('--model_class', type=str, default='SignleInputGNN',
                    choices=['LinearModel', 'ResNet', 'SingleInputGNN', 'SequenceInputGNN'])
parser.add_argument('--neighborhood_aggregation', type=str, default='MeanAggr',
                    choices=['MeanAggr', 'MaxAggr', 'GCNAggr', 'AttnGATAggr', 'AttnTrfAggr'],
                    help='Graph neighborhood aggregation (aka message passing) function for GNNs. '
                            'Only used if model_class is SingleInputGNN or SequenceInputGNN.')
parser.add_argument('--do_not_separate_ego_node_representation', default=False, action='store_true',
                    help='Use ego node representation in graph neighborhood aggregation as if it is one more '
                            'neighbor representation instead of treating it separately by concatenating it to '
                            'aggregated neighbor representations.')
parser.add_argument('--sequence_encoder', type=str, default='RNN',
                    choices=['RNN', 'Transformer'],
                    help='Timestamp sequence encoder applied before graph neighborhood aggregation. '
                            'Only used if model_class is SequenceInputGNN.')
parser.add_argument('--normalization', type=str, default='LayerNorm',
                    choices=['none', 'LayerNorm', 'BatchNorm'],
                    help='Normalization applied in the beginning of each residual block. '
                            'Not used if model_class is LinearModel.')

# Model architecture hyperparameters.
parser.add_argument('--num_residual_blocks', type=int, default=2,
                    help='Number of residual blocks, where each residual block consists of the following sequence '
                            'of layers: normalization, sequence encoder (if model_class is SequenceInputGNN), '
                            'graph neighborhood aggregation (if model_class is SingleInputGNN or SequenceInputGNN), '
                            'two-layer MLP. '
                            'Not used if model_class is LinearModel.')
parser.add_argument('--hidden_dim', type=int, default=512,
                    help='Not used if model_class is LinearModel.')
parser.add_argument('--neighborhood_aggr_attn_num_heads', type=int, default=4,
                    help='Number of attention heads for attention-based graph neighborhood aggregation. '
                            'Only used if model_class is SingleInputGNN or SequenceInputGNN and '
                            'neighborhood_aggregation is AttnGAT or AttnTrf.')
parser.add_argument('--seq_encoder_num_layers', type=int, default=4,
                    help='Number of layers in sequence encoder used in each residual block of the model. '
                            'Only used if model_class is SequenceInputGNN.')
parser.add_argument('--seq_encoder_rnn_type', type=str, default='LSTM', choices=['LSTM', 'GRU'],
                    help='RNN type used as sequence encoder. '
                            'Only used if model_class is SequenceInputGNN and sequence_encoder is RNN.')
parser.add_argument('--seq_encoder_attn_num_heads', type=int, default=8,
                    help='Number of attention heads for attention-based sequence encoders. '
                            'Only used if model_class is SequenceInputGNN and sequence_encoder is Transformer.')
parser.add_argument('--seq_encoder_bidir_attn', default=False, action='store_true',
                    help='Use bidirectional attention instead of unidirectional (aka causal) attention '
                            'in sequence encoder. '
                            'Only used if model_class is SequenceInputGNN and sequence_encoder is Transformer.')

# Regularization.
parser.add_argument('--dropout', type=float, default=0, help='Not used if model_class is LinearModel.')
parser.add_argument('--weight_decay', type=float, default=0)

# Training parameters.
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--num_epochs', type=int, default=10)
parser.add_argument('--train_batch_size', type=int, default=10,
                    help='Effective batch size for each optimization step equals '
                            'train_batch_size * num_accumulation_steps.')
parser.add_argument('--eval_batch_size', type=int, default=None,
                    help='If None, it is set to be the same as train_batch_size. But since evaluation requires '
                            'less VRAM han training, larger batch size can be used.')
parser.add_argument('--num_accumulation_steps', type=int, default=1)
parser.add_argument('--eval_every', type=int, default=1000,
                    help='Evaluate after this many optimization steps. If None, only evaluate at the end of epoch.')

parser.add_argument('--num_runs', type=int, default=5)
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--amp', default=False, action='store_true')
parser.add_argument('--num_threads', type=int, default=32)

arguments = parser.parse_args()


arguments_dict = vars(arguments)
print(f"{arguments_dict=}")


# store_true_args = []

# for action in parser._actions:
#     if isinstance(action, argparse._StoreTrueAction):
#         store_true_args.append(action.option_strings[0][2:])

# print(f"Arguments with action='store_true' ({len(store_true_args)}):")
# for argument in store_true_args:
#     print(argument)
# print(set(store_true_args))



with open("config.txt", "w") as f_write:
    print('{', file=f_write)
    
    for argument, default_value in arguments_dict.items():
        parameter_string = f"    \"{argument}\": ${{global.{argument}!\"{default_value}\"}},"
        print(parameter_string, file=f_write)
    
    print('}', file=f_write)
