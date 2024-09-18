from pathlib import Path
from typing import List
import sys


try:
    import nirvana_dl as ndl
    params = ndl.params()

except ImportError:
    ndl = None
    params = {
        "name": "nirvana_local_test",
        "save_dir": "experiments",
        "dataset": "pems-bay",
        "metric": "RMSE",
        "prediction_horizon": 12,
        "only_predict_at_end_of_horizon": False,
        "direct_lookback_num_steps": 48,
        "seasonal_lookback_periods": None,
        "seasonal_lookback_num_steps": None,
        "drop_early_train_timestamps": "direct",
        "reverse_edges": False,
        "to_undirected": False,
        "use_forward_and_reverse_edges_as_different_edge_types": False,
        "target_transform": "standard-scaler",
        "transform_targets_for_each_node_separately": False,
        "imputation_startegy_for_nan_targets": "prev",
        "add_features_for_nan_targets": False,
        "do_not_use_temporal_features": False,
        "do_not_use_spatial_features": False,
        "do_not_use_spatiotemporal_features": False,
        "use_deepwalk_node_embeddings": False,
        "use_learnable_node_embeddings": False,
        "learnable_node_embeddings_dim": 128,
        "initialize_learnable_node_embeddings_with_deepwalk": False,
        "imputation_strategy_for_numerical_features": "most_frequent",
        "numerical_features_transform": "quantile-transform-normal",
        "use_plr_for_num_features": False,
        "plr_num_features_frequencies_dim": 48,
        "plr_num_features_frequencies_scale": 0.01,
        "plr_num_features_embedding_dim": 16,
        "plr_num_features_shared_linear": False,
        "plr_num_features_shared_frequencies": False,
        "use_plr_for_past_targets": False,
        "plr_past_targets_frequencies_dim": 48,
        "plr_past_targets_frequencies_scale": 0.01,
        "plr_past_targets_embedding_dim": 16,
        "plr_past_targets_shared_linear": False,
        "plr_past_targets_shared_frequencies": False,
        "model_class": "SingleInputGNN",
        "neighborhood_aggregation": "MeanAggr",
        "do_not_separate_ego_node_representation": False,
        "sequence_encoder": "RNN",
        "normalization": "LayerNorm",
        "num_residual_blocks": 2,
        "hidden_dim": 512,
        "neighborhood_aggr_attn_num_heads": 4,
        "seq_encoder_num_layers": 4,
        "seq_encoder_rnn_type": "LSTM",
        "seq_encoder_attn_num_heads": 8,
        "seq_encoder_bidir_attn": False,
        "dropout": 0,
        "weight_decay": 0,
        "lr": 0.0003,
        "num_epochs": 10,
        "train_batch_size": 10,
        "eval_batch_size": None,
        "num_accumulation_steps": 1,
        "eval_every": 1000,
        "num_runs": 2,
        "device": "cuda:0",
        "amp": False,
        "num_threads": 32,
    }
    
    for key in params:
        params[key] = str(params[key])

STORE_TRUE_ARGS = {
    "use_deepwalk_node_embeddings",
    "only_predict_at_end_of_horizon",
    "seq_encoder_bidir_attn",
    "amp",
    "reverse_edges",
    "to_undirected",
    "plr_num_features_shared_linear",
    "use_learnable_node_embeddings",
    "do_not_use_temporal_features",
    "add_features_for_nan_targets",
    "plr_num_features_shared_frequencies",
    "do_not_use_spatiotemporal_features",
    "plr_past_targets_shared_linear",
    "transform_targets_for_each_node_separately",
    "initialize_learnable_node_embeddings_with_deepwalk",
    "do_not_separate_ego_node_representation",
    "use_forward_and_reverse_edges_as_different_edge_types",
    "plr_past_targets_shared_frequencies",
    "use_plr_for_past_targets",
    "do_not_use_spatial_features",
    "use_plr_for_num_features",
}


launch_script_string_container: List[List[str]] = ["python", "run_single_experiment.py"]

for option_name, option_value in params.items():
    print(f"Option: {repr(option_name)}, param: {repr(option_value)}", file=sys.stderr)
    if (option_value == "None" or  option_value is None):
        continue
    
    if option_name in STORE_TRUE_ARGS: # this option value is true and it;s passed as true
        if option_value == "True":
            param_string: str = f"--{option_name}"
    else:
        param_string = f"--{option_name} {option_value}"
    launch_script_string_container.append(param_string)

launch_script_string: str = " ".join(launch_script_string_container)
print(launch_script_string)