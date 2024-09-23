from pathlib import Path
from typing import List, Dict, Tuple
import sys
from ast import literal_eval
from itertools import product

try:
    import nirvana_dl as ndl
    PARAMS = ndl.params()
    print("Imported Nirvana DL package")
except ImportError:
    ndl = None
    PARAMS = {
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
        "train_batch_size": [1, 2, 3, 4, 5],
        "eval_batch_size": None,
        "num_accumulation_steps": 1,
        "eval_every": 1000,
        "num_runs": 2,
        "device": "cuda:0",
        "amp": False,
        "num_threads": [32, 123123123123123123],
    }

    for key in PARAMS:
        PARAMS[key] = str(PARAMS[key])

STORE_TRUE_ARGS = {
    "transform_targets_for_loss_for_each_node_separately",
    "seq_encoder_bidir_attn",
    "plr_past_targets_shared_linear",
    "do_not_use_spatiotemporal_features",
    "reverse_edges",
    "add_indicators_of_nan_targets_to_features",
    "plr_num_features_shared_linear",
    "only_predict_at_end_of_horizon",
    "do_not_use_spatial_features",
    "do_not_separate_ego_node_representation",
    "to_undirected",
    "nirvana",
    "use_forward_and_reverse_edges_as_different_edge_types",
    "initialize_learnable_node_embeddings_with_deepwalk",
    "amp",
    "do_not_use_temporal_features",
    "use_plr_for_past_targets",
    "use_plr_for_num_features",
    "transform_targets_for_features_for_each_node_separately",
    "plr_past_targets_shared_frequencies",
    "use_deepwalk_node_embeddings",
    "use_learnable_node_embeddings",
    "plr_num_features_shared_frequencies",
}


def filter_params_on_single_and_multiple_options():
    params_with_multiple_options_and_values: Dict[str, List[str]] = {}  # those which can be casted to lists of values
    params_with_single_options_and_values: Dict[str, str] = {}  # those which can be casted to lists of values

    def _check_if_iterable(value):
        try:
            s = literal_eval(repr(value))
            if isinstance(s, (list, tuple, set)):
                return True
            return False
        except (ValueError, SyntaxError):
            return isinstance(value, (list, tuple, set))

    for param, value in PARAMS.items():
        if _check_if_iterable(value=value):
            params_with_multiple_options_and_values[param] = [str(x) for x in literal_eval(repr(value))]
        else:
            params_with_single_options_and_values[param] = value

    return params_with_single_options_and_values, params_with_multiple_options_and_values


def create_one_run(params_flattened_one_instance: Dict[str, str]):
    launch_script_string_container: List[List[str]] = ["python", "run_single_experiment.py"]

    for option_name, option_value in params_flattened_one_instance.items():
        print(f"Option: {repr(option_name)}, param: {repr(option_value)}", file=sys.stderr)
        if option_value == "None" or option_value is None:
            continue

        if option_name in STORE_TRUE_ARGS:  # this option value is true and it;s passed as true
            if option_value == "True":
                param_string: str = f"--{option_name}"
            else:
                continue
        else:
            param_string = f"--{option_name} {option_value}"
        launch_script_string_container.append(param_string)

    launch_script_string: str = " ".join(launch_script_string_container)

    return launch_script_string


if __name__ == "__main__":
    single_choice_params, multi_choice_params = filter_params_on_single_and_multiple_options()

    if len(multi_choice_params) > 0:
        multi_choice_containers_per_param: List[List[Tuple[str, str]]] = []

        for param, values in multi_choice_params.items():
            multi_choice_containers_per_param.append([(param, v) for v in values])

        print(f"{multi_choice_containers_per_param=}", file=sys.stderr)
        multi_choice_params_product = product(*multi_choice_containers_per_param)

        launch_strings: List[str] = []

        for params_choice in multi_choice_params_product:
            print(f"{params_choice}", file=sys.stderr)
            parameters = {p: v for p, v in params_choice}
            parameters.update(single_choice_params)

            one_run_string = create_one_run(parameters)

            launch_strings.append(one_run_string)

        print("\n\n".join(launch_strings))
    else:
        print(create_one_run(single_choice_params))
