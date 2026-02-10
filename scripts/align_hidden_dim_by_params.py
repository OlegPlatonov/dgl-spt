#!/usr/bin/env python3
"""
Скрипт выравнивает модели по числу параметров:
- hidden_dim для SingleInputGNN фиксируется (SINGLE_INPUT_GNN_HIDDEN_DIM).
- Для каждого бейзлайна подбирается hidden_dim так, чтобы число параметров
  было как у SingleInputGNN (или максимально близко).

Запуск из корня репозитория:
  python scripts/align_hidden_dim_by_params.py --dataset <имя_или_путь_к_.npz>

Если --dataset не указан, используются прикидочные значения (features_dim=100 и т.д.)
и выводится предупреждение. Для выравнивания под реальные эксперименты лучше
указывать тот же датасет, на котором будут запускаться эксперименты.
"""

import argparse
import sys
from pathlib import Path

import torch
import dgl

# Добавляем корень репозитория в path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from models import ModelRegistry

# Фиксированный hidden_dim для SingleInputGNN (референс по числу параметров)
SINGLE_INPUT_GNN_HIDDEN_DIM = 256

# Fallback: прикидочные значения, если датасет не загружен (только для оценки)
COMMON_ARGS = {
    "targets_dim": 12,
    "num_edge_types": 1,
    "train_batch_size": 1,
    "direct_lookback_num_steps": 48,
    "num_spatiotemporal_blocks": 2,
    "num_temporal_blocks": 2,
    "num_spatial_blocks": 2,
    "num_residual_blocks": 2,
    "temporal_kernel_size": 2,
    "temporal_dilation": 2,
    "spatial_kernel_size": 2,
    "neighborhood_aggregation": "AttnTrfAggr",
    "do_not_separate_ego_node_representation": False,
    "sequence_encoder": "RNN",
    "normalization": "LayerNorm",
    "neighborhood_aggr_attn_num_heads": 4,
    "seq_encoder_num_layers": 4,
    "seq_encoder_rnn_type": "LSTM",
    "seq_encoder_attn_num_heads": 8,
    "seq_encoder_bidir_attn": False,
    "dropout": 0.0,
    "use_learnable_node_embeddings": True,
    "learnable_node_embeddings_dim": 128,
    "initialize_learnable_node_embeddings_with_deepwalk": False,
    "use_plr_for_numerical_features": False,
    "plr_numerical_features_frequencies_dim": 8,
    "plr_numerical_features_frequencies_scale": 1.0,
    "plr_numerical_features_embedding_dim": 8,
    "plr_numerical_features_shared_linear": False,
    "plr_numerical_features_shared_frequencies": False,
    "use_plr_for_past_targets": False,
    "plr_past_targets_frequencies_dim": 8,
    "plr_past_targets_frequencies_scale": 1.0,
    "plr_past_targets_embedding_dim": 8,
    "plr_past_targets_shared_linear": False,
    "plr_past_targets_shared_frequencies": False,
}


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def make_dummy_dgl_graph(num_nodes: int, num_edges: int = 200):
    """Минимальный DGL-граф для инициализации SingleInputGNN (число параметров от структуры не зависит)."""
    src = torch.randint(0, num_nodes, (num_edges,))
    dst = torch.randint(0, num_nodes, (num_edges,))
    g = dgl.graph((src, dst), num_nodes=num_nodes)
    return g


def make_dummy_edge_index(num_nodes: int):
    """Минимальный edge_index [2, E] для бейзлайнов (кольцо 0->1->...->(N-1)->0)."""
    a = torch.arange(num_nodes, dtype=torch.long)
    b = torch.roll(a, 1)
    return torch.stack([a, b], dim=0)


def load_dataset_context(dataset_name_or_path: str):
    """
    Загружает датасет и возвращает контекст для подсчёта параметров:
    features_dim, num_nodes, targets_dim, train_batch_size, edge_index_batched,
    numerical_features_mask, past_targets_mask, deepwalk_embeddings (для инициализации).
    """
    from utils import DummyHandler
    from dataset import Dataset

    checkpoint_dir = REPO_ROOT / "experiments" / "align_hidden_dim_checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    state_handler = DummyHandler(
        checkpoint_file_path=checkpoint_dir / "state.pt",
        checkpoint_dir=checkpoint_dir,
        checkpoint_steps_interval=1000,
    )
    args = COMMON_ARGS.copy()
    dataset = Dataset(
        name_or_path=dataset_name_or_path,
        state_handler=state_handler,
        prediction_horizon=12,
        only_predict_at_end_of_horizon=False,
        provide_sequnce_inputs=True,  # для бейзлайнов (sequence input)
        direct_lookback_num_steps=args["direct_lookback_num_steps"],
        drop_early_train_timestamps="direct",
        reverse_edges=False,
        to_undirected=False,
        use_forward_and_reverse_edges_as_different_edge_types=False,
        add_self_loops=False,
        targets_for_loss_transform="none",
        targets_for_features_transform="none",
        targets_for_features_nan_imputation_strategy="prev",
        add_nan_indicators_to_targets_for_features=False,
        do_not_use_temporal_features=False,
        do_not_use_spatial_features=False,
        do_not_use_spatiotemporal_features=False,
        time_based_features_types=("one-hot", "sin-cos"),
        time_based_features_periods=("auto",),
        use_deepwalk_node_embeddings=False,
        initialize_learnable_node_embeddings_with_deepwalk=False,
        numerical_features_transform="none",
        numerical_features_nan_imputation_strategy="most_frequent",
        train_batch_size=args["train_batch_size"],
        eval_batch_size=None,
        eval_max_num_predictions_per_step=1_000_000_000,
        device="cpu",
        nirvana=False,
        spatiotemporal_features_local_processed_memmap_name=None,
        disable_features_checkpointing=True,
        use_edge_index=True,  # для бейзлайнов нужен edge_index
    )
    # При use_edge_index=True train_batched_graph уже tensor [2, E]
    return {
        "features_dim": dataset.features_dim,
        "num_nodes": dataset.num_nodes,
        "targets_dim": dataset.targets_dim,
        "train_batch_size": dataset.train_batch_size,
        "edge_index_batched": dataset.train_batched_graph,
        "numerical_features_mask": dataset.numerical_features_mask,
        "past_targets_mask": dataset.past_targets_mask,
        "deepwalk_embeddings": dataset.deepwalk_embeddings_for_initializing_learnable_embeddings,
    }


def get_single_input_gnn_param_count(
    hidden_dim: int = SINGLE_INPUT_GNN_HIDDEN_DIM,
    context: dict | None = None,
) -> int:
    args = COMMON_ARGS.copy()
    if context is not None:
        features_dim = context["features_dim"]
        targets_dim = context["targets_dim"]
        num_nodes = context["num_nodes"]
        numerical_mask = context["numerical_features_mask"]
        past_mask = context["past_targets_mask"]
    else:
        features_dim = 100  # fallback
        targets_dim = args.get("targets_dim", 12)
        num_nodes = 100
        numerical_mask = torch.zeros(features_dim, dtype=torch.bool)
        past_mask = torch.zeros(features_dim, dtype=torch.bool)

    graph = make_dummy_dgl_graph(num_nodes)
    if graph.num_edges() == 0:
        graph = dgl.add_self_loop(graph)

    Model = ModelRegistry.get_model_class("SingleInputGNN")
    model = Model(
        neighborhood_aggregation_name=args["neighborhood_aggregation"],
        neighborhood_aggregation_sep=not args["do_not_separate_ego_node_representation"],
        normalization_name=args["normalization"],
        num_edge_types=args["num_edge_types"],
        num_residual_blocks=args["num_residual_blocks"],
        features_dim=features_dim,
        hidden_dim=hidden_dim,
        output_dim=targets_dim,
        neighborhood_aggr_attn_num_heads=args["neighborhood_aggr_attn_num_heads"],
        dropout=args["dropout"],
        use_learnable_node_embeddings=args["use_learnable_node_embeddings"],
        num_nodes=num_nodes,
        learnable_node_embeddings_dim=args["learnable_node_embeddings_dim"],
        initialize_learnable_node_embeddings_with_deepwalk=args["initialize_learnable_node_embeddings_with_deepwalk"],
        deepwalk_node_embeddings=(context.get("deepwalk_embeddings") if context else None),
        use_plr_for_numerical_features=args["use_plr_for_numerical_features"],
        numerical_features_mask=numerical_mask,
        plr_numerical_features_frequencies_dim=args["plr_numerical_features_frequencies_dim"],
        plr_numerical_features_frequencies_scale=args["plr_numerical_features_frequencies_scale"],
        plr_numerical_features_embedding_dim=args["plr_numerical_features_embedding_dim"],
        plr_numerical_features_shared_linear=args["plr_numerical_features_shared_linear"],
        plr_numerical_features_shared_frequencies=args["plr_numerical_features_shared_frequencies"],
        use_plr_for_past_targets=args["use_plr_for_past_targets"],
        past_targets_mask=past_mask,
        plr_past_targets_frequencies_dim=args["plr_past_targets_frequencies_dim"],
        plr_past_targets_frequencies_scale=args["plr_past_targets_frequencies_scale"],
        plr_past_targets_embedding_dim=args["plr_past_targets_embedding_dim"],
        plr_past_targets_shared_linear=args["plr_past_targets_shared_linear"],
        plr_past_targets_shared_frequencies=args["plr_past_targets_shared_frequencies"],
    )
    return count_parameters(model)


def get_baseline_param_count(
    baseline_name: str,
    hidden_dim: int,
    context: dict | None = None,
) -> int:
    args = COMMON_ARGS.copy()
    if context is not None:
        features_dim = context["features_dim"]
        targets_dim = context["targets_dim"]
        num_nodes = context["num_nodes"]
        batch_size = context["train_batch_size"]
        edge_index = context["edge_index_batched"]
        numerical_mask = context["numerical_features_mask"]
        past_mask = context["past_targets_mask"]
    else:
        features_dim = 100
        targets_dim = args.get("targets_dim", 12)
        num_nodes = 100
        batch_size = args["train_batch_size"]
        numerical_mask = torch.zeros(features_dim, dtype=torch.bool)
        past_mask = torch.zeros(features_dim, dtype=torch.bool)
        edge_index = make_dummy_edge_index(num_nodes * batch_size)

    Model = ModelRegistry.get_model_class("BaselineModel")
    try:
        model = Model(
            baseline_name=baseline_name,
            normalization_name=args["normalization"],
            num_spatiotemporal_blocks=args["num_spatiotemporal_blocks"],
            num_temporal_blocks=args["num_temporal_blocks"],
            num_spatial_blocks=args["num_spatial_blocks"],
            features_dim=features_dim,
            hidden_dim=hidden_dim,
            output_dim=targets_dim,
            use_learnable_node_embeddings=args["use_learnable_node_embeddings"],
            num_nodes=num_nodes,
            batch_size=batch_size,
            edge_index_batched=edge_index,
            learnable_node_embeddings_dim=args["learnable_node_embeddings_dim"],
            initialize_learnable_node_embeddings_with_deepwalk=args["initialize_learnable_node_embeddings_with_deepwalk"],
            deepwalk_node_embeddings=(context.get("deepwalk_embeddings") if context else None),
            use_plr_for_numerical_features=args["use_plr_for_numerical_features"],
            numerical_features_mask=numerical_mask,
            plr_numerical_features_frequencies_dim=args["plr_numerical_features_frequencies_dim"],
            plr_numerical_features_frequencies_scale=args["plr_numerical_features_frequencies_scale"],
            plr_numerical_features_embedding_dim=args["plr_numerical_features_embedding_dim"],
            plr_numerical_features_shared_linear=args["plr_numerical_features_shared_linear"],
            plr_numerical_features_shared_frequencies=args["plr_numerical_features_shared_frequencies"],
            use_plr_for_past_targets=args["use_plr_for_past_targets"],
            past_targets_mask=past_mask,
            plr_past_targets_frequencies_dim=args["plr_past_targets_frequencies_dim"],
            plr_past_targets_frequencies_scale=args["plr_past_targets_frequencies_scale"],
            plr_past_targets_embedding_dim=args["plr_past_targets_embedding_dim"],
            plr_past_targets_shared_linear=args["plr_past_targets_shared_linear"],
            plr_past_targets_shared_frequencies=args["plr_past_targets_shared_frequencies"],
            temporal_kernel_size=args["temporal_kernel_size"],
            temporal_dilation=args["temporal_dilation"],
            spatial_kernel_size=args["spatial_kernel_size"],
            seq_encoder_seq_len=args["direct_lookback_num_steps"],
            dropout=args["dropout"],
        )
        return count_parameters(model)
    except Exception as e:
        print(f"  [skip {baseline_name} hidden_dim={hidden_dim}] {e}")
        return -1


def find_hidden_dim_for_target_params(
    baseline_name: str,
    target_params: int,
    context: dict | None = None,
    search_min: int = 32,
    search_max: int = 1024,
    step: int = 16,
) -> tuple[int, int]:
    """Подбирает hidden_dim для бейзлайна, чтобы число параметров было ближе всего к target_params."""
    best_hidden = search_min
    best_count = get_baseline_param_count(baseline_name, search_min, context=context)
    if best_count < 0:
        return search_min, -1

    best_diff = abs(best_count - target_params)
    for h in range(search_min + step, search_max + 1, step):
        cnt = get_baseline_param_count(baseline_name, h, context=context)
        if cnt < 0:
            continue
        diff = abs(cnt - target_params)
        if diff < best_diff:
            best_diff = diff
            best_hidden = h
            best_count = cnt
    return best_hidden, best_count


def main():
    parser = argparse.ArgumentParser(
        description="Выравнивание hidden_dim по числу параметров (референс: SingleInputGNN)."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Имя датасета (из data/) или путь к .npz. Если не указан, используются прикидочные значения.",
    )
    parser.add_argument(
        "--single-input-gnn-hidden-dim",
        type=int,
        default=SINGLE_INPUT_GNN_HIDDEN_DIM,
        dest="single_input_gnn_hidden_dim",
        help=f"Фиксированный hidden_dim для SingleInputGNN (по умолчанию {SINGLE_INPUT_GNN_HIDDEN_DIM}).",
    )
    args_cli = parser.parse_args()

    context = None
    if args_cli.dataset:
        print(f"Загрузка датасета: {args_cli.dataset}")
        try:
            context = load_dataset_context(args_cli.dataset)
            print(
                f"  features_dim={context['features_dim']}, num_nodes={context['num_nodes']}, "
                f"targets_dim={context['targets_dim']}, train_batch_size={context['train_batch_size']}"
            )
        except Exception as e:
            print(f"  Ошибка загрузки датасета: {e}")
            print("  Используются прикидочные значения.")
    else:
        print("Датасет не указан (--dataset). Используются прикидочные значения (features_dim=100, num_nodes=100).")
        print("Для выравнивания под реальные эксперименты укажите тот же датасет, что и в run_single_experiment.")

    hidden_dim_ref = args_cli.single_input_gnn_hidden_dim
    print()
    print("=" * 60)
    print("Выравнивание hidden_dim по числу параметров")
    print("=" * 60)
    print(f"SingleInputGNN: hidden_dim фиксирован = {hidden_dim_ref}")
    print()

    target_params = get_single_input_gnn_param_count(hidden_dim=hidden_dim_ref, context=context)
    print(f"Число параметров SingleInputGNN (target): {target_params:,}")
    print()

    baselines = [
        "DCRNN", "GWN", "GRUGCN", "STGCN"
    ]

    results = {"SingleInputGNN": hidden_dim_ref}
    print("Подбор hidden_dim для бейзлайнов:")
    print("-" * 60)
    for name in baselines:
        hidden, count = find_hidden_dim_for_target_params(name, target_params, context=context)
        results[name] = hidden
        pct = 100.0 * (count - target_params) / target_params if count > 0 and target_params else 0
        status = f"params={count:,} ({pct:+.1f}%)" if count > 0 else "failed"
        print(f"  {name:12s} -> hidden_dim = {hidden:4d}  ({status})")
    print("-" * 60)

    # Сохраняем конфиг для использования в экспериментах
    out_path = REPO_ROOT / "scripts" / "aligned_hidden_dim_config.py"
    dataset_note = f"  # dataset: {args_cli.dataset or 'fallback (no dataset)'}"
    lines = [
        "# Auto-generated by scripts/align_hidden_dim_by_params.py",
        "# Использование: подставлять --hidden_dim <value> в зависимости от --model_class / --baseline_name",
        dataset_note,
        "",
        "SINGLE_INPUT_GNN_HIDDEN_DIM = " + str(hidden_dim_ref),
        "TARGET_PARAM_COUNT = " + str(target_params),
        "",
        "# hidden_dim по модели (для BaselineModel — по baseline_name)",
        "HIDDEN_DIM_BY_MODEL = {",
    ]
    for k, v in results.items():
        lines.append(f'    "{k}": {v},')
    lines.append("}")
    lines.append("")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nКонфиг записан в {out_path}")

    return results


if __name__ == "__main__":
    main()
