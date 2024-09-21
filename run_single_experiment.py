import argparse
from tqdm import tqdm

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from models import ModelRegistry
from datasets import Dataset
from utils import Logger, get_parameter_groups


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', type=str, required=True, help='Experiment name.')
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

    # Targets preprocessing.
    parser.add_argument('--targets_transform', type=str, default='standard-scaler',
                        choices=['none', 'standard-scaler', 'min-max-scaler', 'robust-scaler',
                                 'power-transform-yeo-johnson', 'quantile-transform-normal',
                                 'quantile-transform-uniform'])
    parser.add_argument('--transform_targets_for_each_node_separately', default=False, action='store_true')

    # Past targets used as features preprocessing.
    parser.add_argument('--past_targets_as_features_transform', type=str, default='quantile-transform-normal',
                        choices=['none', 'standard-scaler', 'min-max-scaler', 'robust-scaler',
                                 'power-transform-yeo-johnson', 'quantile-transform-normal',
                                 'quantile-transform-uniform'])
    parser.add_argument('--transform_past_targets_as_features_for_each_node_separately', default=False,
                        action='store_true')

    # Past targets used as features imputation.
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
    parser.add_argument('--in_nirvana', default=False, action='store_true', help='Launch in Nirvana.')

    args = parser.parse_args()

    return args


def compute_loss(model, dataset, timestamps_batch, loss_fn, amp=False):
    features, targets, targets_nan_mask = dataset.get_timestamps_batch_features_and_targets(timestamps_batch)

    with autocast(enabled=amp):
        preds = model(graph=dataset.train_batched_graph, x=features)
        loss = loss_fn(input=preds, target=targets, reduction='none')
        loss[targets_nan_mask] = 0
        loss = loss.sum() / (~targets_nan_mask).sum()

    return loss


def optimizer_step(loss, optimizer, scaler):
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()


@torch.no_grad()
def evaluate_on_val_or_test(model, dataset, split, timestamps_loader, loss_fn, metric, amp=False):
    preds = []
    for timestamps_batch in timestamps_loader:
        padded = False
        if len(timestamps_batch) != dataset.eval_batch_size:
            padding_size = dataset.eval_batch_size - len(timestamps_batch)
            padding = torch.zeros(padding_size, dtype=torch.int32)
            timestamps_batch = torch.cat([timestamps_batch, padding], axis=0)
            padded = True

        features = dataset.get_timestamps_batch_features(timestamps_batch)
        with autocast(enabled=amp):
            cur_preds = model(graph=dataset.eval_batched_graph, x=features)

        cur_preds = cur_preds.reshape(dataset.eval_batch_size, dataset.num_nodes, dataset.targets_dim).squeeze(2)
        if padded:
            cur_preds = cur_preds[:-padding_size]

        preds.append(cur_preds)

    preds = torch.cat(preds, axis=0)
    preds_orig = dataset.transform_preds_to_orig(preds)

    if split == 'val':
        targets_orig, targets_nan_mask = dataset.get_val_targets_orig()
    elif split == 'test':
        targets_orig, targets_nan_mask = dataset.get_test_targets_orig()
    else:
        raise ValueError(f'Unknown split: {split}. Split argument should be either val or test.')

    loss = loss_fn(input=preds_orig, target=targets_orig, reduction='none')
    loss[targets_nan_mask] = 0
    loss = loss.sum() / (~targets_nan_mask).sum()

    metric = loss.sqrt().item() if metric == 'RMSE' else loss.item()

    return metric


@torch.no_grad()
def evaluate_on_val_and_test(model, dataset, val_timestamps_loader, test_timestamps_loader, loss_fn, metric, amp=False):
    val_metric = evaluate_on_val_or_test(model=model, dataset=dataset, split='val',
                                         timestamps_loader=val_timestamps_loader,
                                         loss_fn=loss_fn, metric=metric, amp=amp)
    test_metric = evaluate_on_val_or_test(model=model, dataset=dataset, split='test',
                                          timestamps_loader=test_timestamps_loader,
                                          loss_fn=loss_fn, metric=metric, amp=amp)

    metrics = {
        f'val {metric}': val_metric,
        f'test {metric}': test_metric
    }

    return metrics


def train(model, dataset, loss_fn, metric, logger, num_epochs, num_accumulation_steps, eval_every, lr, weight_decay,
          run_id, device, amp=False, seed=None):
    if seed is not None:
        torch.manual_seed(seed)

    train_timestamps_loader = DataLoader(dataset.train_timestamps, batch_size=dataset.train_batch_size, shuffle=True,
                                         drop_last=True, num_workers=1)
    val_timestamps_loader = DataLoader(dataset.val_timestamps, batch_size=dataset.eval_batch_size, shuffle=False,
                                       drop_last=False, num_workers=1)
    test_timestamps_loader = DataLoader(dataset.test_timestamps, batch_size=dataset.eval_batch_size, shuffle=False,
                                        drop_last=False, num_workers=1)

    num_steps = len(train_timestamps_loader) * num_epochs

    model.to(device)

    parameter_groups = get_parameter_groups(model)
    optimizer = torch.optim.AdamW(parameter_groups, lr=lr, weight_decay=weight_decay)
    scaler = GradScaler(enabled=amp)

    logger.start_run(run=run_id)
    epoch = 1
    steps_till_optimizer_step = num_accumulation_steps
    optimizer_steps_till_eval = eval_every
    optimizer_steps_done = 0
    loss = 0
    metrics = {}
    train_timestamps_loader_iterator = iter(train_timestamps_loader)
    model.train()
    with tqdm(total=num_steps, desc=f'Run {run_id}') as progress_bar:
        for step in range(1, num_steps + 1):
            cur_train_timestamps_batch = next(train_timestamps_loader_iterator)
            cur_step_loss = compute_loss(model=model, dataset=dataset, timestamps_batch=cur_train_timestamps_batch,
                                         loss_fn=loss_fn, amp=amp)
            loss += cur_step_loss
            steps_till_optimizer_step -= 1

            if steps_till_optimizer_step == 0:
                loss /= num_accumulation_steps
                optimizer_step(loss=loss, optimizer=optimizer, scaler=scaler)
                loss = 0
                optimizer_steps_done += 1
                steps_till_optimizer_step = num_accumulation_steps
                optimizer_steps_till_eval -= 1

            if (optimizer_steps_till_eval == 0 or
                    train_timestamps_loader_iterator._num_yielded == len(train_timestamps_loader)):
                progress_bar.set_postfix_str('     Evaluating...     ' + progress_bar.postfix)
                model.eval()
                metrics = evaluate_on_val_and_test(model=model, dataset=dataset,
                                                   val_timestamps_loader=val_timestamps_loader,
                                                   test_timestamps_loader=test_timestamps_loader,
                                                   loss_fn=loss_fn, metric=metric, amp=amp)
                logger.update_metrics(metrics=metrics, step=optimizer_steps_done, epoch=epoch)
                model.train()

                if optimizer_steps_till_eval == 0:
                    optimizer_steps_till_eval = eval_every

            cur_step_metric = cur_step_loss.sqrt().item() if metric == 'RMSE' else cur_step_loss.item()

            progress_bar.update()
            progress_bar.set_postfix(
                {metric: f'{value:.2f}' for metric, value in metrics.items()} |
                {'cur step metric': f'{cur_step_metric:.2f}', 'epoch': epoch}
            )

            if train_timestamps_loader_iterator._num_yielded == len(train_timestamps_loader):
                train_timestamps_loader_iterator = iter(train_timestamps_loader)
                epoch += 1

    logger.finish_run()
    model.cpu()


def main():
    args = get_args()

    torch.set_num_threads(args.num_threads)

    Model = ModelRegistry.get_model_class(args.model_class)

    dataset = Dataset(
        name_or_path=args.dataset,
        prediction_horizon=args.prediction_horizon,
        only_predict_at_end_of_horizon=args.only_predict_at_end_of_horizon,
        provide_sequnce_inputs=Model.sequence_input,
        direct_lookback_num_steps=args.direct_lookback_num_steps,
        seasonal_lookback_periods=args.seasonal_lookback_periods,
        seasonal_lookback_num_steps=args.seasonal_lookback_num_steps,
        drop_early_train_timestamps=args.drop_early_train_timestamps,
        reverse_edges=args.reverse_edges,
        to_undirected=args.to_undirected,
        use_forward_and_reverse_edges_as_different_edge_types=\
            args.use_forward_and_reverse_edges_as_different_edge_types,
        add_self_loops=args.do_not_separate_ego_node_representation,
        targets_transform=args.targets_transform,
        transform_targets_for_each_node_separately=args.transform_targets_for_each_node_separately,
        past_targets_as_features_transform=args.past_targets_as_features_transform,
        transform_past_targets_as_features_for_each_node_separately=\
            args.transform_past_targets_as_features_for_each_node_separately,
        imputation_startegy_for_nan_targets=args.imputation_startegy_for_nan_targets,
        add_features_for_nan_targets=args.add_features_for_nan_targets,
        do_not_use_temporal_features=args.do_not_use_temporal_features,
        do_not_use_spatial_features=args.do_not_use_spatial_features,
        do_not_use_spatiotemporal_features=args.do_not_use_spatiotemporal_features,
        use_deepwalk_node_embeddings=args.use_deepwalk_node_embeddings,
        initialize_learnable_node_embeddings_with_deepwalk=args.initialize_learnable_node_embeddings_with_deepwalk,
        imputation_strategy_for_num_features=args.imputation_strategy_for_numerical_features,
        num_features_transform=args.numerical_features_transform,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        device=args.device,
        in_nirvana=args.in_nirvana
    )

    if args.metric == 'RMSE':
        loss_fn = F.mse_loss
    elif args.metric == 'MAE':
        loss_fn = F.l1_loss
    else:
        raise ValueError(f'Unsupported metric: {args.metric}.')

    logger = Logger(args)

    for run in range(1, args.num_runs + 1):
        model = Model(
            neighborhood_aggregation_name=args.neighborhood_aggregation,
            neighborhood_aggregation_sep=not args.do_not_separate_ego_node_representation,
            sequence_encoder_name=args.sequence_encoder,
            normalization_name=args.normalization,
            num_edge_types=len(dataset.graph.etypes),
            num_residual_blocks=args.num_residual_blocks,
            features_dim=dataset.features_dim,
            hidden_dim=args.hidden_dim,
            output_dim=dataset.targets_dim,
            neighborhood_aggr_attn_num_heads=args.neighborhood_aggr_attn_num_heads,
            seq_encoder_num_layers=args.seq_encoder_num_layers,
            seq_encoder_rnn_type_name=args.seq_encoder_rnn_type,
            seq_encoder_attn_num_heads=args.seq_encoder_attn_num_heads,
            seq_encoder_bidir_attn=args.seq_encoder_bidir_attn,
            seq_encoder_seq_len=args.direct_lookback_num_steps,
            dropout=args.dropout,
            use_learnable_node_embeddings=args.use_learnable_node_embeddings,
            num_nodes=dataset.graph.num_nodes(),
            learnable_node_embeddings_dim=args.learnable_node_embeddings_dim,
            initialize_learnable_node_embeddings_with_deepwalk=args.initialize_learnable_node_embeddings_with_deepwalk,
            deepwalk_node_embeddings=dataset.deepwalk_embeddings_for_initializing_learnable_embeddings,
            use_plr_for_num_features=args.use_plr_for_num_features,
            num_features_mask=dataset.num_features_mask,
            plr_num_features_frequencies_dim=args.plr_num_features_frequencies_dim,
            plr_num_features_frequencies_scale=args.plr_num_features_frequencies_scale,
            plr_num_features_embedding_dim=args.plr_num_features_embedding_dim,
            plr_num_features_shared_linear=args.plr_num_features_shared_linear,
            plr_num_features_shared_frequencies=args.plr_num_features_shared_frequencies,
            use_plr_for_past_targets=args.use_plr_for_past_targets,
            past_targets_mask=dataset.past_targets_mask,
            plr_past_targets_frequencies_dim=args.plr_past_targets_frequencies_dim,
            plr_past_targets_frequencies_scale=args.plr_past_targets_frequencies_scale,
            plr_past_targets_embedding_dim=args.plr_past_targets_embedding_dim,
            plr_past_targets_shared_linear=args.plr_past_targets_shared_linear,
            plr_past_targets_shared_frequencies=args.plr_past_targets_shared_frequencies
        )

        train(model=model, dataset=dataset, loss_fn=loss_fn, metric=args.metric, logger=logger,
              num_epochs=args.num_epochs, num_accumulation_steps=args.num_accumulation_steps,
              eval_every=args.eval_every, lr=args.lr, weight_decay=args.weight_decay, run_id=run,
              device=args.device, amp=args.amp, seed=run)

    logger.print_metrics_summary()


if __name__ == '__main__':
    main()
