import argparse
from tqdm import tqdm

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from models import LinearModel, NeuralNetworkModel
from datasets import Dataset
from utils import Logger, get_parameter_groups


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', type=str, required=True, help='Experiment name.')
    parser.add_argument('--save_dir', type=str, default='experiments', help='Base directory for saving information.')
    parser.add_argument('--dataset', type=str, default='pems-bay',
                        choices=['metr-la', 'pems-bay', 'largest', 'largest-2019', 'city-roads-spt-M',
                                 'city-roads-spt-L', 'weatherbecnh-era5', 'weatherbecnh-era5-usa'])
    parser.add_argument('--metric', type=str, default='RMSE', choices=['RMSE', 'MAE'])

    # Select future timestamps targets from which will be used for prediction.
    parser.add_argument('--prediction_horizon', type=int, default=12)
    parser.add_argument('--only_predict_at_end_of_horizon', default=False, action='store_true')

    # Select past timestamps targets from which will be used as features.
    parser.add_argument('--direct_lookback_num_steps', type=int, default=48)
    parser.add_argument('--seasonal_lookback_periods', nargs='+', type=int, default=None,
                        help='Should have the same number of values as seasonal_lookback_num_steps argument.')
    parser.add_argument('--seasonal_lookback_num_steps', nargs='+', type=int, default=None,
                        help='Should have the same number of values as seasonal_lookback_periods argument.')
    parser.add_argument('--drop_early_train_timestamps', type=str, default='direct', choices=['all', 'direct', 'none'])

    # Graph preprocessing (use at most one of these arguments).
    parser.add_argument('--reverse_edges', default=False, action='store_true')
    parser.add_argument('--to_undirected', default=False, action='store_true')

    # Target preprocessing.
    parser.add_argument('--target_transform', type=str, default='standard-scaler',
                        choices=['none', 'standard-scaler', 'min-max-scaler', 'robust-scaler',
                                 'power-transform-yeo-johnson', 'quantile-transform-normal',
                                 'quantile-transform-uniform'])
    parser.add_argument('--transform_targets_for_each_node_separately', default=False, action='store_true')

    # Target imputation.
    parser.add_argument('--imputation_startegy_for_nan_targets', type=str, default='prev', choices=['prev', 'zero'])
    parser.add_argument('--add_features_for_nan_targets', default=False, action='store_true')

    # Select node features.
    parser.add_argument('--do_not_use_temporal_features', default=False, action='store_true')
    parser.add_argument('--do_not_use_spatial_features', default=False, action='store_true')
    parser.add_argument('--do_not_use_spatiotemporal_features', default=False, action='store_true')
    parser.add_argument('--use_deepwalk_node_embeddings', default=False, action='store_true')
    parser.add_argument('--use_learnable_node_embeddings', default=False, action='store_true')
    parser.add_argument('--learnable_node_embeddings_dim', type=int, default=128)
    parser.add_argument('--initialize_learnable_node_embeddings_with_deepwalk', default=False, action='store_true')

    # Numerical features preprocessing.
    parser.add_argument('--imputation_strategy_for_numerical_features', type=str, default='most_frequent',
                        choices=['mean', 'median', 'most_frequent'],
                        help='Only used for datasets that have NaNs in static numerical features.')
    parser.add_argument('--numerical_features_transform', type=str, default='quantile-transform-normal',
                        choices=['none', 'standard-scaler', 'min-max-scaler', 'robust-scaler',
                                 'power-transform-yeo-johnson', 'quantile-transform-normal',
                                 'quantile-transform-uniform'])

    # PLR embeddings for numerical features.
    parser.add_argument('--plr', default=False, action='store_true', help='Use PLR embeddings for numerical features.')
    parser.add_argument('--plr_apply_to_past_targets', default=False, action='store_true')
    parser.add_argument('--plr_num_frequencies', type=int, default=48, help='Only used if plr is True')
    parser.add_argument('--plr_frequency_scale', type=float, default=0.01, help='Only used if plr is True')
    parser.add_argument('--plr_embedding_dim', type=int, default=16, help='Only used if plr is True')
    parser.add_argument('--plr_lite', default=False, action='store_true', help='Only used if plr is True')

    # Model architecture.
    parser.add_argument('--model', type=str, default='ResNet-MeanAggr',
                        choices=['linear', 'ResNet', 'ResNet-MeanAggr', 'ResNet-AttnGATAggr', 'ResNet-AttnTrfAggr'])
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--normalization', type=str, default='LayerNorm', choices=['None', 'LayerNorm', 'BatchNorm'])

    # Regularization.
    parser.add_argument('--dropout', type=float, default=0)
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

    args = parser.parse_args()

    return args


def compute_loss(model, dataset, timestamps_batch, loss_fn, amp=False):
    features, targets, targets_nan_mask = dataset.get_timestamps_batch_data(timestamps_batch)

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
def evaluate(model, dataset, val_timestamps_loader, test_timestamps_loader, loss_fn, metric, amp=False):
    val_preds = []
    for val_timestamps_batch in val_timestamps_loader:
        padded = False
        if len(val_timestamps_batch) != dataset.eval_batch_size:
            padding_size = dataset.eval_batch_size - len(val_timestamps_batch)
            padding = torch.zeros(padding_size, dtype=torch.int32)
            val_timestamps_batch = torch.cat([val_timestamps_batch, padding], axis=0)
            padded = True

        features, _, _ = dataset.get_timestamps_batch_data(val_timestamps_batch)
        with autocast(enabled=amp):
            preds = model(graph=dataset.eval_batched_graph, x=features)

        preds = preds.reshape(dataset.eval_batch_size, dataset.num_nodes, dataset.targets_dim).squeeze(2)
        if padded:
            preds = preds[:-padding_size]

        val_preds.append(preds)

    val_preds = torch.cat(val_preds, axis=0)
    val_preds_orig = dataset.transform_preds_to_orig(val_preds)

    val_targets_orig, val_targets_nan_mask = dataset.get_val_targets_orig()

    val_loss = loss_fn(input=val_preds_orig, target=val_targets_orig, reduction='none')
    val_loss[val_targets_nan_mask] = 0
    val_loss = val_loss.sum() / (~val_targets_nan_mask).sum()

    val_metric = val_loss.sqrt().item() if metric == 'RMSE' else val_loss.item()

    test_preds = []
    for test_timestamps_batch in test_timestamps_loader:
        padded = False
        if len(test_timestamps_batch) != dataset.eval_batch_size:
            padding_size = dataset.eval_batch_size - len(test_timestamps_batch)
            padding = torch.zeros(padding_size, dtype=torch.int32)
            test_timestamps_batch = torch.cat([test_timestamps_batch, padding], axis=0)
            padded = True

        features, _, _ = dataset.get_timestamps_batch_data(test_timestamps_batch)
        with autocast(enabled=amp):
            preds = model(graph=dataset.eval_batched_graph, x=features)

        preds = preds.reshape(dataset.eval_batch_size, dataset.num_nodes, dataset.targets_dim).squeeze(2)
        if padded:
            preds = preds[:-padding_size]

        test_preds.append(preds)

    test_preds = torch.cat(test_preds, axis=0)
    test_preds_orig = dataset.transform_preds_to_orig(test_preds)

    test_targets_orig, test_targets_nan_mask = dataset.get_test_targets_orig()

    test_loss = loss_fn(input=test_preds_orig, target=test_targets_orig, reduction='none')
    test_loss[test_targets_nan_mask] = 0
    test_loss = test_loss.sum() / (~test_targets_nan_mask).sum()

    test_metric = test_loss.sqrt().item() if metric == 'RMSE' else test_loss.item()

    metrics = {
        f'val {metric}': val_metric,
        f'test {metric}': test_metric
    }

    return metrics


def main():
    args = get_args()

    torch.set_num_threads(args.num_threads)

    torch.manual_seed(0)

    dataset = Dataset(name=args.dataset,
                      prediction_horizon=args.prediction_horizon,
                      only_predict_at_end_of_horizon=args.only_predict_at_end_of_horizon,
                      direct_lookback_num_steps=args.direct_lookback_num_steps,
                      seasonal_lookback_periods=args.seasonal_lookback_periods,
                      seasonal_lookback_num_steps=args.seasonal_lookback_num_steps,
                      drop_early_train_timestamps=args.drop_early_train_timestamps,
                      reverse_edges=args.reverse_edges,
                      to_undirected=args.to_undirected,
                      target_transform=args.target_transform,
                      transform_targets_for_each_node_separately=args.transform_targets_for_each_node_separately,
                      imputation_startegy_for_nan_targets=args.imputation_startegy_for_nan_targets,
                      add_features_for_nan_targets=args.add_features_for_nan_targets,
                      do_not_use_temporal_features=args.do_not_use_temporal_features,
                      do_not_use_spatial_features=args.do_not_use_spatial_features,
                      do_not_use_spatiotemporal_features=args.do_not_use_spatiotemporal_features,
                      use_deepwalk_node_embeddings=args.use_deepwalk_node_embeddings,
                      initialize_learnable_node_embeddings_with_deepwalk=\
                          args.initialize_learnable_node_embeddings_with_deepwalk,
                      imputation_strategy_for_num_features=args.imputation_strategy_for_numerical_features,
                      num_features_transform=args.numerical_features_transform,
                      plr_apply_to_past_targets=args.plr_apply_to_past_targets,
                      train_batch_size=args.train_batch_size,
                      eval_batch_size=args.eval_batch_size,
                      device=args.device)

    train_timestamps_loader = DataLoader(dataset.train_timestamps, batch_size=dataset.train_batch_size, shuffle=True,
                                         drop_last=True, num_workers=1)
    val_timestamps_loader = DataLoader(dataset.val_timestamps, batch_size=dataset.eval_batch_size, shuffle=False,
                                       drop_last=False, num_workers=1)
    test_timestamps_loader = DataLoader(dataset.test_timestamps, batch_size=dataset.eval_batch_size, shuffle=False,
                                        drop_last=False, num_workers=1)

    if args.metric == 'RMSE':
        loss_fn = F.mse_loss
    elif args.metric == 'MAE':
        loss_fn = F.l1_loss
    else:
        raise ValueError(f'Unsupported metric: {args.metric}')

    logger = Logger(args)

    num_steps = len(train_timestamps_loader) * args.num_epochs

    for run in range(1, args.num_runs + 1):
        if args.model == 'linear':
            model = LinearModel(features_dim=dataset.features_dim, output_dim=dataset.targets_dim)
        else:
            model = NeuralNetworkModel(model_name=args.model,
                                       num_layers=args.num_layers,
                                       features_dim=dataset.features_dim,
                                       hidden_dim=args.hidden_dim,
                                       output_dim=dataset.targets_dim,
                                       num_heads=args.num_heads,
                                       normalization=args.normalization,
                                       dropout=args.dropout,
                                       use_learnable_node_embeddings=args.use_learnable_node_embeddings,
                                       num_nodes=dataset.graph.num_nodes(),
                                       learnable_node_embeddings_dim=args.learnable_node_embeddings_dim,
                                       initialize_learnable_node_embeddings_with_deepwalk=\
                                           args.initialize_learnable_node_embeddings_with_deepwalk,
                                       deepwalk_node_embeddings=\
                                           dataset.deepwalk_embeddings_for_initializing_learnable_embeddings,
                                       use_plr=args.plr,
                                       num_features_mask=dataset.num_features_mask,
                                       plr_num_frequencies=args.plr_num_frequencies,
                                       plr_frequency_scale=args.plr_frequency_scale,
                                       plr_embedding_dim=args.plr_embedding_dim,
                                       use_plr_lite=args.plr_lite)

        model.to(args.device)

        parameter_groups = get_parameter_groups(model)
        optimizer = torch.optim.AdamW(parameter_groups, lr=args.lr, weight_decay=args.weight_decay)
        scaler = GradScaler(enabled=args.amp)

        logger.start_run(run=run)
        epoch = 1
        steps_till_optimizer_step = args.num_accumulation_steps
        optimizer_steps_till_eval = args.eval_every
        optimizer_steps_done = 0
        loss = 0
        metrics = {}
        train_timestamps_loader_iterator = iter(train_timestamps_loader)
        model.train()
        with tqdm(total=num_steps, desc=f'Run {run}') as progress_bar:
            for step in range(1, num_steps + 1):
                cur_train_timestamps_batch = next(train_timestamps_loader_iterator)
                cur_step_loss = compute_loss(model=model, dataset=dataset, timestamps_batch=cur_train_timestamps_batch,
                                             loss_fn=loss_fn, amp=args.amp)
                loss += cur_step_loss
                steps_till_optimizer_step -= 1

                if steps_till_optimizer_step == 0:
                    loss /= args.num_accumulation_steps
                    optimizer_step(loss=loss, optimizer=optimizer, scaler=scaler)
                    loss = 0
                    optimizer_steps_done += 1
                    steps_till_optimizer_step = args.num_accumulation_steps
                    optimizer_steps_till_eval -= 1

                if (optimizer_steps_till_eval == 0 or
                        train_timestamps_loader_iterator._num_yielded == len(train_timestamps_loader)):
                    model.eval()
                    metrics = evaluate(model=model, dataset=dataset, val_timestamps_loader=val_timestamps_loader,
                                       test_timestamps_loader=test_timestamps_loader, loss_fn=loss_fn,
                                       metric=args.metric, amp=args.amp)
                    logger.update_metrics(metrics=metrics, step=optimizer_steps_done)
                    model.train()

                    if optimizer_steps_till_eval == 0:
                        optimizer_steps_till_eval = args.eval_every

                cur_step_metric = cur_step_loss.sqrt().item() if args.metric == 'RMSE' else cur_step_loss.item()

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

    logger.print_metrics_summary()


if __name__ == '__main__':
    main()
