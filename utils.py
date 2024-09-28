import os
import yaml
import numpy as np


class Logger:
    def __init__(self, args):
        if args.dataset.endswith('.npz'):
            dataset_name = os.path.splitext(os.path.basename(args.dataset))[0].replace('_', '-')
        else:
            dataset_name = args.dataset

        self.save_dir = self.get_save_dir(base_dir=args.save_dir, dataset_name=dataset_name, experiment_name=args.name)
        self.nirvana = args.nirvana
        self.metric = args.metric
        self.do_not_evaluate_on_test = args.do_not_evaluate_on_test
        self.val_metrics = []
        self.test_metrics = None if args.do_not_evaluate_on_test else []
        self.best_steps = []
        self.best_epochs = []
        self.num_runs = args.num_runs
        self.cur_run = None

        print(f'Results will be saved to {self.save_dir}.')
        with open(os.path.join(self.save_dir, 'args.yaml'), 'w') as file:
            yaml.safe_dump(vars(args), file, sort_keys=False)

    def start_run(self, run):
        self.cur_run = run

        self.val_metrics.append(None)
        if not self.do_not_evaluate_on_test:
            self.test_metrics.append(None)

        self.best_steps.append(None)
        self.best_epochs.append(None)

        print(f'Starting run {run}/{self.num_runs}...')

    def update_metrics(self, metrics, step, epoch):
        if self.val_metrics[-1] is None or metrics[f'val {self.metric}'] < self.val_metrics[-1]:
            self.val_metrics[-1] = metrics[f'val {self.metric}']
            if not self.do_not_evaluate_on_test:
                self.test_metrics[-1] = metrics[f'test {self.metric}']

            self.best_steps[-1] = step
            self.best_epochs[-1] = epoch

    def finish_run(self):
        self.save_metrics()

        if self.do_not_evaluate_on_test:
            print(f'Finished run {self.cur_run}. '
                  f'Best val {self.metric}: {self.val_metrics[-1]:.4f} '
                  f'(step {self.best_steps[-1]}, epoch {self.best_epochs[-1]}).\n')

        else:
            print(f'Finished run {self.cur_run}. '
                  f'Best val {self.metric}: {self.val_metrics[-1]:.4f}, '
                  f'corresponding test {self.metric}: {self.test_metrics[-1]:.4f} '
                  f'(step {self.best_steps[-1]}, epoch {self.best_epochs[-1]}).\n')

    def save_metrics(self):
        num_runs = len(self.val_metrics)

        val_metric_mean = np.mean(self.val_metrics).item()
        val_metric_std = np.std(self.val_metrics, ddof=1).item() if len(self.val_metrics) > 1 else np.nan

        if not self.do_not_evaluate_on_test:
            test_metric_mean = np.mean(self.test_metrics).item()
            test_metric_std = np.std(self.test_metrics, ddof=1).item() if len(self.test_metrics) > 1 else np.nan

            metrics = {
                'num runs': num_runs,
                f'val {self.metric} mean': val_metric_mean,
                f'val {self.metric} std': val_metric_std,
                f'test {self.metric} mean': test_metric_mean,
                f'test {self.metric} std': test_metric_std,
                f'val {self.metric} values': self.val_metrics,
                f'test {self.metric} values': self.test_metrics,
                'best steps': self.best_steps,
                'best epochs': self.best_epochs
            }

        else:
            metrics = {
                'num runs': num_runs,
                f'val {self.metric} mean': val_metric_mean,
                f'val {self.metric} std': val_metric_std,
                f'val {self.metric} values': self.val_metrics,
                'best steps': self.best_steps,
                'best epochs': self.best_epochs
            }

        with open(os.path.join(self.save_dir, 'metrics.yaml'), 'w') as file:
            yaml.safe_dump(metrics, file, sort_keys=False)

    def print_metrics_summary(self):
        with open(os.path.join(self.save_dir, 'metrics.yaml'), 'r') as file:
            metrics = yaml.safe_load(file)

        print(f'Finished {metrics["num runs"]} runs.')
        print(f'Val {self.metric} mean: {metrics[f"val {self.metric} mean"]:.4f}')
        print(f'Val {self.metric} std: {metrics[f"val {self.metric} std"]:.4f}')
        if not self.do_not_evaluate_on_test:
            print(f'Test {self.metric} mean: {metrics[f"test {self.metric} mean"]:.4f}')
            print(f'Test {self.metric} std: {metrics[f"test {self.metric} std"]:.4f}')

    @staticmethod
    def get_save_dir(base_dir, dataset_name, experiment_name):
        idx = 1
        save_dir = os.path.join(base_dir, dataset_name, f'{experiment_name}_{idx:02d}')
        while os.path.exists(save_dir):
            idx += 1
            save_dir = os.path.join(base_dir, dataset_name, f'{experiment_name}_{idx:02d}')

        os.makedirs(save_dir)

        return save_dir


def get_parameter_groups(model):
    no_weight_decay_names = ['bias', 'normalization', 'frequencies']

    parameter_groups = [
        {
            'params': [param for name, param in model.named_parameters()
                       if not any(no_weight_decay_name in name for no_weight_decay_name in no_weight_decay_names)]
        },
        {
            'params': [param for name, param in model.named_parameters()
                       if any(no_weight_decay_name in name for no_weight_decay_name in no_weight_decay_names)],
            'weight_decay': 0
        },
    ]

    return parameter_groups


def _check_dim_and_num_heads_consistency(dim, num_heads):
    if dim % num_heads != 0:
        raise ValueError('Dimension mismatch: hidden_dim should be a multiple of num_heads.')
