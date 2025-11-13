import os
import typing as tp
import yaml
from pathlib import Path
import numpy as np
import torch
from nirvana_utils import copy_snapshot_to_out, copy_out_to_snapshot
from time import perf_counter

TorchStateDict = tp.Mapping[str, torch.FloatTensor]


class Logger:
    
    def __init__(self, args, start_from_scratch=True):
        if args.dataset.endswith('.npz'):
            dataset_name = os.path.splitext(os.path.basename(args.dataset))[0].replace('_', '-')
        else:
            dataset_name = args.dataset

        self.nirvana = args.nirvana
        self.primary_metric = args.metric  # Renamed from self.metric for clarity
        self.do_not_evaluate_on_test = args.do_not_evaluate_on_test
        
        # store dictionaries of all metrics instead of single values
        self.val_metrics = []  # List of dicts: [{'RMSE': 0.5, 'MAE': 0.3, ...}, ...]
        self.test_metrics = None if args.do_not_evaluate_on_test else []
        
        # Keep track of best primary metric values for each run
        self.best_val_primary_metrics = []  # List of single values for primary metric
        self.best_test_primary_metrics = None if args.do_not_evaluate_on_test else []
        
        self.best_steps = []
        self.best_epochs = []
        self.num_runs = args.num_runs
        self.cur_run = None
        self.in_nirvana = args.nirvana

        if start_from_scratch:
            self.save_dir = self.get_save_dir(base_dir=args.save_dir, dataset_name=dataset_name, experiment_name=args.name)

            print(f'Results will be saved to {self.save_dir}.')
            with open(os.path.join(self.save_dir, 'args.yaml'), 'w') as file:
                yaml.safe_dump(vars(args), file, sort_keys=False)
            self.current_run_already_started: bool | None = False
            self.elapsed_time = 0
            self.max_memory_allocated = 0
        else:
            self.save_dir = None  # Will be set during restarting
            self.current_run_already_started = None
            self.elapsed_time = None
            self.max_memory_allocated = None

        self._start_time = perf_counter()

    def set_parameters_from_restarted_job(self, val_metrics, test_metrics, cur_run, best_steps, best_epochs, 
                                         save_dir, current_run_already_started, elapsed_time, max_memory_allocated,
                                         best_val_primary_metrics=None, best_test_primary_metrics=None):
        self.val_metrics = val_metrics
        self.test_metrics = test_metrics
        self.cur_run = cur_run
        self.best_steps = best_steps
        self.best_epochs = best_epochs
        self.save_dir = save_dir
        self.current_run_already_started = current_run_already_started
        self.elapsed_time = elapsed_time
        self.max_memory_allocated = max_memory_allocated
        
        # Handle backward compatibility: if old checkpoint doesn't have these, reconstruct them
        if best_val_primary_metrics is not None:
            self.best_val_primary_metrics = best_val_primary_metrics
        else:
            # Reconstruct from val_metrics if loading old checkpoint
            self.best_val_primary_metrics = [m.get(f'val {self.primary_metric}') if isinstance(m, dict) else m 
                                             for m in val_metrics]
        
        if best_test_primary_metrics is not None:
            self.best_test_primary_metrics = best_test_primary_metrics
        elif not self.do_not_evaluate_on_test:
            self.best_test_primary_metrics = [m.get(f'test {self.primary_metric}') if isinstance(m, dict) else m 
                                              for m in test_metrics]
        
        print(f"Logging will be resumed at save directory {self.save_dir}")

    def _update_timer_and_torch_monitor(self):
        # elapsed is updated here:
        time_spent_after_last_elapse = perf_counter() - self._start_time
        self.elapsed_time += time_spent_after_last_elapse
        self._start_time = perf_counter()

        self.max_memory_allocated = max(self.max_memory_allocated, torch.cuda.max_memory_allocated())

    def get_parameters_for_checkpoint(self) -> dict[str, tp.Any]:
        self._update_timer_and_torch_monitor()
        return dict(
            val_metrics=self.val_metrics,
            test_metrics=self.test_metrics,
            cur_run=self.cur_run,
            best_steps=self.best_steps,
            best_epochs=self.best_epochs,
            save_dir=self.save_dir,
            current_run_already_started=self.current_run_already_started,
            elapsed_time=self.elapsed_time,
            max_memory_allocated=self.max_memory_allocated,
            best_val_primary_metrics=self.best_val_primary_metrics,
            best_test_primary_metrics=self.best_test_primary_metrics,
        )

    def start_run(self, run):
        assert self.current_run_already_started is not None
        self._start_time = perf_counter()

        if not self.current_run_already_started:
            self.current_run_already_started = True
            self.cur_run = run

            self.val_metrics.append(None)
            if not self.do_not_evaluate_on_test:
                self.test_metrics.append(None)

            self.best_val_primary_metrics.append(None)
            if not self.do_not_evaluate_on_test:
                self.best_test_primary_metrics.append(None)

            self.best_steps.append(None)
            self.best_epochs.append(None)

            print(f'Starting run {run}/{self.num_runs}...')
        else:
            print(f'Resuming run {run}/{self.num_runs}...')

    def update_metrics(self, metrics, step, epoch):
        """
        Update metrics for current run. Uses primary_metric to determine best checkpoint.
        
        Args:
            metrics: Dictionary with keys like 'val RMSE', 'val MAE', 'test RMSE', etc.
            step: Current optimizer step
            epoch: Current epoch
        """
        # Extract validation and test metrics
        val_metrics_dict = {k.replace('val ', ''): v for k, v in metrics.items() if k.startswith('val ')}
        
        primary_metric_key = f'val {self.primary_metric}'
        if primary_metric_key not in metrics:
            raise KeyError(f"Primary metric '{primary_metric_key}' not found in metrics. Available: {list(metrics.keys())}")
        
        current_val_primary = metrics[primary_metric_key]
        
        # Check if this is the best checkpoint based on primary metric
        if self.best_val_primary_metrics[-1] is None or current_val_primary < self.best_val_primary_metrics[-1]:
            # Update best metrics
            self.best_val_primary_metrics[-1] = current_val_primary
            self.val_metrics[-1] = val_metrics_dict.copy()
            
            if not self.do_not_evaluate_on_test:
                test_metrics_dict = {k.replace('test ', ''): v for k, v in metrics.items() if k.startswith('test ')}
                self.test_metrics[-1] = test_metrics_dict.copy()
                self.best_test_primary_metrics[-1] = metrics[f'test {self.primary_metric}']

            self.best_steps[-1] = step
            self.best_epochs[-1] = epoch

    def finish_run(self):
        self.save_metrics()
        self.current_run_already_started = False

        if self.do_not_evaluate_on_test:
            print(f'Finished run {self.cur_run}. '
                  f'Best val {self.primary_metric}: {self.best_val_primary_metrics[-1]:.4f} '
                  f'(step {self.best_steps[-1]}, epoch {self.best_epochs[-1]}).')
            
            # Print all validation metrics for best checkpoint
            print(f'All validation metrics at best checkpoint:')
            for metric_name, value in self.val_metrics[-1].items():
                print(f'  {metric_name}: {value:.4f}')
            print()

        else:
            print(f'Finished run {self.cur_run}. '
                  f'Best val {self.primary_metric}: {self.best_val_primary_metrics[-1]:.4f}, '
                  f'corresponding test {self.primary_metric}: {self.best_test_primary_metrics[-1]:.4f} '
                  f'(step {self.best_steps[-1]}, epoch {self.best_epochs[-1]}).')
            
            # Print all metrics for best checkpoint
            print(f'All metrics at best checkpoint:')
            print(f'  Validation:')
            for metric_name, value in self.val_metrics[-1].items():
                print(f'    {metric_name}: {value:.4f}')
            print(f'  Test:')
            for metric_name, value in self.test_metrics[-1].items():
                print(f'    {metric_name}: {value:.4f}')
            print()

    def save_metrics(self):
        self._update_timer_and_torch_monitor()
        num_runs = len(self.val_metrics)

        # Compute statistics for primary metric
        val_primary_mean = np.mean(self.best_val_primary_metrics).item()
        val_primary_std = np.std(self.best_val_primary_metrics, ddof=1).item() if len(self.best_val_primary_metrics) > 1 else np.nan
        best_val_primary = np.min(self.best_val_primary_metrics).item()
        
        # Compute statistics for all metrics across runs
        val_metrics_aggregated = self._aggregate_metrics_across_runs(self.val_metrics)
        
        metrics_dict = {
            'num runs': num_runs,
            'primary_metric': self.primary_metric,
            
            # Primary metric statistics (for backward compatibility)
            f'val {self.primary_metric} mean': val_primary_mean,
            f'val {self.primary_metric} std': val_primary_std,
            f'best val {self.primary_metric}': best_val_primary,
            
            # All validation metrics aggregated
            'val_metrics_mean': val_metrics_aggregated['mean'],
            'val_metrics_std': val_metrics_aggregated['std'],
            'val_metrics_min': val_metrics_aggregated['min'],
            'val_metrics_max': val_metrics_aggregated['max'],
            
            # Raw values for each run
            'val_metrics_per_run': self.val_metrics,
            'best_val_primary_metric_per_run': self.best_val_primary_metrics,
            
            # Metadata
            'best steps': self.best_steps,
            'best epochs': self.best_epochs,
            'elapsed_time': self.elapsed_time,
            'max_memory_allocated': self.max_memory_allocated,
            'max_memory_allocated_mb': self.max_memory_allocated // 2 ** 20,
        }

        if not self.do_not_evaluate_on_test:
            test_primary_mean = np.mean(self.best_test_primary_metrics).item()
            test_primary_std = np.std(self.best_test_primary_metrics, ddof=1).item() if len(self.best_test_primary_metrics) > 1 else np.nan
            best_test_primary = np.min(self.best_test_primary_metrics).item()
            
            test_metrics_aggregated = self._aggregate_metrics_across_runs(self.test_metrics)
            
            metrics_dict.update({
                # Primary metric statistics (for backward compatibility)
                f'test {self.primary_metric} mean': test_primary_mean,
                f'test {self.primary_metric} std': test_primary_std,
                f'best test {self.primary_metric}': best_test_primary,
                
                # All test metrics aggregated
                'test_metrics_mean': test_metrics_aggregated['mean'],
                'test_metrics_std': test_metrics_aggregated['std'],
                'test_metrics_min': test_metrics_aggregated['min'],
                'test_metrics_max': test_metrics_aggregated['max'],
                
                # Raw values for each run
                'test_metrics_per_run': self.test_metrics,
                'best_test_primary_metric_per_run': self.best_test_primary_metrics,
            })

        with open(os.path.join(self.save_dir, 'metrics.yaml'), 'w') as file:
            yaml.safe_dump(metrics_dict, file, sort_keys=False)

    def _aggregate_metrics_across_runs(self, metrics_list):
        """Aggregate metrics across multiple runs.
        
        Args:
            metrics_list: List of dicts, one per run
            
        Returns:
            Dict with 'mean', 'std', 'min', 'max' for each metric
        """
        if not metrics_list or metrics_list[0] is None:
            return {'mean': {}, 'std': {}, 'min': {}, 'max': {}}
        
        # Get all metric names from first run
        metric_names = list(metrics_list[0].keys())
        
        aggregated = {
            'mean': {},
            'std': {},
            'min': {},
            'max': {}
        }
        
        for metric_name in metric_names:
            values = [run_metrics[metric_name] for run_metrics in metrics_list if run_metrics is not None]
            
            if values:
                aggregated['mean'][metric_name] = np.mean(values).item()
                aggregated['std'][metric_name] = np.std(values, ddof=1).item() if len(values) > 1 else np.nan
                aggregated['min'][metric_name] = np.min(values).item()
                aggregated['max'][metric_name] = np.max(values).item()
        
        return aggregated

    def print_metrics_summary(self):
        with open(os.path.join(self.save_dir, 'metrics.yaml'), 'r') as file:
            metrics = yaml.safe_load(file)

        print(f'\n{"="*80}')
        print(f'METRICS SUMMARY - {metrics["num runs"]} runs completed')
        print(f'{"="*80}')
        
        primary_metric = metrics.get('primary_metric', self.primary_metric)
        
        print(f'\nPrimary Metric: {primary_metric}')
        print(f'-' * 40)
        print(f'Val {primary_metric} mean: {metrics[f"val {primary_metric} mean"]:.4f} ± {metrics[f"val {primary_metric} std"]:.4f}')
        
        if not self.do_not_evaluate_on_test:
            print(f'Test {primary_metric} mean: {metrics[f"test {primary_metric} mean"]:.4f} ± {metrics[f"test {primary_metric} std"]:.4f}')
        
        # Print all validation metrics
        print(f'\nAll Validation Metrics (mean ± std):')
        print(f'-' * 40)
        for metric_name, mean_val in metrics['val_metrics_mean'].items():
            std_val = metrics['val_metrics_std'][metric_name]
            print(f'{metric_name:8s}: {mean_val:.4f} ± {std_val:.4f}')
        
        # Print all test metrics if available
        if not self.do_not_evaluate_on_test:
            print(f'\nAll Test Metrics (mean ± std):')
            print(f'-' * 40)
            for metric_name, mean_val in metrics['test_metrics_mean'].items():
                std_val = metrics['test_metrics_std'][metric_name]
                print(f'{metric_name:8s}: {mean_val:.4f} ± {std_val:.4f}')
        
        print(f'\nTraining Info:')
        print(f'-' * 40)
        print(f'Elapsed time: {metrics["elapsed_time"]:.2f} seconds')
        print(f'Max memory allocated: {metrics["max_memory_allocated_mb"]} MB')
        print(f'{"="*80}\n')

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


class NirvanaNpzDataWrapper:
    """Mimics default numpy npz dictionary, as Nirvana automatically unpacks it to separate arrays."""

    def __init__(self, root_path: str):
        self.root_path = root_path

    def get_array_path(self, array_name: str):
        return os.path.join(self.root_path, f'{array_name}.npy')

    def __getitem__(self, array_name: str):
        array_path = self.get_array_path(array_name)

        print(f"Accessing `{array_name}` array at {array_path}")
        array = np.load(array_path, allow_pickle=True)

        return array

    def __contains__(self, array_name: str):
        return os.path.exists(self.get_array_path(array_name))


class StateHandler:
    def __init__(self, checkpoint_file_path: Path, checkpoint_dir: Path, checkpoint_steps_interval: int) -> None:
        self.checkpoint_file_path = checkpoint_file_path
        self.checkpoint_steps_interval = checkpoint_steps_interval
        self.checkpoint_dir = checkpoint_dir

        self.num_runs_completed: int = 0
        self.epochs_finished: int = 0
        self.steps_after_run_start: int = 0
        self.optimizer_steps_done: int = 0
        self.loss: float = 0.0

        self.model: torch.nn.Module = ...
        self.optimizer: torch.optim.Optimizer = ...
        self.grad_scaler: torch.amp.GradScaler = ...
        self.logger: Logger = ...

        self._model_state: TorchStateDict | None = None
        self._optimizer_state: TorchStateDict | None = None
        self._grad_scaler_state: TorchStateDict | None = None
        self._logger_state: dict[str, tp.Any] | None = None

        self.predictions_targets_dict : dict[str, torch.Tensor] = {}

    def load_checkpoint(self, initial_loading: bool = False) -> None:
        pass

    def add_logger(self, logger: Logger) -> None:
        assert self.logger is Ellipsis
        self.logger = logger

        if self._logger_state is not None:
            self.logger.set_parameters_from_restarted_job(**self._logger_state)
            del self._logger_state

    def add_model(self, model: torch.nn.Module) -> None:
        assert self.model is Ellipsis
        self.model = model

        if self._model_state is not None:
            self.model.load_state_dict(self._model_state)
            del self._model_state

    def add_optimizer(self, optimizer: torch.optim.Optimizer) -> None:
        assert self.optimizer is Ellipsis
        self.optimizer = optimizer

        if self._optimizer_state is not None:
            self.optimizer.load_state_dict(self._optimizer_state)
            del self._optimizer_state

    def add_grad_scaler(self, scaler: torch.amp.GradScaler) -> None:
        assert self.grad_scaler is Ellipsis
        self.grad_scaler = scaler

        if self._grad_scaler_state is not None:
            self.grad_scaler.load_state_dict(self._grad_scaler_state)
            del self._grad_scaler_state

    def step(self) -> None:
        pass
    
    def finish_epoch(self) -> None:
        pass

    def finish_run(self, predictions_targets_dict: dict[str, torch.Tensor]) -> None:
        del self.model
        del self.optimizer
        del self.grad_scaler

        self.model = ...
        self.optimizer = ...
        self.grad_scaler = ...

    def save_checkpoint(self, finish_run: bool = False) -> None:
        pass


class NirvanaStateHandler(StateHandler):

    def __init__(self, checkpoint_file_path: Path, checkpoint_dir: Path, checkpoint_steps_interval: int) -> None:
        # initialize training attributes
        super().__init__(checkpoint_file_path=checkpoint_file_path, checkpoint_dir=checkpoint_dir, checkpoint_steps_interval=checkpoint_steps_interval)

    @property
    def current_run_model_state_dict(self) -> TorchStateDict:
        return self.model.state_dict()

    @property
    def current_run_optimizer_state_dict(self) -> TorchStateDict:
        return self.optimizer.state_dict()

    @property
    def current_run_scaler_state(self) -> TorchStateDict:
        return self.grad_scaler.state_dict()

    @property
    def logger_state(self) -> dict[str, tp.Any]:
        return self.logger.get_parameters_for_checkpoint()

    def load_checkpoint(self, initial_loading: bool = False):
        if initial_loading:
            copy_snapshot_to_out(self.checkpoint_dir)

        if self.checkpoint_file_path.exists():
            # if path exists, thus logger state always nonempty
            state_dict: dict[str, TorchStateDict | dict[str, tp.Any] | int | float | torch.Tensor] = torch.load(self.checkpoint_file_path, weights_only=True)
            self._logger_state = state_dict["logger_state"]
            self._model_state = state_dict["model_state"]
            self._optimizer_state = state_dict["optimizer_state"]
            self._grad_scaler_state = state_dict["scaler_state"]

            self.steps_after_run_start = state_dict["steps_after_run_start"]
            self.epochs_finished = state_dict["epochs_finished"]
            self.optimizer_steps_done = state_dict["optimizer_steps_done"]
            self.loss = state_dict["loss"]
            self.num_runs_completed = state_dict["runs_completed"]

    def save_checkpoint(self, finish_run: bool = False) -> None:
        print(f"Saving checkpoint to {self.checkpoint_file_path}")
        if finish_run:
            # if run is finished, there is no need to save model's weights and optimizer
            overall_state_dict = dict(
                steps_after_run_start=0,
                epochs_finished=0,
                optimizer_steps_done=0,
                loss=0.0,
                logger_state=self.logger_state,
                model_state=self.current_run_model_state_dict,
                optimizer_state=None,
                scaler_state=None,
                runs_completed=self.num_runs_completed,
            )
        else:
            overall_state_dict = dict(
                steps_after_run_start=self.steps_after_run_start,
                epochs_finished=self.epochs_finished,
                optimizer_steps_done=self.optimizer_steps_done,
                loss=self.loss,
                logger_state=self.logger_state,
                model_state=self.current_run_model_state_dict,
                optimizer_state=self.current_run_optimizer_state_dict,
                scaler_state=self.current_run_scaler_state,
                runs_completed=self.num_runs_completed,
            )

        overall_state_dict.update(self.predictions_targets_dict)

        torch.save(overall_state_dict, f=self.checkpoint_file_path)
        copy_out_to_snapshot(self.checkpoint_dir, dump=True)

    def step(self) -> None:
        self.steps_after_run_start += 1
        if self.steps_after_run_start % self.checkpoint_steps_interval == 0:
            self.save_checkpoint()

    def finish_epoch(self) -> None:
        self.epochs_finished += 1
        self.save_checkpoint()

    def finish_run(self, predictions_targets_dict: dict[str, torch.Tensor]) -> None:
        self.predictions_targets_dict = predictions_targets_dict
        self.num_runs_completed += 1
        self.save_checkpoint(finish_run=True)
        super().finish_run(predictions_targets_dict)


class DummyHandler(StateHandler):
    pass


def getitem_wrapper(func: tp.Callable[[int | torch.Tensor], torch.Tensor]):
    def _inner_func(idx: int | torch.Tensor):
        print(f"Accessing {idx=}")

        result = func(idx)

        return result

    return _inner_func


class TensorMemmapAdapter:
    """
    Wraps memmap numpy object and supports
    """
    def __init__(self, memmap_object: torch.Tensor) -> None:
        self._inner_memmap: torch.Tensor = memmap_object

    def __repr__(self) -> str:
        return repr(self._inner_memmap)

    @getitem_wrapper
    def __getitem__(self, idx: int | torch.Tensor) -> torch.Tensor:
        return torch.from_numpy(self._inner_memmap[idx])


def get_tensor_or_wrap_memmap(array_or_memmap: np.ndarray | torch.Tensor | np.memmap) -> torch.Tensor | TensorMemmapAdapter:
    """
    Either returns tensor or wraps tensor logic aroung numpy memmap file
    """
    if isinstance(array_or_memmap, np.ndarray):
        return torch.from_numpy(array_or_memmap)
    elif isinstance(array_or_memmap, np.memmap):
        return torch.from_numpy(array_or_memmap)
    else:
        return array_or_memmap  # for debug can be replaced with TensorMemmapWrapper


def read_memmap(filepath: str,
                shape: tuple[int, ...],
                dtype: torch.dtype = torch.float32) -> torch.Tensor:
    number_of_elements = np.prod(shape)

    _, file_extension = os.path.splitext(filepath)
    if file_extension == '.pt':
        return torch.load(f=filepath, weights_only=True)
    return torch.from_file(
        filename=filepath, size=number_of_elements, dtype=dtype, shared=False
    ).reshape(shape)
    # return torch.tensor(np.memmap(filename=filepath, dtype="float32", mode="r", shape=shape))
