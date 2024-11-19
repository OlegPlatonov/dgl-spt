import os
import typing as tp
from pathlib import Path
import yaml
import numpy as np
import torch

from nirvana_utils import copy_snapshot_to_out, copy_out_to_snapshot

TorchStateDict = tp.Mapping[str, torch.FloatTensor]

class Logger:
    def __init__(self, args, start_from_scratch=True):
        if args.dataset.endswith('.npz'):
            dataset_name = os.path.splitext(os.path.basename(args.dataset))[0].replace('_', '-')
        else:
            dataset_name = args.dataset

        self.nirvana = args.nirvana
        self.metric = args.metric
        self.do_not_evaluate_on_test = args.do_not_evaluate_on_test
        self.val_metrics = []
        self.test_metrics = None if args.do_not_evaluate_on_test else []
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
            self._restarted = False
        else:
            self.save_dir = None  # Will be set during restarting
            self._restarted = True

    def set_parameters_from_restarted_job(self, val_metrics, test_metrics, cur_run, best_steps, best_epochs, save_dir):
        self.val_metrics = val_metrics
        self.test_metrics = test_metrics
        self.cur_run = cur_run
        self.best_steps = best_steps
        self.best_epochs = best_epochs
        self.save_dir = save_dir

    def get_parameters_for_checkpoint(self) -> dict[str, tp.Any]:
        return dict(
            val_metrics=self.val_metrics,
            test_metrics=self.test_metrics,
            cur_run=self.cur_run,
            best_steps=self.best_steps,
            best_epochs=self.best_epochs,
            save_dir=self.save_dir,
        )

    def start_run(self, run):
        if not self._restarted:
            self.cur_run = run

            self.val_metrics.append(None)
            if not self.do_not_evaluate_on_test:
                self.test_metrics.append(None)

            self.best_steps.append(None)
            self.best_epochs.append(None)

            print(f'Starting run {run}/{self.num_runs}...')
        else:
            print(f'Resuming run {run}/{self.num_runs}...')


    def update_metrics(self, metrics, step, epoch):
        if self.val_metrics[-1] is None or metrics[f'val {self.metric}'] < self.val_metrics[-1]:
            self.val_metrics[-1] = metrics[f'val {self.metric}']
            if not self.do_not_evaluate_on_test:
                self.test_metrics[-1] = metrics[f'test {self.metric}']

            self.best_steps[-1] = step
            self.best_epochs[-1] = epoch

    def finish_run(self):
        self.save_metrics()
        self._restarted = False  # TODO reassure that this is desired behaviour

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
        self.steps_after_epoch_start: int = 0
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


    def load_checkpoint(self) -> None:
        pass

    def add_logger(self, logger: Logger) -> None:
        self.logger = logger

        if self._logger_state is not None:
            self.logger.set_parameters_from_restarted_job(**self._logger_state)
            del self._logger_state

    def add_model(self, model: torch.nn.Module) -> None:
        del self.model
        self.model = model

        if self._model_state is not None:
            self.model.load_state_dict(self._model_state)
            del self._model_state

    def add_optimizer(self, optimizer: torch.optim.Optimizer) -> None:
        del self.optimizer
        self.optimizer = optimizer

        if self._optimizer_state is not None:
            self.optimizer.load_state_dict(self._optimizer_state)
            del self._optimizer_state

    def add_grad_scaler(self, scaler: torch.amp.GradScaler) -> None:
        del self.grad_scaler
        self.grad_scaler = scaler

        if self._grad_scaler_state is not None:
            self.grad_scaler.load_state_dict(self._grad_scaler_state)
            del self._grad_scaler_state

    def step(self) -> None:
        pass
    
    def finish_epoch(self) -> None:
        pass

    def finish_run(self) -> None:
        pass

    def save_checkpoint(self, finish_run: bool = False) -> None:
        pass


# TODO check that the model is the same after being passed here
# TODO checkpoint handling
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

        # self.logger.val_metrics = val_metrics
        # self.logger.test_metrics = test_metrics
        # self.logger.cur_run = cur_run
        # self.logger.best_steps = best_steps
        # self.logger.best_epochs = best_epochs
        # self.logger.save_dir = save_dir

    def load_checkpoint(self, initial_loading: bool = False):
        if initial_loading:
            copy_snapshot_to_out(self.checkpoint_dir)

        if self.checkpoint_file_path.exists():
            # if path exists, thus logger state always nonempty
            state_dict: dict[str, TorchStateDict | dict[str, tp.Any] | int | float] = torch.load(self.checkpoint_file_path)
            self._logger_state = state_dict["logger_state"]
            self._model_state = state_dict["model_state"]
            self._optimizer_state = state_dict["optimizer_state"]
            self._grad_scaler_state = state_dict["scaler_state"]

            self.steps_after_epoch_start = state_dict["steps_after_epoch_start"]
            self.epochs_finished = state_dict["epochs_finished"]
            self.optimizer_steps_done = state_dict["optimizer_steps_done"]
            self.loss = state_dict["loss"]
            self.num_runs_completed = state_dict["num_runs_completed"]

    def save_checkpoint(self, finish_run: bool = False) -> None:
        print(f"Saving checkpoint to {self.checkpoint_file_path}")
        if finish_run:
            # if run is finished, there is no need to save model's weights and optimizer
            overall_state_dict = dict(
                steps_after_epoch_start=0,
                epochs_finished=0,
                optimizer_steps_done=0,
                loss=0.0,
                logger_state=self.logger_state,
                model_state=None,
                optimizer_state=None,
                scaler_state=None,
                runs_completed=self.num_runs_completed,
            )
        else:
            overall_state_dict = dict(
                steps_after_epoch_start=self.steps_after_epoch_start,
                epochs_finished=self.epochs_finished,
                optimizer_steps_done=self.optimizer_steps_done,
                loss=self.loss,
                logger_state=self.logger_state,
                model_state=self.current_run_model_state_dict,
                optimizer_state=self.current_run_optimizer_state_dict,
                scaler_state=self.current_run_scaler_state,
                runs_completed=self.num_runs_completed,
            )

        torch.save(overall_state_dict, f=self.checkpoint_file_path)
        copy_out_to_snapshot(self.checkpoint_file_path, dump=True)

    def step(self) -> None:
        self.steps_after_epoch_start += 1
        if self.steps_after_epoch_start % self.checkpoint_steps_interval == 0:
            self.save_checkpoint()

    def finish_epoch(self) -> None:
        self.steps_after_epoch_start = 0
        self.epochs_finished += 1
        self.save_checkpoint()

    def finish_run(self) -> None:
        self.num_runs_completed += 1
        self.save_checkpoint(finished_run=True)


class DummyHandler(StateHandler):
    pass
