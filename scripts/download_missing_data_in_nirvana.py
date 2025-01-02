
# based on https://a.yandex-team.ru/arcadia/ml/torch/deepspeed_megatron_3d/megatron/yt_utils.py
# based on fast yt reposotory by @optimus
import logging
import queue
import os

import re
import time
from argparse import ArgumentParser
from pathlib import Path
from typing import List

import yt.wrapper as yt
import asyncio
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

try:
    import nirvana_dl
except ImportError:
    raise ImportError("This script is intended to use only in Nirvana")


DEFAULT_PROXY = "hahn"
BYTES_IN_MB = 2**20
DEFAULT_SLICE_SIZE = 64 * BYTES_IN_MB
DEFAULT_NUM_WORKERS = 16

logger = logging.getLogger(__name__)


class FastYT:
    def __init__(self, proxy=DEFAULT_PROXY, slice_size=DEFAULT_SLICE_SIZE, num_workers=DEFAULT_NUM_WORKERS):
        self.proxy = proxy
        self.slice_size = slice_size
        self.num_workers = num_workers
        self.token = os.environ.get("YT_TOKEN")
        self.client = yt.YtClient(proxy=self.proxy, token=self.token)

    @staticmethod
    def _init_worker(proxy, token):
        global client
        client = yt.YtClient(proxy=proxy, token=token)

    @staticmethod
    def _read_from_yt_sync(task_args):
        yt_path, local_path, offset, length = task_args["yt_path"], task_args["local_path"], task_args["offset"], task_args["length"]
        with open(local_path, "r+b") as f:
            data = client.read_file(yt_path, offset=offset, length=length).read()
            f.seek(offset)
            f.write(data)

    async def _process_slice_async(self, executor, slice_task, progress_bar=None):
        loop = asyncio.get_event_loop()
        task_id = slice_task.pop("task_id")
        operation = slice_task.pop("operation")

        logger.info(f"Slice task {task_id} `{operation}` started")
        start = time.time()

        try:
            if operation == "read_from_yt":
                await loop.run_in_executor(executor, self._read_from_yt_sync, slice_task)
            else:
                raise NotImplementedError(f"Operation {operation} not implemented")
            if progress_bar:
                progress_bar.update(slice_task["length"])
        except Exception as e:
            logger.error(f"Error processing slice task {task_id}: {e}")
            return task_id, False

        elapsed = time.time() - start
        length_mb = slice_task["length"] / BYTES_IN_MB
        logger.info(f"Slice task {task_id} `{operation}` complete in {elapsed:.3f} s, loaded: {length_mb:,.2f} MB, speed: {length_mb / elapsed:,.1f} MB/s")

        return task_id, True

    async def _process_tasks(self, slice_tasks):
        progress_bar = tqdm(total=sum([x["length"] for x in slice_tasks]), unit="B", desc="FastYT", unit_scale=True, unit_divisor=1024)
        executor = ProcessPoolExecutor(
            max_workers=self.num_workers,
            initializer=self._init_worker,
            initargs=(self.proxy, self.token),
        )
        tasks = [self._process_slice_async(executor, slice_task, progress_bar) for slice_task in slice_tasks]
        results = await asyncio.gather(*tasks)
        executor.shutdown(wait=True)
        return results

    def slice_filenodes_into_tasks(self, filenodes, fs_path, task_id_start=0):
        tasks_list = []
        for n in filenodes:
            for task_id, offset in enumerate(range(0, n['size'], self.slice_size), start=task_id_start):
                tasks_list.append({
                    "operation": "read_from_yt",
                    "task_id": task_id,
                    "offset": offset,
                    "length": min(n["size"] - offset, self.slice_size),
                    "yt_path": n["cpath"],
                    "local_path": os.path.join(fs_path, n['path']),
                })
        return tasks_list

    def preallocate_fs_files(self, file_nodes, fs_path):
        for n in file_nodes:
            os.system(f"truncate -s {n['size']} {os.path.join(fs_path, n['path'])}")

    def run_tasks(self, tasklist):
        results = asyncio.run(self._process_tasks(tasklist))
        failed_slices = [i for i, success in results if not success]

        if failed_slices:
            logger.error(f"Failed slices: {failed_slices}")
            raise ValueError("Failed to load some slices")
        return results

    def read_from_yt_fast(self, yt_path, local_path):
        """sample application to read single file"""

        # Create the output dir
        local_path_dir = os.path.dirname(local_path)
        os.makedirs(local_path_dir, exist_ok=True)
        logger.info(f"Directory {local_path_dir} ready")

        # create filenodes as a list of 1 filenode dict
        yt_file_size = self.client.get_attribute(yt_path, "uncompressed_data_size")
        filenodes = [dict(cpath=yt_path, fs_path=local_path, size=yt_file_size)]

        # Get the file size and preallocate file
        os.system(f"truncate -s {yt_file_size} {local_path}")
        logger.info(f"Preallocated file {local_path} with size {yt_file_size:,}")

        # Create the slice tasks
        slice_tasks = self.slice_filenodes_into_tasks(filenodes, fs_path=local_path_dir)
        logger.info(f"Loading params: num_slices={len(slice_tasks)} slice_size={self.slice_size:,} yt_file_size={yt_file_size:,}")

        results = self.run_tasks(slice_tasks)
        return results


def get_yt_client(yt_cluster, max_thread_count=50, max_upload_thread_count=1, ssd=False, config=None, token=None):
    if config is None:
        if ssd:
            config = {
                "read_parallel": {
                    "enable": True,
                    # Parallel reading will only be used for very large files.
                    # For small files it just increases master RPS and slows everything down.
                    "data_size_per_thread": 4 * 1024 * 1024 * 1024,
                },
                "read_retries": {
                    # Transactions and locks increase master RPS and slow everything down.
                    # Usually they aren't needed, at least for checkpoint downloading.
                    "create_transaction_and_take_snapshot_lock": False,
                },
                # Parallel uploading works by splitting a file into chunks, uploading the chunks as separate files
                # and concatenating them together. Our files are typically not large enough for this to be worth it.
                "write_parallel": {
                    "enable": False,
                },
                "write_retries": {
                    # Standard retries create transactions for every file.
                    "enable": False,
                    # Even with retries disabled, write_file() will create a transaction for every file unless we
                    # disable it.
                    "transaction_id": "0-0-0-0",
                },
                "proxy": {
                    "commands_with_framing": ["read_file", "write_file"],
                    # Too low timeout will cause excessive retries. Too high timeout will cause us to wait
                    # for minutes for a response that will never come instead of retrying.
                    "heavy_request_timeout": 60000,
                    "request_timeout": 60000,
                    # Disable compression. YT client uses gzip for compression, it's too slow and
                    # can't compress our data.
                    "content_encoding": "identity",
                },
            }
        else:
            config = {
                "read_parallel": {
                    "enable": True,
                    "max_thread_count": max_thread_count,
                    "data_size_per_thread": 256 * 1024 * 1024,
                },
                "write_parallel": {
                    "enable": True,
                    "max_thread_count": max_upload_thread_count,
                    "memory_limit": 12 * 1024**3,
                },
                "proxy": {
                    "commands_with_framing": ["read_file", "write_file"],
                    "heavy_request_timeout": 120 * 1000,
                    "request_timeout": 100 * 1000,
                },
            }

    if token is None:
        token = os.getenv("YT_TOKEN")
    if token is None:
        with open(os.path.join(os.path.expanduser("~"), ".yt/token"), "r") as f:
            token = f.read().strip()

    return yt.YtClient(yt_cluster, config=config, token=token)


def yt_client_wrapper(func):
    """
    It's a decorator, that take any function, that uses yt_client, and transform it into
    function, that take yt_cluster instead. E.g.

        @yt_client_wrapper
        def read_one_row(yt_client, table_path):
            return next(iter(yt_client.read_table(table_path)))

        read_one_row("hahn", ...)

    The function will return the row from the table, which is contained in "hahn" cluster.

    Note, if token specified explicitly via **kwargs, then that token will be used in
        yt_client instantiantion instead of YT_TOKEN env variable.

    """

    def wrapped_func(yt_cluster, *args, **kwargs):
        return func(get_yt_client(yt_cluster, token=kwargs.get("token")), *args, **kwargs)

    return wrapped_func


def check_cached_file(yt_client, yt_path: str, fs_path: str) -> bool:
    """
    Check if yt file is the same size as file in fs for weak caching.
    Returns `True` if the file is the same size on fs as it is on yt.
    """
    fs_path = Path(fs_path)
    if not fs_path.exists():
        return False
    yt_file_size = yt_client.get_attribute(yt_path, "uncompressed_data_size")
    fs_file_size = fs_path.stat().st_size
    return yt_file_size == fs_file_size


@yt_client_wrapper
def scan_yt_dir(yt_client, root_dir: str):
    q = queue.Queue()
    q.put(root_dir)
    results = []

    while not q.empty():
        _cpath = q.get()
        try:
            items = yt_client.list(_cpath, absolute=True, attributes=["type", "uncompressed_data_size", "target_path", "modification_time"])
            for item in items:
                if item.attributes["type"] == "map_node":
                    q.put(str(item))
                    results.append(
                        {"name": os.path.basename(str(item)), "cpath": str(item), "modification_time": item.attributes["modification_time"], "type": "map_node"}
                    )
                elif item.attributes["type"] == "file":
                    results.append(
                        {
                            "name": os.path.basename(str(item)),
                            "cpath": str(item),
                            "size": item.attributes["uncompressed_data_size"],
                            "modification_time": item.attributes["modification_time"],
                            "type": "file",
                        }
                    )
                elif item.attributes["type"] == "link":
                    results.append(
                        {
                            "name": os.path.basename(str(item)),
                            "cpath": str(item),
                            "target_path": item.attributes["target_path"],
                            "modification_time": item.attributes["modification_time"],
                            "type": "link",
                        }
                    )
                else:
                    raise ValueError(f"unknown node type: {item.attributes['type']=}")
            results.append({"name": os.path.basename(_cpath), "cpath": _cpath, "size": len(items), "type": "map_node"})
        except yt.YtResponseError as exc:
            # most likely happens if _cpath points to a file.
            _type = yt_client.get_attribute(_cpath, "type")
            if _type == "file":
                _size = yt_client.get_attribute(_cpath, "uncompressed_data_size")
                results.append({"name": os.path.basename(_cpath), "cpath": _cpath, "size": _size, "type": _type})
            else:
                print(f"Exception occurred while processing {_cpath}: {exc}")

    for node in results:
        node["path"] = str(os.path.relpath(node["cpath"], root_dir))
        if "target_path" in node:
            node["target_path"] = str(os.path.relpath(node["target_path"], root_dir))

    return results


def download_dir_from_yt_fast(
    yt_cluster: str,
    yt_path: str,
    fs_path: str,
    num_tries: int = 10,
    check_cache: bool = False,
    exclude_regexp: str = None,
    filter_by_regexp: str = None,
    files_to_load: List[str] = None,
    num_processes: int = 16,
    yt_client: yt.YtClient = None,
    ssd: bool = False,
    slice_size: int = 16 * 2 ** 20,
    **kwargs,
):
    fs_path = os.path.abspath(fs_path)
    Path(fs_path).mkdir(parents=True, exist_ok=True)
    yt_client = yt_client or get_yt_client(yt_cluster, ssd=ssd)
    nodes = scan_yt_dir(yt_cluster, root_dir=yt_path)
    download_nodes = []
    for n in nodes:
        file_path = n["path"]
        if exclude_regexp is not None and re.match(exclude_regexp, file_path):
            continue
        if filter_by_regexp is not None and not re.match(filter_by_regexp, file_path):
            continue
        if files_to_load is not None and file_path not in files_to_load:
            continue
        download_nodes.append(n)

    # create dirs
    for n in download_nodes:
        if n["type"] == "map_node":
            Path(os.path.join(fs_path, n['path'])).mkdir(parents=True, exist_ok=True)

    # create files
    download_files = [n for n in download_nodes if n["type"] == "file"]

    fyt_client = FastYT(proxy=yt_cluster, slice_size=slice_size, num_workers=num_processes)
    fyt_client.preallocate_fs_files(download_files, fs_path)
    tasklist = fyt_client.slice_filenodes_into_tasks(download_files, fs_path)
    _ = fyt_client.run_tasks(tasklist)

    # create symlinks
    for n in download_nodes:
        if n["type"] == "link":
            os.symlink(os.path.join(fs_path, n["target_path"]), os.path.join(fs_path, n["path"]))


def main():
    parser = ArgumentParser()
    parser.add_argument("--save_path", type=str, default="data", help="Path to save loaded file")
    parser.add_argument("--processes", type=int, default=16, help="Number of processed to load file[s]")
    args = parser.parse_args()
    save_path: str = args.save_path

    config: dict[str, str | int] = nirvana_dl.params()

    numerical_features_transform_type = config["numerical_features_transform"]
    dataset_type = config["dataset"]

    if dataset_type == "music":
        yt_datadir: str = "//home/yr/fvelikon/graph_time_series/datasets/music"
    elif dataset_type == "music_truncated":
        yt_datadir: str = "//home/yr/fvelikon/graph_time_series/datasets/music_truncated"
    else:
        print(f"For dataset {dataset_type} no additional data is needed. Quitting.")
        return 0

    if numerical_features_transform_type not in {"min-max-scaler", "standard-scaler"}:
        print(f"For dataset {dataset_type} preprocessed spatiotemporal features are available only for {set('min-max-scaler', 'standard-scaler')} options. Setting to default 'standard-scaler'")
        numerical_features_transform_type = "standard-scaler"

    if numerical_features_transform_type == "min-max-scaler":
        remote_file_name = "min_max_scaled.memmap"        
    elif numerical_features_transform_type == "standard-scaler":
        remote_file_name = "standard_scaled.memmap"

    remote_path = f"{yt_datadir}/{remote_file_name}"
    print(f"Loading {remote_path}")

    download_dir_from_yt_fast(yt_cluster="hahn", yt_path=yt_datadir, fs_path=save_path, files_to_load=[remote_file_name], num_processes=args.processes)
    return 0

if __name__ == '__main__':
    main()