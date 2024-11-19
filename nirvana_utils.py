# Nirvana dependencies
import yt.wrapper as yt

from typing import List, Any, Dict

try:
    import nirvana_dl
    from distutils.dir_util import copy_tree
    import os
except ImportError:
    nirvana_dl = None


def copy_snapshot_to_out(out):
    """The preempted run transfers its "state" to the restarted run through "snapshot path".
    "state" is a tar-archive that contains all files put into "snapshot path" by the preempted run.
    This function moves the files in the "state" archive to you local "out" dir.
    """
    out = str(out)

    if nirvana_dl:
        snapshot_path = nirvana_dl.snapshot.get_snapshot_path()
        print(f"Copy the previous state from {snapshot_path} to {out}")
        copy_tree(snapshot_path, out)
        # os.system(f"tar -xf {out}/state -C {out}/")


def copy_out_to_snapshot(out, dump=True):
    """This function copies all files in the local "out" directory to "snapshot path".
    dump: If True, put these files into tar-archive "state" and
          send it to the Python DL output.
    """
    out = str(out)

    if nirvana_dl:
        snapshot_path = nirvana_dl.snapshot.get_snapshot_path()
        print(f"Copy {out} to the snapshot path: {snapshot_path}")

        # Delete previous state to avoid memory explosion
        os.system(f"rm {snapshot_path}/state")
        copy_tree(out, snapshot_path)

        if dump:
            # Make it visible in the Python DL output
            nirvana_dl.snapshot.dump_snapshot(snapshot_path)


def write_output_to_YT(
    output: List[dict[str, Any]],
    table_path_root: str,  # default is `//home/tmp/` in utils.Config
    yt_client: yt.YtClient,
    proxy: str = "hahn",
) -> dict[str, str]:
    """Function to write output to YT
    :param: output - List of dicts - each dict for each prediction row
    :param: table_path_root - rot for temporary table path
    :param: proxy - proxy for YT
    :returns: MR Table output
    """

    def generate_random_name_with_path(k: int = 20) -> str:
        """
        Generates random name for tmp table with defined root
        :param: k - length for the random part of a name
        :returns: path to a random name
        """
        from string import ascii_lowercase
        from random import choices

        random_name: str = f"__tmp_table_{''.join(choices(ascii_lowercase, k=k))}"

        table_path_with_random_name: str = os.path.join(table_path_root, random_name)

        return table_path_with_random_name

    @yt.yt_dataclass
    class Row:
        # TODO add user_id for more general approach
        predict: float
        target: float

    out_table_path: str = generate_random_name_with_path()

    table_rows: List[Row] = [
        Row(predict=row["predict"], target=row["target"]) for row in output
    ]

    print(f"Trying to save the table to {out_table_path}")

    yt.write_table_structured(out_table_path, Row, table_rows, client=yt_client)

    print(f"Table was saved to {out_table_path}")

    mr_table = dict(cluster="hahn", table=out_table_path)

    return mr_table


def read_output_from_yt(
    mr_table: Dict[str, str], yt_client: yt.YtClient
) -> List[Dict[str, float]]:
    ds_length = yt_client.get_attribute(mr_table["table"], "row_count")
    table_iterator = yt_client.read_table(table=mr_table["table"], format="json")
    rows = []
    for i, row in enumerate(table_iterator):
        if i % 100000 == 0:
            print(f"Current progress {i}/{ds_length} or {i / ds_length * 100:.2f}%")
        rows.append(row)
    return rows


def read_table_from_yt(path):
    pass  # TODO