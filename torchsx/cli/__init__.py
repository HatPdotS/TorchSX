from pathlib import Path

from torchsx._core import abspath_str


def path_arg(value: str) -> Path:
    """argparse ``type`` for paths: ``~`` expanded, absolute against the working
    directory, so ``.``, ``..`` and a ``~`` the shell left alone (``--out=~/run``)
    name what they mean."""
    return Path(abspath_str(value))
