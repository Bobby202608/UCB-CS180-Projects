import sys
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from typing import Iterator, TextIO


class _Tee:
    def __init__(self, terminal: TextIO, log_file: TextIO) -> None:
        self.terminal = terminal
        self.log_file = log_file

    def write(self, text: str) -> int:
        self.terminal.write(text)
        self.log_file.write(text)
        return len(text)

    def flush(self) -> None:
        self.terminal.flush()
        self.log_file.flush()


@contextmanager
def print_logs(log_path: str | Path) -> Iterator[None]:
    """
        Copy standard output to a .log file while preserving terminal output.
    """

    with open(log_path, "a", encoding="utf-8", buffering=1) as log_file:
        with redirect_stdout(_Tee(sys.stdout, log_file)):
            yield