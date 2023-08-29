"""Wrapper for Cpplint"""

import os
import pathlib
import subprocess
import sys

ROOT_PATH = pathlib.Path(__file__).parent.parent.expanduser().resolve()


def main() -> None:
    """Wrapper for Pylint."""

    options = [
        "--linelength=100",
        "--recursive",
        ",".join(
            [
                "--filter=-build/c++11",
                "-build/include",
                "-build/namespaces_literals",
                "-runtime/references",
                "-build/include_order",
                "+build/include_what_you_use",
            ]
        ),
        "--root=include",
    ]

    # sys.argv[1:]: List of source files to check
    subprocess.run(
        ["cpplint"] + options + sys.argv[1:],
        check=True,
        env=os.environ.copy(),
    )


if __name__ == "__main__":
    main()
