"""CSE 258 Assignment 1 package.

CLI (from pyproject [project.scripts]):
    uv run assignment-1              # run every implemented task -> dist/
    uv run assignment-1 1 3          # run only tasks 1 and 3
    uv run assignment-1 --data-dir /path/to/data --out-dir dist
"""

import argparse
import importlib
import pkgutil


def available_tasks():
    """Task numbers N for which src/assignment_1/tasks/taskN.py exists."""
    from assignment_1.modules import tasks

    return sorted(
        int(m.name.removeprefix("task"))
        for m in pkgutil.iter_modules(tasks.__path__)
        if m.name.removeprefix("task").isdigit()
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="assignment-1", description="Generate task prediction files."
    )
    parser.add_argument(
        "tasks", nargs="*", type=int, help="task numbers (default: all implemented)"
    )
    parser.add_argument(
        "--data-dir", default=None, help="default: $A1_DATA_DIR or ./data"
    )
    parser.add_argument("--out-dir", default="dist")
    args = parser.parse_args(argv)

    for n in args.tasks or available_tasks():
        module = importlib.import_module(f"assignment_1.tasks.task{n}")
        out = getattr(module, f"run_task{n}")(args.data_dir, args.out_dir)
        print(f"Task {n} -> {out}")
