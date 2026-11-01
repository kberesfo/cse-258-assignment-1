"""Command-line entry point: generate task prediction files.

uv run assignment-1              # every implemented task -> dist/
uv run assignment-1 1 3          # only tasks 1 and 3
uv run assignment-1 --data-dir data --out-dir dist
python -m assignment_1 ...       # same, via __main__.py
"""

import argparse
import importlib
import pkgutil

TASKS_PACKAGE = "assignment_1.modules.tasks"


def available_tasks() -> list[int]:
    """Task numbers N for which modules/tasks/taskN.py exists."""
    tasks = importlib.import_module(TASKS_PACKAGE)
    return sorted(
        int(m.name.removeprefix("task"))
        for m in pkgutil.iter_modules(tasks.__path__)
        if m.name.removeprefix("task").isdigit()
    )


def main(argv: list[str] | None = None) -> None:
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

    implemented = available_tasks()
    for n in args.tasks or implemented:
        if n not in implemented:
            parser.error(f"task {n} not implemented (have: {implemented})")
        module = importlib.import_module(f"{TASKS_PACKAGE}.task{n}")
        out = getattr(module, f"run_task{n}")(args.data_dir, args.out_dir)
        print(f"Task {n} -> {out}")
