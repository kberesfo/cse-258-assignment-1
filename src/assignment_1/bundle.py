"""Bundle src/assignment_1/modules into a single dist/assignment1.py for submission.

Order: shared modules (data, metrics, models/*) then tasks/task1..6, each under a section header.
- Third-party/stdlib imports are hoisted to the top and de-duplicated.
- Internal imports, absolute (`from assignment_1.modules.x import y`) or relative (`from .x import y`),
  are dropped, since everything is in one file.
- Fails loudly on anything concatenation would silently break:
    * two modules defining the same top-level name
    * internal imports that rename (`as`) or import a module object (`from assignment_1 import data`)
- Appends a __main__ block that runs every run_taskN(data_dir, out_dir).

Usage: python scripts/bundle.py [out_path]
"""

import ast
import sys
from pathlib import Path

# src/assignment_1/bundle.py -> repo root
ROOT = Path(__file__).resolve().parents[2]
SRC_PKG = (
    ROOT / "src" / "assignment_1"
)  # package root (cli.py, __main__.py: not bundled)
PKG = SRC_PKG / "modules"  # everything bundled lives here
PKG_NAME = "assignment_1"


def import_target(node, path):
    """Path (no suffix) of the package module an internal `from X import ...` refers to.

    Handles absolute (`from assignment_1.modules.util.data import f`) and relative
    (`from .data import f`, `from ..models.bpr import g`) imports.
    """
    if node.level > 0:
        base = path.parent
        for _ in range(node.level - 1):
            base = base.parent
        return base.joinpath(*node.module.split(".")) if node.module else base
    return SRC_PKG.joinpath(*node.module.split(".")[1:])


def internal_deps(path):
    """Files under modules/ that `path` imports from (via `from assignment_1.modules.x import y`)."""
    deps = []
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.ImportFrom) and is_internal(node):
            target = import_target(node, path).with_suffix(".py")
            if target.exists() and target != path:
                deps.append(target)
    return deps


def dependency_order(paths):
    """Order files so every module comes after the modules it imports (stable otherwise)."""
    ordered, visiting = [], set()

    def visit(p):
        if p in ordered:
            return
        if p in visiting:
            sys.exit(
                f"bundle: circular import involving {p.relative_to(ROOT)}")
        visiting.add(p)
        for d in internal_deps(p):
            if d not in paths:
                sys.exit(
                    f"bundle: {p.relative_to(ROOT)} imports {d.relative_to(ROOT)}, which isn't bundled"
                )
            visit(d)
        visiting.discard(p)
        ordered.append(p)

    for p in paths:
        visit(p)
    return ordered


def module_order():
    tasks_dir = PKG / "tasks"
    if not tasks_dir.exists():
        sys.exit(
            f"bundle: expected {tasks_dir.relative_to(ROOT)} — package layout changed?"
        )
    # every library module under modules/ (any depth), except tasks/
    shared = sorted(
        p
        for p in PKG.rglob("*.py")
        if p.name != "__init__.py" and tasks_dir not in p.parents
    )
    tasks = sorted(
        (p for p in tasks_dir.glob("task*.py") if p.stem[4:].isdigit()),
        key=lambda p: int(p.stem[4:]),
    )
    if not tasks:
        sys.exit(
            f"bundle: no tasks/taskN.py found in {tasks_dir.relative_to(ROOT)}")
    library = dependency_order(shared)
    # validates tasks only import bundled modules
    dependency_order(library + tasks)
    return library, tasks


def is_internal(node):
    if isinstance(node, ast.ImportFrom):
        return node.level > 0 or (node.module or "").split(".")[0] == PKG_NAME
    return any(a.name.split(".")[0] == PKG_NAME for a in node.names)


def check_internal_import(node, path):
    where = f"{path.relative_to(ROOT)}:{node.lineno}"
    if isinstance(node, ast.Import):
        sys.exit(
            f"bundle: {where}: use `from module import name`, not `import {PKG_NAME}...`"
        )
    for alias in node.names:
        if alias.asname:
            sys.exit(
                f"bundle: {where}: don't rename internal imports (`as {alias.asname}`)"
            )
        mod_path = import_target(node, path) / alias.name
        if mod_path.with_suffix(".py").exists() or mod_path.is_dir():
            sys.exit(
                f"bundle: {where}: import names from modules, not modules (`{alias.name}`)"
            )


def is_main_guard(node):
    return (
        isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "__name__"
    )


def defined_names(node):
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return [node.name]
    if isinstance(node, (ast.Assign, ast.AnnAssign)):
        targets = node.targets if isinstance(
            node, ast.Assign) else [node.target]
        return [n.id for t in targets for n in ast.walk(t) if isinstance(n, ast.Name)]
    return []


def process(path, imports, owners):
    src = path.read_text()
    tree = ast.parse(src)
    lines = src.splitlines()
    drop = set()
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if is_internal(node):
                check_internal_import(node, path)
            else:
                stmt = ast.get_source_segment(src, node)
                if stmt not in imports:
                    imports.append(stmt)
            drop.update(range(node.lineno - 1, node.end_lineno))
        elif is_main_guard(node):
            drop.update(range(node.lineno - 1, node.end_lineno))
        for name in defined_names(node):
            if name in owners and owners[name] != path:
                sys.exit(
                    f"bundle: name `{name}` defined in both {owners[name].relative_to(ROOT)} "
                    f"and {path.relative_to(ROOT)} — rename one"
                )
            owners[name] = path
    body = "\n".join(line for i, line in enumerate(lines) if i not in drop)
    return body.strip("\n")


def header(title):
    bar = "# " + "=" * 76
    return f"\n\n{bar}\n# {title}\n{bar}\n"


def main():
    out = (
        Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT /
        "dist" / "assignment1.py"
    ).resolve()
    shared, tasks = module_order()
    imports, owners, sections = [], {}, []

    for p in shared:
        sections.append(
            header(f"Shared code: {p.relative_to(PKG)}")
            + "\n"
            + process(p, imports, owners)
        )
    run_fns = []
    for p in tasks:
        n = int(p.stem[4:])
        doc = ast.get_docstring(ast.parse(p.read_text())) or f"Task {n}"
        sections.append(
            header(doc.splitlines()[0]) + "\n" + process(p, imports, owners)
        )
        if f"run_task{n}" not in owners:
            sys.exit(
                f"bundle: {p.relative_to(ROOT)} must define run_task{n}(data_dir, out_dir)"
            )
        run_fns.append(f"run_task{n}")

    for name in owners:
        for stmt in imports:
            if name in {
                a.asname or a.name.split(".")[0] for a in ast.parse(stmt).body[0].names
            }:
                sys.exit(
                    f"bundle: `{name}` is both defined in the package and imported by `{stmt}`"
                )

    main_block = (
        header("Entry point")
        + '\nif __name__ == "__main__":\n'
        + "    import sys\n\n"
        + "    # python assignment1.py [data_dir] [out_dir] [task ...]  (no tasks: all)\n"
        + '    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data"\n'
        + '    out_dir = sys.argv[2] if len(sys.argv) > 2 else "."\n'
        + "    selected = {int(a) for a in sys.argv[3:]}\n"
        + "    tasks = {"
        + ", ".join(f"{f.removeprefix('run_task')}: {f}" for f in run_fns)
        + "}\n"
        + "    unknown = selected - tasks.keys()\n"
        + "    if unknown:\n"
        + '        sys.exit(f"unknown task(s) {sorted(unknown)}; have {sorted(tasks)}")\n'
        + "    for n, run in tasks.items():\n"
        + "        if not selected or n in selected:\n"
        + "            print(run(data_dir, out_dir))\n"
    )
    text = (
        '"""CSE 258 Assignment 1 — all tasks in one file.\n\n'
        "Auto-generated by scripts/bundle.py from src/assignment_1/modules/; edit the package, not this file.\n"
        "Run: python assignment1.py [data_dir] [out_dir] [task ...]\n"
        '"""\n\n' + "\n".join(imports) + "".join(sections) + main_block
    )
    compile(text, str(out), "exec")  # syntax check before writing
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(
        f"bundle: wrote {out.relative_to(ROOT)} ({len(tasks)} task(s): {', '.join(run_fns)})"
    )


if __name__ == "__main__":
    main()
