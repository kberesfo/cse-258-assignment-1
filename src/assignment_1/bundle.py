"""Bundle src/assignment_1/modules into a single dist/assignment1.py for submission.

Order: shared modules (data, metrics, models/*) then tasks/task1..6, each under a section header.
- Third-party/stdlib imports are hoisted to the top and de-duplicated.
- Internal imports (`from assignment_1.x import y`) are dropped, since everything is in one file.
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
ROOT = Path(__file__).resolve().parents[3]
# package root (cli.py, __main__.py: not bundled)
SRC_PKG = ROOT / "src" / "assignment_1"
# everything bundled lives here
PKG = SRC_PKG / "modules"
PKG_NAME = "assignment_1"


def internal_deps(path):
    """Files under modules/ that `path` imports from (via `from assignment_1.modules.x import y`)."""
    deps = []
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
            PKG_NAME + "."
        ):
            target = SRC_PKG.joinpath(*node.module.split(".")[1:]).with_suffix(".py")
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
            sys.exit(f"bundle: circular import involving {p.relative_to(ROOT)}")
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
    shared = [PKG / "data.py", PKG / "metrics.py"]
    for p in shared + [PKG / "models", PKG / "tasks"]:
        if not p.exists():
            sys.exit(
                f"bundle: expected {p.relative_to(ROOT)} — package layout changed?"
            )
    models = sorted(
        p for p in (PKG / "models").rglob("*.py") if p.name != "__init__.py"
    )
    tasks = sorted(
        (p for p in (PKG / "tasks").glob("task*.py") if p.stem[4:].isdigit()),
        key=lambda p: int(p.stem[4:]),
    )
    if not tasks:
        sys.exit(
            f"bundle: no tasks/taskN.py found in {(PKG / 'tasks').relative_to(ROOT)}"
        )
    library = dependency_order(shared + models)
    dependency_order(library + tasks)  # validates tasks only import bundled modules
    return library, tasks


def is_internal(node):
    if isinstance(node, ast.ImportFrom):
        return node.level > 0 or (node.module or "").split(".")[0] == PKG_NAME
    return any(a.name.split(".")[0] == PKG_NAME for a in node.names)


def check_internal_import(node, path):
    where = f"{path.relative_to(ROOT)}:{node.lineno}"
    if isinstance(node, ast.Import) or node.level > 0:
        sys.exit(
            f"bundle: {where}: use absolute `from {PKG_NAME}.module import name` imports"
        )
    for alias in node.names:
        if alias.asname:
            sys.exit(
                f"bundle: {where}: don't rename internal imports (`as {alias.asname}`)"
            )
        mod_path = SRC_PKG.joinpath(*node.module.split(".")[1:], alias.name)
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
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
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
        Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "dist" / "assignment1.py"
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
        + "    # python assignment1.py [data_dir] [out_dir]\n"
        + '    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data"\n'
        + '    out_dir = sys.argv[2] if len(sys.argv) > 2 else "."\n'
        + f"    for run in ({', '.join(run_fns)},):\n"
        + "        print(run(data_dir, out_dir))\n"
    )
    text = (
        '"""CSE 258 Assignment 1 — all tasks in one file.\n\n'
        "Auto-generated by scripts/bundle.py from src/assignment_1/modules/; edit the package, not this file.\n"
        "Run: python assignment1.py [data_dir] [out_dir]\n"
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
