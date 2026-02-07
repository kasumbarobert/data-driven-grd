"""Batch driver that regenerates every manuscript figure module."""

from __future__ import annotations

import argparse
import importlib
import pkgutil
import re
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PACKAGE_DIR = Path(__file__).resolve().parent
PACKAGE_NAME = PACKAGE_DIR.name
FIGURE_PREFIX = "figure_"


@dataclass
class RunResult:
    module_name: str
    figure_id: str
    status: str
    detail: Optional[str] = None


def _sort_key(module_name: str) -> tuple[int, str]:
    match = re.match(rf"{FIGURE_PREFIX}(\\d+)(.*)", module_name)
    if match:
        number = int(match.group(1))
        suffix = match.group(2)
    else:
        number = sys.maxsize
        suffix = module_name
    return number, suffix


def _figure_id(module_name: str) -> str:
    return module_name[len(FIGURE_PREFIX) :]


def iter_figure_modules() -> List[str]:
    modules: List[str] = []
    for module_info in pkgutil.iter_modules([str(PACKAGE_DIR)]):
        name = module_info.name
        if name.startswith(FIGURE_PREFIX):
            modules.append(name)
    modules.sort(key=_sort_key)
    return modules


def resolve_requested(all_modules: Sequence[str], requested: Sequence[str]) -> List[str]:
    if not requested:
        return list(all_modules)

    module_lookup = {module.lower(): module for module in all_modules}
    figure_id_lookup = {_figure_id(module).lower(): module for module in all_modules}

    resolved: List[str] = []
    missing: List[str] = []
    for raw in requested:
        key = raw.lower()
        module_name: Optional[str] = None

        if key in module_lookup:
            module_name = module_lookup[key]
        elif key.startswith(FIGURE_PREFIX):
            module_name = module_lookup.get(key)
            if module_name is None:
                module_name = figure_id_lookup.get(key[len(FIGURE_PREFIX) :])
        else:
            module_name = figure_id_lookup.get(key)

        if module_name is None:
            missing.append(raw)
        elif module_name not in resolved:
            resolved.append(module_name)

    if missing:
        raise ValueError(f"Unknown figure identifiers: {', '.join(missing)}")

    return resolved


def run_module(module_name: str) -> RunResult:
    figure_id = _figure_id(module_name)
    dotted_name = f"{PACKAGE_NAME}.{module_name}"
    try:
        module = importlib.import_module(dotted_name)
    except Exception as exc:  # noqa: BLE001
        detail = "".join(traceback.format_exception(exc))
        return RunResult(module_name, figure_id, "error", detail)

    main_callable = getattr(module, "main", None)
    if main_callable is None:
        return RunResult(module_name, figure_id, "missing-main", "Module has no main() entry point.")

    try:
        main_callable()
    except NotImplementedError as exc:
        detail = str(exc) or "Figure generation not implemented."
        return RunResult(module_name, figure_id, "not-implemented", detail)
    except Exception as exc:  # noqa: BLE001
        detail = "".join(traceback.format_exception(exc))
        return RunResult(module_name, figure_id, "error", detail)

    return RunResult(module_name, figure_id, "ok")


def print_summary(results: Sequence[RunResult]) -> None:
    if not results:
        print("No figures selected.")
        return

    id_width = max(len(result.figure_id) for result in results)
    status_width = max(len(result.status) for result in results)

    print("\nSummary:")
    for result in results:
        detail = f" - {result.detail.splitlines()[0]}" if result.detail else ""
        print(f"  Figure {result.figure_id:<{id_width}}  {result.status.upper():<{status_width}}{detail}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate every available manuscript figure.")
    parser.add_argument("figures", nargs="*", help="Optional list of figure IDs (e.g. 5a) or module names to run.")
    parser.add_argument(
        "--list",
        action="store_true",
        help="Only list available figure driver modules without executing them.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    modules = iter_figure_modules()

    if args.list:
        print("Available figure drivers:")
        for module_name in modules:
            print(f"  {module_name} (Figure {_figure_id(module_name)})")
        return 0

    try:
        modules_to_run = resolve_requested(modules, args.figures)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2

    results: List[RunResult] = []
    for module_name in modules_to_run:
        figure_id = _figure_id(module_name)
        print(f"=== Figure {figure_id} ({module_name}) ===", flush=True)
        result = run_module(module_name)
        results.append(result)

        if result.status == "ok":
            print(f"Figure {figure_id} generated successfully.\n", flush=True)
        elif result.status == "not-implemented":
            print(f"Skipped Figure {figure_id}: {result.detail}\n", flush=True)
        elif result.status == "missing-main":
            print(f"Skipped Figure {figure_id}: {result.detail}\n", flush=True)
        else:
            print(f"Error while generating Figure {figure_id}:")
            if result.detail:
                print(result.detail.rstrip())
            print()

    print_summary(results)

    has_errors = any(result.status == "error" for result in results)
    return 1 if has_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
