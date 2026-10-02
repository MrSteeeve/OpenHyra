"""Concrete search over complete, finite Python programs.

The generic discovery protocol in :mod:`algorithm_discovery` deliberately
does not prescribe how candidates are produced.  This module supplies that
missing implementation.  A candidate owns all of its source files; a task may
ask an LLM-backed callback to write a new whole program, or may derive a child
through executable AST mutation and two-parent function crossover.

The search space knows nothing about Bermudan options, neural networks, or a
fixed list of representations.  Tasks provide only an entrypoint contract and
an evaluator.  Evaluation results are observed here solely to choose parents
for the next generation.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Protocol, Sequence, runtime_checkable

from algorithm_discovery import AlgorithmSpec, EvaluationResult, SearchSpace


PYTHON_PROGRAM_FAMILY = "python_program"


@dataclass(frozen=True)
class ProgramGenerationRequest:
    """One whole-program request passed to a generator callback."""

    context: Mapping[str, Any]
    slot: int
    operator: str
    parents: tuple[AlgorithmSpec, ...]
    observations: tuple[EvaluationResult, ...]
    entrypoint: str
    required_symbol: str


@runtime_checkable
class WholeProgramGenerator(Protocol):
    """LLM/tool callback that returns every source file for a candidate.

    A callback may return a plain ``{relative_path: text}`` mapping, or a
    structured mapping with a ``source`` mapping and optional ``metadata``,
    ``mechanism_id``, ``prediction`` and ``falsifier`` fields.
    """

    def __call__(self, request: ProgramGenerationRequest) -> Mapping[str, Any]:
        ...


@dataclass(frozen=True)
class _GeneratedProgram:
    source: Mapping[str, str]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    mechanism_id: str = ""
    prediction: Any = "not_observed"
    falsifier: Any = "not_observed"


def _source_from(candidate: AlgorithmSpec) -> dict[str, str]:
    implementation = candidate.implementation
    if not isinstance(implementation, Mapping):
        raise ValueError("Python program implementation must be a mapping")
    source = implementation.get("source")
    if not isinstance(source, Mapping) or not source:
        raise ValueError("Python program implementation.source must be a non-empty mapping")
    if any(not isinstance(path, str) or not isinstance(text, str) for path, text in source.items()):
        raise ValueError("Python program source must map relative paths to text")
    return dict(source)


def _source_digest(source: Mapping[str, str]) -> str:
    """Return a stable digest used to collapse duplicate search proposals."""
    digest = hashlib.sha256()
    for path, text in sorted(source.items()):
        digest.update(path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(text.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _normalise_generated(raw: Mapping[str, Any]) -> _GeneratedProgram:
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError("whole-program generator must return a non-empty mapping")
    if "source" not in raw:
        if any(not isinstance(path, str) or not isinstance(text, str) for path, text in raw.items()):
            raise ValueError("plain generator output must map source paths to text")
        return _GeneratedProgram(source=dict(raw))

    source = raw.get("source")
    if not isinstance(source, Mapping) or not source:
        raise ValueError("generator output.source must be a non-empty mapping")
    if any(not isinstance(path, str) or not isinstance(text, str) for path, text in source.items()):
        raise ValueError("generator source must map paths to text")
    metadata = raw.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise ValueError("generator metadata must be a mapping")
    return _GeneratedProgram(
        source=dict(source),
        metadata=dict(metadata),
        mechanism_id=str(raw.get("mechanism_id", "")),
        prediction=raw.get("prediction", "not_observed"),
        falsifier=raw.get("falsifier", "not_observed"),
    )


def _is_main_guard(node: ast.stmt) -> bool:
    if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
        return False
    compare = node.test
    return (
        isinstance(compare.left, ast.Name)
        and compare.left.id == "__name__"
        and len(compare.ops) == 1
        and isinstance(compare.ops[0], ast.Eq)
        and len(compare.comparators) == 1
        and isinstance(compare.comparators[0], ast.Constant)
        and compare.comparators[0].value == "__main__"
    )


def _parent_factory(
    module: ast.Module, label: str, required_symbol: str,
) -> tuple[ast.FunctionDef, ast.Assign, list[ast.ImportFrom], str, bool]:
    entrypoints = [
        node for node in module.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == required_symbol
    ]
    if not entrypoints:
        raise ValueError(f"parent entrypoint does not define {required_symbol}()")
    is_async = isinstance(entrypoints[-1], ast.AsyncFunctionDef)
    body: list[ast.stmt] = []
    future: list[ast.ImportFrom] = []
    for original in module.body:
        if _is_main_guard(original):
            continue
        if isinstance(original, ast.ImportFrom) and original.module == "__future__":
            future.append(copy.deepcopy(original))
            continue
        if isinstance(original, ast.ImportFrom) and any(alias.name == "*" for alias in original.names):
            raise ValueError("AST crossover does not support wildcard imports")
        body.append(copy.deepcopy(original))

    factory_name = f"_build_{label}"
    bound_name = f"_{label}_{required_symbol}"
    factory = ast.FunctionDef(
        name=factory_name,
        args=ast.arguments(
            posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]
        ),
        body=[*body, ast.Return(value=ast.Name(id=required_symbol, ctx=ast.Load()))],
        decorator_list=[],
    )
    binding = ast.Assign(
        targets=[ast.Name(id=bound_name, ctx=ast.Store())],
        value=ast.Call(func=ast.Name(id=factory_name, ctx=ast.Load()), args=[], keywords=[]),
    )
    return factory, binding, future, bound_name, is_async


def _deduplicate_future_imports(nodes: Sequence[ast.ImportFrom]) -> list[ast.ImportFrom]:
    seen: set[tuple[str, ...]] = set()
    output: list[ast.ImportFrom] = []
    for node in nodes:
        names = tuple(sorted(alias.name for alias in node.names))
        if names not in seen:
            seen.add(names)
            output.append(node)
    return output


def _crossover_entrypoint(
    left_text: str,
    right_text: str,
    *,
    required_symbol: str,
) -> str:
    left = ast.parse(left_text)
    right = ast.parse(right_text)
    left_factory, left_binding, left_future, left_symbol, left_async = _parent_factory(
        left, "parent_a", required_symbol
    )
    right_factory, right_binding, right_future, right_symbol, right_async = _parent_factory(
        right, "parent_b", required_symbol
    )
    if left_async != right_async:
        raise ValueError("AST crossover parents must agree on sync or async entrypoint")

    wrapper_source = f"""
def _combine_parent_results(left, right):
    if isinstance(left, bool) and isinstance(right, bool):
        return left or right
    try:
        return left / 2 + right / 2
    except (TypeError, ValueError, ZeroDivisionError):
        return right if right is not None else left

def {required_symbol}(*args, **kwargs):
    left = {left_symbol}(*args, **kwargs)
    right = {right_symbol}(*args, **kwargs)
    return _combine_parent_results(left, right)
"""
    if left_async:
        wrapper_source = f"""
def _combine_parent_results(left, right):
    if isinstance(left, bool) and isinstance(right, bool):
        return left or right
    try:
        return left / 2 + right / 2
    except (TypeError, ValueError, ZeroDivisionError):
        return right if right is not None else left

async def {required_symbol}(*args, **kwargs):
    left = await {left_symbol}(*args, **kwargs)
    right = await {right_symbol}(*args, **kwargs)
    return _combine_parent_results(left, right)
"""
    wrapper = ast.parse(wrapper_source).body
    module = ast.Module(
        body=[
            *_deduplicate_future_imports([*left_future, *right_future]),
            left_factory,
            left_binding,
            right_factory,
            right_binding,
            *wrapper,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    compile(module, "<crossover>", "exec")
    return ast.unparse(module) + "\n"


def _validate_fit_predict_parent(source: str) -> None:
    module = ast.parse(source)
    functions = {
        node.name: node
        for node in module.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    missing = {"fit", "predict"} - functions.keys()
    if missing:
        raise ValueError(
            "fit/predict crossover parent is missing: "
            + ", ".join(sorted(missing))
        )
    asynchronous = {
        name
        for name in ("fit", "predict")
        if isinstance(functions[name], ast.AsyncFunctionDef)
    }
    if asynchronous:
        raise ValueError(
            "fit/predict crossover requires synchronous CLI functions: "
            + ", ".join(sorted(asynchronous))
        )


def _is_generated_fit_predict_crossover(source: str) -> bool:
    """Recognise the two-parent dispatcher emitted by this module.

    A dispatcher embeds both complete parents as source strings. Combining two
    such dispatchers duplicates both ancestry trees and can double source size
    at every generation. The detector lets crossover distinguish that balanced
    growth from an incremental composite-plus-atomic composition.
    """

    try:
        module = ast.parse(source)
    except SyntaxError:
        return False
    assignments = {
        target.id
        for node in module.body
        if isinstance(node, ast.Assign) and len(node.targets) == 1
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    functions = {
        node.name
        for node in module.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    dispatcher_functions = {
        "_load_parent_module", "_combine_predictions", "fit", "predict",
    }
    return (
        {"_PARENT_A_SOURCE", "_PARENT_B_SOURCE"}.issubset(assignments)
        and dispatcher_functions.issubset(functions)
    )


def _crossover_fit_predict_program(
    left_text: str,
    right_text: str,
    *,
    decision_interface: bool = False,
    combiner_source: str | None = None,
) -> str:
    """Compose two programs with a fixed dispatcher and evolvable combiner."""
    _validate_fit_predict_parent(left_text)
    _validate_fit_predict_parent(right_text)
    composite_parent_count = sum(
        _is_generated_fit_predict_crossover(source)
        for source in (left_text, right_text)
    )
    if composite_parent_count == 2:
        raise ValueError(
            "fit/predict crossover will not embed two existing two-parent "
            "dispatchers because that recursively doubles both source trees; "
            "consolidate one parent with a whole-program rewrite or pair the "
            "composite with an atomic parent"
        )
    combiner_source = combiner_source or (
        "def _combine_predictions(left, right):\n"
        "    return _openhyra_numpy.logical_or(left, right)\n"
        if decision_interface
        else (
            "def _combine_predictions(left, right):\n"
            "    if left.dtype == _openhyra_numpy.bool_ and right.dtype == _openhyra_numpy.bool_:\n"
            "        return _openhyra_numpy.logical_or(left, right)\n"
            "    return (\n"
            "        _openhyra_numpy.asarray(left, dtype=_openhyra_numpy.float64) / 2.0\n"
            "        + _openhyra_numpy.asarray(right, dtype=_openhyra_numpy.float64) / 2.0\n"
            "    )\n"
        )
    )
    combiner_source = _extract_crossover_combiner(combiner_source)
    child_source = f"""
import argparse as _openhyra_argparse
from pathlib import Path as _OpenHyraPath
import shutil as _openhyra_shutil
import sys as _openhyra_sys
import types as _openhyra_types
import numpy as _openhyra_numpy

_PARENT_A_SOURCE = {left_text!r}
_PARENT_B_SOURCE = {right_text!r}

def _load_parent_module(name, source):
    module = _openhyra_types.ModuleType(name)
    module.__file__ = name + ".py"
    module.__package__ = name.rpartition(".")[0]
    _openhyra_sys.modules[name] = module
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module

_PARENT_A_MODULE = "_openhyra_parent_a"
_PARENT_B_MODULE = "_openhyra_parent_b"
_parent_namespace = globals().get("__name__", "_openhyra_composite")
_parent_a = _load_parent_module(
    _parent_namespace + "." + _PARENT_A_MODULE, _PARENT_A_SOURCE
)
_parent_b = _load_parent_module(
    _parent_namespace + "." + _PARENT_B_MODULE, _PARENT_B_SOURCE
)
_parent_a_fit = _parent_a.fit
_parent_b_fit = _parent_b.fit
_parent_a_predict = _parent_a.predict
_parent_b_predict = _parent_b.predict

{combiner_source.rstrip()}

def fit(input_dir, output_dir, seed):
    output_root = _OpenHyraPath(output_dir)
    left_model = output_root / "parent_a"
    right_model = output_root / "parent_b"
    left_model.mkdir(parents=True, exist_ok=True)
    right_model.mkdir(parents=True, exist_ok=True)
    _parent_a_fit(_OpenHyraPath(input_dir), left_model, seed)
    _parent_b_fit(_OpenHyraPath(input_dir), right_model, seed)

def predict(model_dir, input_dir, output_dir):
    model_root = _OpenHyraPath(model_dir)
    output_root = _OpenHyraPath(output_dir)
    scratch = output_root / "_parent_predictions"
    left_output = scratch / "parent_a"
    right_output = scratch / "parent_b"
    left_output.mkdir(parents=True, exist_ok=True)
    right_output.mkdir(parents=True, exist_ok=True)
    _parent_a_predict(model_root / "parent_a", _OpenHyraPath(input_dir), left_output)
    _parent_b_predict(model_root / "parent_b", _OpenHyraPath(input_dir), right_output)
    left = _openhyra_numpy.load(left_output / "predictions.npy", allow_pickle=False)
    right = _openhyra_numpy.load(right_output / "predictions.npy", allow_pickle=False)
    if left.shape != right.shape:
        raise ValueError("parent prediction shapes differ")
    combined = _combine_predictions(left, right)
    _openhyra_shutil.rmtree(scratch)
    _openhyra_numpy.save(
        output_root / "predictions.npy", combined, allow_pickle=False
    )

def main():
    parser = _openhyra_argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    fit_parser = commands.add_parser("fit")
    fit_parser.add_argument("--input", required=True)
    fit_parser.add_argument("--output", required=True)
    fit_parser.add_argument("--seed", required=True, type=int)
    predict_parser = commands.add_parser("predict")
    predict_parser.add_argument("--model", required=True)
    predict_parser.add_argument("--input", required=True)
    predict_parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "fit":
        fit(_OpenHyraPath(args.input), _OpenHyraPath(args.output), args.seed)
    else:
        predict(
            _OpenHyraPath(args.model),
            _OpenHyraPath(args.input),
            _OpenHyraPath(args.output),
        )

if __name__ == "__main__":
    main()
"""
    compile(child_source, "<fit-predict-crossover>", "exec")
    return child_source


def _extract_crossover_combiner(source: str) -> str:
    """Extract the self-contained function that a crossover proposal evolves."""
    module = ast.parse(source)
    functions = [
        node
        for node in module.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "_combine_predictions"
    ]
    if len(functions) != 1 or isinstance(functions[0], ast.AsyncFunctionDef):
        raise ValueError(
            "crossover program must define one synchronous "
            "_combine_predictions(left, right) function"
        )
    function = functions[0]
    positional = list(function.args.posonlyargs) + list(function.args.args)
    if (
        [argument.arg for argument in positional] != ["left", "right"]
        or function.args.vararg is not None
        or function.args.kwarg is not None
        or function.args.kwonlyargs
    ):
        raise ValueError(
            "_combine_predictions must accept exactly (left, right)"
        )
    return ast.unparse(function) + "\n"


def _mutate_tree(
    module: ast.Module,
    slot: int,
    *,
    protected_functions: Sequence[str] = ("main",),
    mutation_kind: str | None = None,
) -> str:
    """Apply one deterministic mutation inside the algorithmic function graph.

    Command dispatch and the ``__main__`` guard are protocol plumbing, not an
    algorithm hypothesis. Keeping them outside the mutation roots prevents a
    syntactically valid child from silently skipping ``fit`` or ``predict``.
    """

    protocol_names = {
        "argparse", "args", "input_dir", "model_dir", "output_dir", "Path",
    }
    io_attributes = {
        "add_argument", "add_parser", "add_subparsers", "load", "mkdir",
        "open", "parse_args", "read_bytes", "read_text", "save", "savez",
        "write_bytes", "write_text",
    }

    def is_protocol_or_io_expression(node: ast.AST) -> bool:
        for descendant in ast.walk(node):
            if isinstance(descendant, ast.Name) and descendant.id in protocol_names:
                return True
            if (
                isinstance(descendant, ast.Attribute)
                and descendant.attr in io_attributes
            ):
                return True
            if (
                isinstance(descendant, ast.Constant)
                and isinstance(descendant.value, str)
            ):
                return True
        return False

    protected = set(protected_functions)
    roots: list[ast.AST] = [
        node
        for node in module.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name not in protected
    ]
    if not roots:
        roots = [node for node in module.body if not _is_main_guard(node)]

    def is_state_update(statement: ast.stmt) -> bool:
        if is_protocol_or_io_expression(statement):
            return False
        if isinstance(statement, ast.AugAssign):
            return True
        if isinstance(statement, (ast.Assign, ast.AnnAssign)):
            targets = (
                statement.targets
                if isinstance(statement, ast.Assign)
                else [statement.target]
            )
            target_names = {
                node.id
                for target in targets
                for node in ast.walk(target)
                if isinstance(node, ast.Name)
            }
            value = statement.value
            if value is None:
                return False
            loaded_names = {
                node.id
                for node in ast.walk(value)
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
            }
            return bool(target_names.intersection(loaded_names))
        return False

    def statement_lists(node: ast.AST):
        for _field, value in ast.iter_fields(node):
            if isinstance(value, list):
                if value and all(isinstance(item, ast.stmt) for item in value):
                    yield value
                for child in value:
                    if isinstance(child, ast.AST):
                        yield from statement_lists(child)
            elif isinstance(value, ast.AST):
                yield from statement_lists(value)

    sites: dict[str, list[Any]] = {
        "binary_operator": [],
        "branch_expansion": [],
        "branch_swap": [],
        "loop_update_repeat": [],
        "state_update_repeat": [],
        "comparison_operator": [],
        "boolean_operator": [],
        "branch_condition": [],
        "numeric_literal": [],
        # The original mutation families above are deliberately retained in
        # their historical order for replay compatibility.  These operators
        # add a structural neighbourhood over typed expression subtrees and
        # statements, rather than another fixed numeric tweak.
        "subtree_swap": [],
        "function_extract": [],
        "control_flow_guard": [],
    }
    for root in roots:
        for node in ast.walk(root):
            if isinstance(node, ast.BinOp) and not is_protocol_or_io_expression(node):
                sites["binary_operator"].append(node)
            elif (
                isinstance(node, ast.Compare)
                and node.ops
                and not is_protocol_or_io_expression(node)
            ):
                sites["comparison_operator"].append(node)
            elif isinstance(node, ast.BoolOp) and not is_protocol_or_io_expression(node):
                sites["boolean_operator"].append(node)
            elif isinstance(node, ast.If) and not is_protocol_or_io_expression(node.test):
                sites["branch_condition"].append(node)
                sites[
                    "branch_swap" if node.orelse else "branch_expansion"
                ].append(node)
            elif (
                isinstance(node, ast.Constant)
                and isinstance(node.value, (int, float))
                and not isinstance(node.value, bool)
                and not is_protocol_or_io_expression(node)
            ):
                sites["numeric_literal"].append(node)

        # Swap complete expression subtrees only with another subtree of the
        # same AST class.  This preserves the expression/value kind (and hence
        # Python's compile-time contract) while allowing a search to move
        # across independently generated algorithmic fragments.
        expression_groups: dict[tuple[type[ast.expr], tuple[str, ...]], list[ast.expr]] = {}
        for node in ast.walk(root):
            if (
                isinstance(node, ast.expr)
                and not isinstance(node, (ast.Name, ast.Load, ast.Store, ast.Del))
                and not is_protocol_or_io_expression(node)
            ):
                loaded_names = tuple(sorted({
                    descendant.id
                    for descendant in ast.walk(node)
                    if isinstance(descendant, ast.Name)
                    and isinstance(descendant.ctx, ast.Load)
                }))
                expression_groups.setdefault((type(node), loaded_names), []).append(node)
        for group in expression_groups.values():
            if len(group) >= 2:
                for index in range(len(group) - 1):
                    sites["subtree_swap"].append((group[index], group[index + 1]))

        # Extract an expression into a helper function whose arguments are the
        # names it reads.  This is a genuine function-level mutation: the
        # helper can subsequently be rewritten or crossed over independently,
        # while the generated call remains syntactically and lexically safe.
        for node in ast.walk(root):
            if not isinstance(node, ast.expr) or is_protocol_or_io_expression(node):
                continue
            if isinstance(
                node,
                (
                    ast.Lambda,
                    ast.ListComp,
                    ast.SetComp,
                    ast.DictComp,
                    ast.GeneratorExp,
                    ast.Await,
                    ast.Yield,
                    ast.YieldFrom,
                    ast.NamedExpr,
                ),
            ):
                continue
            if isinstance(node, (ast.Name, ast.Constant)):
                continue
            loaded = []
            seen_loaded: set[str] = set()
            for descendant in ast.walk(node):
                if isinstance(descendant, ast.Name) and isinstance(descendant.ctx, ast.Load):
                    if descendant.id not in seen_loaded and descendant.id.isidentifier():
                        loaded.append(descendant.id)
                        seen_loaded.add(descendant.id)
            sites["function_extract"].append((node, tuple(loaded)))

        # Wrap executable statements in a real control-flow node.  The guard
        # condition is chosen at application time (True/False by slot), making
        # this a bounded branch mutation rather than a source-only annotation.
        for statements in statement_lists(root):
            for index, statement in enumerate(statements):
                if (
                    isinstance(statement, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                    or is_protocol_or_io_expression(statement)
                    or isinstance(statement, ast.If) and _is_main_guard(statement)
                ):
                    continue
                sites["control_flow_guard"].append((statements, index, statement))

        loop_updates: set[int] = set()
        for loop in (
            node for node in ast.walk(root)
            if isinstance(node, (ast.For, ast.While))
        ):
            for index, statement in enumerate(loop.body):
                if is_state_update(statement):
                    sites["loop_update_repeat"].append(
                        (loop.body, index, statement)
                    )
                    loop_updates.add(id(statement))
        for statements in statement_lists(root):
            for index, statement in enumerate(statements):
                if id(statement) not in loop_updates and is_state_update(statement):
                    sites["state_update_repeat"].append(
                        (statements, index, statement)
                    )

    strategy_order = (
        "binary_operator",
        "branch_expansion",
        "branch_swap",
        "loop_update_repeat",
        "state_update_repeat",
        "comparison_operator",
        "boolean_operator",
        "branch_condition",
        "numeric_literal",
        "subtree_swap",
        "function_extract",
        "control_flow_guard",
    )
    available = [name for name in strategy_order if sites[name]]
    if mutation_kind is not None:
        if mutation_kind not in sites:
            raise ValueError(f"unknown AST mutation kind: {mutation_kind}")
        if not sites[mutation_kind]:
            raise ValueError(
                f"program has no site for AST mutation kind {mutation_kind}"
            )
        name = mutation_kind
        site_index = int(slot) % len(sites[name])
    elif available:
        name = available[int(slot) % len(available)]
        site_index = (int(slot) // len(available)) % len(sites[name])
    else:
        name = ""
        site_index = 0

    if name:
        target = sites[name][site_index]
        if name == "binary_operator":
            assert isinstance(target, ast.BinOp)
            replacements: list[tuple[type[ast.operator], type[ast.operator]]] = [
                (ast.Add, ast.Sub),
                (ast.Sub, ast.Add),
                (ast.Mult, ast.Div),
                (ast.Div, ast.Mult),
                (ast.Pow, ast.Mult),
                (ast.Mod, ast.Add),
            ]
            for before, after in replacements:
                if isinstance(target.op, before):
                    target.op = after()
                    break
            else:
                target.op = ast.Add()
        elif name == "comparison_operator":
            assert isinstance(target, ast.Compare)
            replacements: list[tuple[type[ast.cmpop], type[ast.cmpop]]] = [
                (ast.Gt, ast.Lt),
                (ast.Lt, ast.Gt),
                (ast.GtE, ast.LtE),
                (ast.LtE, ast.GtE),
                (ast.Eq, ast.NotEq),
                (ast.NotEq, ast.Eq),
                (ast.In, ast.NotIn),
                (ast.NotIn, ast.In),
            ]
            first = target.ops[0]
            for before, after in replacements:
                if isinstance(first, before):
                    target.ops[0] = after()
                    break
            else:
                target.ops[0] = ast.NotEq()
        elif name == "boolean_operator":
            assert isinstance(target, ast.BoolOp)
            target.op = ast.Or() if isinstance(target.op, ast.And) else ast.And()
        elif name == "branch_condition":
            assert isinstance(target, ast.If)
            target.test = ast.UnaryOp(op=ast.Not(), operand=target.test)
        elif name == "numeric_literal":
            assert isinstance(target, ast.Constant)
            target.value = target.value + (1 if isinstance(target.value, int) else 0.5)
        elif name == "subtree_swap":
            left, right = target
            if type(left) is not type(right):
                raise ValueError("subtree_swap requires equal AST expression types")
            # Exchange fields in place so parent pointers (implicit in the
            # Python AST) remain valid without a fragile NodeTransformer.
            left_fields = {field: copy.deepcopy(getattr(left, field)) for field in left._fields}
            right_fields = {field: copy.deepcopy(getattr(right, field)) for field in right._fields}
            for field, value in right_fields.items():
                setattr(left, field, value)
            for field, value in left_fields.items():
                setattr(right, field, value)
        elif name == "function_extract":
            expression, loaded_names = target
            assert isinstance(expression, ast.expr)
            helper_name = f"_openhyra_extracted_{int(slot)}"
            existing_names = {
                node.name
                for node in module.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            }
            while helper_name in existing_names:
                helper_name += "_x"
            helper = ast.FunctionDef(
                name=helper_name,
                args=ast.arguments(
                    posonlyargs=[],
                    args=[ast.arg(arg=name) for name in loaded_names],
                    kwonlyargs=[], kw_defaults=[], defaults=[],
                ),
                body=[ast.Return(value=copy.deepcopy(expression))],
                decorator_list=[],
            )
            # ``from __future__`` imports must remain the first statements in
            # a module; place the extracted helper immediately after them.
            insertion_index = 0
            while (
                insertion_index < len(module.body)
                and isinstance(module.body[insertion_index], ast.ImportFrom)
                and module.body[insertion_index].module == "__future__"
            ):
                insertion_index += 1
            module.body.insert(insertion_index, helper)
            replacement = ast.Call(
                func=ast.Name(id=helper_name, ctx=ast.Load()),
                args=[ast.Name(id=name, ctx=ast.Load()) for name in loaded_names],
                keywords=[],
            )
            # Copy the call's fields into the existing expression object.  The
            # expression class itself may differ (e.g. BinOp -> Call), so use
            # the generic parent replacement below when needed.
            class _Replace(ast.NodeTransformer):
                def visit(self, current):  # type: ignore[override]
                    if current is expression:
                        return replacement
                    return super().visit(current)
            _Replace().visit(module)
        elif name == "control_flow_guard":
            statements, index, statement = target
            guard = ast.If(
                test=ast.Constant(value=(int(slot) % 2 == 0)),
                body=[copy.deepcopy(statement)],
                orelse=[],
            )
            statements[index] = guard
        elif name == "branch_expansion":
            assert isinstance(target, ast.If) and not target.orelse
            target.orelse = copy.deepcopy(target.body)
        elif name == "branch_swap":
            assert isinstance(target, ast.If) and target.orelse
            target.body, target.orelse = (
                copy.deepcopy(target.orelse),
                copy.deepcopy(target.body),
            )
        elif name in {"loop_update_repeat", "state_update_repeat"}:
            statements, index, statement = target
            statements.insert(index + 1, copy.deepcopy(statement))
    else:
        returns = [
            node
            for root in roots
            for node in ast.walk(root)
            if isinstance(node, ast.Return) and node.value
        ]
        if not returns:
            raise ValueError("program has no AST expression that can be mutated")
        name = "return_expression"
        target = returns[int(slot) % len(returns)]
        assert isinstance(target, ast.Return) and target.value is not None
        target.value = ast.UnaryOp(op=ast.Not(), operand=target.value)

    ast.fix_missing_locations(module)
    compile(module, "<mutation>", "exec")
    return name


class PythonProgramSearchSpace(SearchSpace):
    """Population search over complete Python source trees.

    ``propose`` supports four operators:

    - ``llm_generate``: write a program from a blank slate;
    - ``llm_rewrite``: rewrite one or two elite whole programs;
    - ``ast_mutation``: structurally mutate an elite program;
    - ``ast_crossover``: compose the two highest-scoring parents.

    Passing ``context={"operator": ...}`` selects an operator explicitly.
    Otherwise the available operators are rotated deterministically by slot.
    """

    def __init__(
        self,
        *,
        generator: WholeProgramGenerator | Callable[[ProgramGenerationRequest], Mapping[str, Any]] | None = None,
        seeds: Sequence[AlgorithmSpec | Mapping[str, str]] = (),
        entrypoint: str = "algorithm.py",
        required_symbol: str = "solve",
        direction: str = "max",
        elite_size: int = 4,
    ):
        if direction not in {"max", "min"}:
            raise ValueError("direction must be 'max' or 'min'")
        if not isinstance(elite_size, int) or elite_size < 1:
            raise ValueError("elite_size must be a positive int")
        if not isinstance(required_symbol, str) or not required_symbol.isidentifier():
            raise ValueError("required_symbol must be a Python identifier")
        if not isinstance(entrypoint, str):
            raise ValueError("entrypoint must be a relative Python path")
        entrypoint_path = PurePosixPath(entrypoint)
        if (
            entrypoint_path.is_absolute()
            or ".." in entrypoint_path.parts
            or entrypoint_path.suffix != ".py"
        ):
            raise ValueError("entrypoint must be a relative Python path")
        self.generator = generator
        self.entrypoint = entrypoint
        self.required_symbol = required_symbol
        self.direction = direction
        self.elite_size = elite_size
        self._candidates: dict[str, AlgorithmSpec] = {}
        self._observations: dict[str, EvaluationResult] = {}
        self._seed_candidate_ids: list[str] = []
        self._round_parent_ids: tuple[str, ...] | None = None
        self._counter = 0

        for seed in seeds:
            if isinstance(seed, AlgorithmSpec):
                candidate = seed
            else:
                candidate = self._build_candidate(
                    source=dict(seed),
                    operator="seed",
                    parent_ids=(),
                    slot=len(self._candidates),
                    context={},
                )
            if isinstance(seed, AlgorithmSpec):
                self.validate(candidate)
                if candidate.candidate_id in self._candidates:
                    raise ValueError(f"duplicate candidate_id: {candidate.candidate_id}")
                self._candidates[candidate.candidate_id] = candidate
            self._seed_candidate_ids.append(candidate.candidate_id)

    @property
    def candidates(self) -> tuple[AlgorithmSpec, ...]:
        return tuple(self._candidates.values())

    @property
    def observations(self) -> tuple[EvaluationResult, ...]:
        return tuple(self._observations.values())

    @staticmethod
    def source_digest(source: Mapping[str, str]) -> str:
        """Expose the canonical source fingerprint for proposal deduplication."""
        return _source_digest(source)

    def equivalent_candidates(self, candidate: AlgorithmSpec) -> tuple[AlgorithmSpec, ...]:
        """Return candidates with byte-identical source trees, in insertion order."""
        digest = _source_digest(_source_from(candidate))
        return tuple(
            existing
            for existing in self._candidates.values()
            if _source_digest(_source_from(existing)) == digest
        )

    def _next_id(self) -> str:
        while True:
            self._counter += 1
            candidate_id = f"program_{self._counter:06d}"
            if candidate_id not in self._candidates:
                return candidate_id

    def _build_candidate(
        self,
        *,
        source: Mapping[str, str],
        operator: str,
        parent_ids: Sequence[str],
        slot: int,
        context: Mapping[str, Any],
        generated: _GeneratedProgram | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> AlgorithmSpec:
        generated = generated or _GeneratedProgram(source=source)
        combined_metadata = {
            "slot": int(slot),
            "entrypoint": self.entrypoint,
            "required_symbol": self.required_symbol,
            **dict(generated.metadata),
            **dict(metadata or {}),
            "source_digest": _source_digest(source),
        }
        candidate = AlgorithmSpec(
            candidate_id=self._next_id(),
            family=PYTHON_PROGRAM_FAMILY,
            implementation={
                "entrypoint": self.entrypoint,
                "required_symbol": self.required_symbol,
                "source": dict(source),
            },
            parent_ids=tuple(parent_ids),
            mechanism_id=generated.mechanism_id or str(context.get("mechanism_id", "")),
            operator=operator,
            prediction=generated.prediction if generated.prediction != "not_observed" else context.get("prediction", "not_observed"),
            falsifier=generated.falsifier if generated.falsifier != "not_observed" else context.get("falsifier", "not_observed"),
            metadata=combined_metadata,
        )
        self.validate(candidate)
        self._candidates[candidate.candidate_id] = candidate
        return candidate

    def _ranked_parents(self) -> list[AlgorithmSpec]:
        if self._round_parent_ids is not None:
            return [
                self._candidates[candidate_id]
                for candidate_id in self._round_parent_ids
            ]

        def key(candidate: AlgorithmSpec) -> tuple[float, str]:
            result = self._observations.get(candidate.candidate_id)
            assert result is not None and result.score is not None
            score = float(result.score)
            primary = -score if self.direction == "max" else score
            return (primary, candidate.candidate_id)

        successful = [
            candidate
            for candidate in self._candidates.values()
            if (
                (result := self._observations.get(candidate.candidate_id))
                is not None
                and result.status == "ok"
                and result.score is not None
                and math.isfinite(float(result.score))
            )
        ]
        if successful:
            return sorted(successful, key=key)[: self.elite_size]
        return [
            self._candidates[candidate_id]
            for candidate_id in self._seed_candidate_ids
            if candidate_id not in self._observations
        ][: self.elite_size]

    def begin_round(self) -> None:
        """Freeze the scored parent population for one proposal cohort."""
        if self._round_parent_ids is not None:
            raise RuntimeError("program-search round is already active")
        self._round_parent_ids = tuple(
            candidate.candidate_id for candidate in self._ranked_parents()
        )

    def end_round(self) -> None:
        self._round_parent_ids = None

    def _select_operator(
        self,
        context: Mapping[str, Any],
        slot: int,
        parents: Sequence[AlgorithmSpec],
    ) -> str:
        requested = context.get("operator")
        if requested:
            return str(requested)
        if not parents:
            return "llm_generate"
        operators = ["ast_mutation"]
        if len(parents) >= 2:
            operators.append("ast_crossover")
        if self.generator is not None:
            operators.append("llm_rewrite")
        return operators[int(slot) % len(operators)]

    def _generate(
        self,
        *,
        context: Mapping[str, Any],
        slot: int,
        operator: str,
        parents: Sequence[AlgorithmSpec],
    ) -> AlgorithmSpec:
        if self.generator is None:
            raise ValueError(f"{operator} requires a whole-program generator callback")
        request = ProgramGenerationRequest(
            context=dict(context),
            slot=int(slot),
            operator=operator,
            parents=tuple(parents),
            observations=self.observations,
            entrypoint=self.entrypoint,
            required_symbol=self.required_symbol,
        )
        generated = _normalise_generated(self.generator(request))
        return self._build_candidate(
            source=generated.source,
            operator=operator,
            parent_ids=[parent.candidate_id for parent in parents],
            slot=slot,
            context=context,
            generated=generated,
        )

    def mutate(
        self,
        parent: AlgorithmSpec,
        *,
        slot: int = 0,
        context: Mapping[str, Any] | None = None,
    ) -> AlgorithmSpec:
        self.validate(parent)
        source = _source_from(parent)
        python_files = [
            path for path in sorted(source)
            if path.endswith(".py")
        ]
        if not python_files:
            raise ValueError("program has no Python source file to mutate")
        preferred = self.entrypoint if self.entrypoint in python_files else python_files[0]
        module = ast.parse(source[preferred], filename=preferred)
        protected_functions = () if self.required_symbol == "main" else ("main",)
        requested_mutation = (context or {}).get("mutation_kind")
        if requested_mutation is not None and not isinstance(
            requested_mutation, str
        ):
            raise ValueError("mutation_kind must be a string")
        mutation = _mutate_tree(
            module,
            slot,
            protected_functions=protected_functions,
            mutation_kind=requested_mutation,
        )
        source[preferred] = ast.unparse(module) + "\n"
        return self._build_candidate(
            source=source,
            operator="ast_mutation",
            parent_ids=(parent.candidate_id,),
            slot=slot,
            context=context or {},
            metadata={"mutation": mutation, "mutated_file": preferred},
        )

    def crossover(
        self,
        left: AlgorithmSpec,
        right: AlgorithmSpec,
        *,
        slot: int = 0,
        context: Mapping[str, Any] | None = None,
    ) -> AlgorithmSpec:
        self.validate(left)
        self.validate(right)
        if left.candidate_id == right.candidate_id:
            raise ValueError("crossover requires two distinct parents")
        left_source = _source_from(left)
        right_source = _source_from(right)
        if self.entrypoint not in left_source or self.entrypoint not in right_source:
            raise ValueError("both parents must contain the configured entrypoint")

        # Preserve the complete left tree and every non-conflicting right-side
        # file.  The entrypoint itself is rebuilt from both AST function graphs.
        child_source = dict(left_source)
        for path, text in right_source.items():
            if path == self.entrypoint:
                continue
            if path == "manifest.json" and path in child_source:
                try:
                    manifests_match = (
                        json.loads(child_source[path]) == json.loads(text)
                    )
                except (TypeError, ValueError):
                    manifests_match = child_source[path] == text
                if not manifests_match:
                    raise ValueError(
                        "crossover parents contain conflicting source file: "
                        "manifest.json"
                    )
                continue
            if path in child_source and child_source[path] != text:
                raise ValueError(
                    f"crossover parents contain conflicting source file: {path}"
                )
            child_source[path] = text
        fit_predict_contract = (
            self.required_symbol == "predict"
            and all(
                f"def {symbol}" in left_source[self.entrypoint]
                and f"def {symbol}" in right_source[self.entrypoint]
                for symbol in ("fit", "predict")
            )
        )
        left_interface = ""
        right_interface = ""
        try:
            left_interface = str(
                json.loads(left_source.get("manifest.json", "{}"))
                .get("interface", "")
            )
            right_interface = str(
                json.loads(right_source.get("manifest.json", "{}"))
                .get("interface", "")
            )
        except (TypeError, ValueError):
            pass
        if left_interface and right_interface and left_interface != right_interface:
            raise ValueError("crossover parents declare different program interfaces")
        decision_interface = left_interface == right_interface == "decision"
        child_source[self.entrypoint] = (
            _crossover_fit_predict_program(
                left_source[self.entrypoint],
                right_source[self.entrypoint],
                decision_interface=decision_interface,
            )
            if fit_predict_contract
            else _crossover_entrypoint(
                left_source[self.entrypoint],
                right_source[self.entrypoint],
                required_symbol=self.required_symbol,
            )
        )
        return self._build_candidate(
            source=child_source,
            operator="ast_crossover",
            parent_ids=(left.candidate_id, right.candidate_id),
            slot=slot,
            context=context or {},
            metadata={
                "crossover": (
                    "fit_predict_program_composition"
                    if fit_predict_contract
                    else "namespaced_function_composition"
                ),
                "prediction_interface": (
                    left_interface if left_interface == right_interface else ""
                ),
            },
        )

    def propose(self, context: Mapping[str, Any], slot: int) -> AlgorithmSpec:
        if not isinstance(context, Mapping):
            raise ValueError("context must be a mapping")
        parents = self._ranked_parents()
        operator = self._select_operator(context, slot, parents)
        if operator == "llm_generate":
            return self._generate(context=context, slot=slot, operator=operator, parents=())
        if operator == "llm_rewrite":
            requested = 2 if context.get("two_parent_rewrite") and len(parents) >= 2 else 1
            if not parents:
                return self._generate(context=context, slot=slot, operator="llm_generate", parents=())
            return self._generate(
                context=context,
                slot=slot,
                operator=operator,
                parents=parents[:requested],
            )
        if operator == "ast_mutation":
            if not parents:
                raise ValueError("ast_mutation requires at least one parent")
            return self.mutate(parents[0], slot=slot, context=context)
        if operator == "ast_crossover":
            if len(parents) < 2:
                raise ValueError("ast_crossover requires at least two parents")
            return self.crossover(parents[0], parents[1], slot=slot, context=context)
        raise ValueError(f"unknown program-search operator: {operator}")

    def observe(self, result: EvaluationResult) -> None:
        if not isinstance(result, EvaluationResult):
            result = EvaluationResult.from_dict(result)
        result.validate()
        if result.candidate_id not in self._candidates:
            raise ValueError(f"cannot observe unknown candidate: {result.candidate_id}")
        self._observations[result.candidate_id] = result

    def validate(self, candidate: AlgorithmSpec) -> None:
        candidate.validate()
        if candidate.family != PYTHON_PROGRAM_FAMILY:
            raise ValueError(f"family must be {PYTHON_PROGRAM_FAMILY}")
        source = _source_from(candidate)
        implementation = candidate.implementation
        assert isinstance(implementation, Mapping)
        if implementation.get("entrypoint") != self.entrypoint:
            raise ValueError(f"implementation.entrypoint must be {self.entrypoint}")
        if implementation.get("required_symbol") != self.required_symbol:
            raise ValueError(f"implementation.required_symbol must be {self.required_symbol}")
        if self.entrypoint not in source:
            raise ValueError(f"source is missing entrypoint {self.entrypoint}")

        entrypoint_tree: ast.Module | None = None
        for raw_path, text in source.items():
            path = PurePosixPath(raw_path)
            if path.is_absolute() or ".." in path.parts or str(path) in {"", "."}:
                raise ValueError(f"source path must be relative: {raw_path}")
            if path.suffix != ".py":
                continue
            tree = ast.parse(text, filename=raw_path)
            compile(tree, raw_path, "exec")
            if raw_path == self.entrypoint:
                entrypoint_tree = tree
        if entrypoint_tree is None:
            raise ValueError("entrypoint must be a Python file")
        symbols = {
            node.name
            for node in entrypoint_tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        if self.required_symbol not in symbols:
            raise ValueError(
                f"entrypoint must define top-level {self.required_symbol}()"
            )

    def materialize(self, candidate: AlgorithmSpec, directory: str | Path) -> tuple[Path, ...]:
        self.validate(candidate)
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []
        for relative, text in sorted(_source_from(candidate).items()):
            destination = root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(text, encoding="utf-8")
            paths.append(destination)
        return tuple(paths)


__all__ = [
    "PYTHON_PROGRAM_FAMILY",
    "ProgramGenerationRequest",
    "WholeProgramGenerator",
    "PythonProgramSearchSpace",
]
