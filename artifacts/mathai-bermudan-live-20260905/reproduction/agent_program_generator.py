"""LLM-backed whole-program generation for the standalone search engine.

The main Harness already uses Proposal Agents over Experience Bank records.
This adapter gives :class:`program_search.PythonProgramSearchSpace` the same
concrete generation capability when it is used directly through
``AlgorithmDiscoveryLoop.run_search``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from llm_backend import run_agent
from program_search import ProgramGenerationRequest, _source_from


DEFAULT_SOURCE_SUFFIXES = (".py", ".json", ".toml", ".yaml", ".yml")


def _blank_entrypoint(request: ProgramGenerationRequest) -> str:
    if request.required_symbol == "predict":
        return (
            '"""Whole-program draft. Replace this scaffold."""\n\n'
            "def fit(input_dir, output_dir, seed):\n"
            "    raise NotImplementedError(\"implement fit\")\n\n"
            "def predict(model_dir, input_dir, output_dir):\n"
            "    raise NotImplementedError(\"implement predict\")\n"
        )
    return (
        '"""Whole-program draft. Replace this scaffold."""\n\n'
        f"def {request.required_symbol}(*args, **kwargs):\n"
        f"    raise NotImplementedError(\"implement {request.required_symbol}\")\n"
    )


def _render_parent(parent, index: int) -> str:
    chunks = []
    for path, source in sorted(_source_from(parent).items()):
        chunks.append(f"## Parent {index} file: {path}\n```text\n{source}\n```")
    return "\n\n".join(chunks)


class AgentWholeProgramGenerator:
    """Generate or rewrite complete source trees with the configured LLM CLI.

    Each request receives a fresh editable directory beneath ``workspace_root``.
    Parent source is copied in as the initial program, while every parent is
    also rendered in the prompt for explicit one- or two-parent rewriting.
    The returned mapping contains the complete text source tree, not model
    parameters or a registered family choice.
    """

    def __init__(
        self,
        workspace_root: str | Path,
        *,
        backend: str | None = None,
        model: str | None = None,
        timeout_s: int = 600,
        initial_files: Mapping[str, str] | None = None,
        source_suffixes: Sequence[str] = DEFAULT_SOURCE_SUFFIXES,
    ) -> None:
        self.workspace_root = Path(workspace_root)
        self.backend = backend
        self.model = model
        self.timeout_s = int(timeout_s)
        self.initial_files = dict(initial_files or {})
        self.source_suffixes = tuple(source_suffixes)
        self._counter = 0

    def _next_workspace(self) -> Path:
        self._counter += 1
        workspace = self.workspace_root / f"candidate_{self._counter:06d}"
        workspace.mkdir(parents=True, exist_ok=False)
        return workspace

    @staticmethod
    def _write_source(workspace: Path, source: Mapping[str, str]) -> None:
        for relative, text in source.items():
            destination = workspace / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(text, encoding="utf-8")

    def _initial_source(self, request: ProgramGenerationRequest) -> dict[str, str]:
        if request.parents:
            return _source_from(request.parents[0])
        source = dict(self.initial_files)
        source.setdefault(request.entrypoint, _blank_entrypoint(request))
        return source

    @staticmethod
    def _prompt(request: ProgramGenerationRequest) -> str:
        context = json.dumps(
            dict(request.context), ensure_ascii=False, sort_keys=True, indent=2,
        )
        parents = "\n\n".join(
            _render_parent(parent, index)
            for index, parent in enumerate(request.parents, start=1)
        ) or "No parent program: generate from a genuinely new starting principle."
        return f"""Create one complete finite Python program for an algorithm-search candidate.

Operator: {request.operator}
Entrypoint: {request.entrypoint}
Required top-level function: {request.required_symbol}

Research context:
{context}

Parent programs:
{parents}

Edit the source files in the current directory. You may replace the complete
algorithm, add Python helper modules, change representations, objectives,
optimizers, control flow, state and data structures. Preserve only the external
entrypoint contract. Do not return a parameter choice for a registered model
family. Leave the finished implementation in the workspace.
"""

    def _read_source(self, workspace: Path) -> dict[str, str]:
        source = {}
        for path in sorted(workspace.rglob("*")):
            if (
                path.is_file()
                and not any(part.startswith(".") for part in path.relative_to(workspace).parts)
                and path.suffix in self.source_suffixes
            ):
                source[path.relative_to(workspace).as_posix()] = path.read_text(
                    encoding="utf-8"
                )
        return source

    def __call__(self, request: ProgramGenerationRequest) -> Mapping[str, Any]:
        workspace = self._next_workspace()
        self._write_source(workspace, self._initial_source(request))
        result = run_agent(
            self._prompt(request),
            cwd=workspace,
            writable=True,
            timeout_s=self.timeout_s,
            backend=self.backend,
            model=self.model,
        )
        if result.returncode != 0:
            detail = result.stderr.strip().splitlines()
            tail = detail[-1] if detail else "no backend error text"
            raise RuntimeError(
                f"whole-program agent exited with {result.returncode}: {tail[:500]}"
            )
        source = self._read_source(workspace)
        if request.entrypoint not in source:
            raise ValueError(
                f"whole-program agent did not produce {request.entrypoint}"
            )
        return {
            "source": source,
            "metadata": {
                "generator": "llm_agent",
                "operator": request.operator,
            },
        }


__all__ = ["AgentWholeProgramGenerator", "DEFAULT_SOURCE_SUFFIXES"]
