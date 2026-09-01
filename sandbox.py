"""Isolated candidate execution followed by snapshot-based trusted scoring."""

import hashlib
import json
import math
import os
import shutil
import signal
import stat
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MAX_ARTIFACT_BYTES = 1024 * 1024
READ_CHUNK_BYTES = 64 * 1024
SOURCE_TREE_IGNORES = {
    ".venv", "__pycache__", ".git", ".tmp",
    "run.log", "train.log", "solution.json", "solution.snapshot.json",
    "evidence.json",
}
HEX_DIGITS = frozenset("0123456789abcdef")
EVALUATION_REQUEST_SCHEMA = "openhyra-evaluation-request.v1"
NUMERIC_THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
}

SANDBOX_PROFILE = """(version 1)
(allow default)
(deny network*)
(deny file-write*)
(allow file-write* (subpath "{sandbox}"))
(allow file-write* (literal "/dev/null"))
(deny file-read* (literal "{evaluator}"))
"""


# This profile is deliberately separate from ``SANDBOX_PROFILE``.  The latter
# is a compatibility boundary for existing candidates; training candidates are
# arbitrary code and need a default-deny confidentiality boundary instead.
TRAINING_SANDBOX_LOG_BYTES = 64 * 1024
TRAINING_SANDBOX_DEFAULT_MEMORY_BYTES = 1024 * 1024 * 1024
TRAINING_SANDBOX_DEFAULT_FILE_SIZE_BYTES = 64 * 1024 * 1024
TRAINING_SANDBOX_DEFAULT_TOTAL_OUTPUT_BYTES = 64 * 1024 * 1024
TRAINING_SANDBOX_DEFAULT_OUTPUT_ENTRIES = 256
TRAINING_SANDBOX_DEVICE_READS = (
    Path("/dev/null"),
    Path("/dev/random"),
    Path("/dev/urandom"),
)
TRAINING_SANDBOX_DYLD_METADATA_ROOTS = (
    Path("/System/Cryptexes"),
    Path("/System/Volumes/Preboot/Cryptexes"),
)
TRAINING_SANDBOX_FORBIDDEN_BROAD_ROOTS = tuple(Path(value) for value in (
    "/Applications", "/Library", "/Network", "/System", "/Users",
    "/Volumes", "/cores", "/dev", "/etc", "/home", "/opt",
    "/private", "/private/tmp", "/private/var", "/sbin", "/tmp",
    "/usr", "/usr/local", "/var",
))


@dataclass(frozen=True)
class TrainingSandboxPaths:
    """Canonical, non-overlapping roots exposed to one training process."""

    source_dir: Path
    input_dir: Path
    output_dir: Path
    tmp_dir: Path
    runtime_roots: tuple


def _resolved_directory(path, label):
    """Resolve one allowlist root and reject a link used as that root."""
    try:
        raw = Path(path)
    except TypeError as exc:
        raise ValueError(f"{label} must be a filesystem path") from exc
    try:
        raw_info = os.lstat(raw)
    except FileNotFoundError as exc:
        raise ValueError(f"{label} does not exist") from exc
    except OSError as exc:
        raise ValueError(f"could not inspect {label}: {exc}") from exc
    if stat.S_ISLNK(raw_info.st_mode):
        raise ValueError(f"{label} must not be a symbolic link")
    try:
        resolved = raw.resolve(strict=True)
        resolved_info = os.lstat(resolved)
    except (FileNotFoundError, OSError) as exc:
        raise ValueError(f"could not resolve {label}: {exc}") from exc
    if stat.S_ISLNK(resolved_info.st_mode):
        raise ValueError(f"{label} must not be a symbolic link")
    if not stat.S_ISDIR(resolved_info.st_mode):
        raise ValueError(f"{label} must be a directory")
    return resolved


def _forbidden_training_root(path):
    """Return the protected root contained by an overly broad allowlist."""
    if path in TRAINING_SANDBOX_FORBIDDEN_BROAD_ROOTS:
        return path
    protected = {Path("/")}
    for candidate in (Path.home(), Path(__file__).resolve().parent):
        try:
            protected.add(candidate.resolve(strict=True))
        except OSError:
            protected.add(candidate.resolve())
    for target in protected:
        if path == target or path in target.parents:
            return target
    return None


def _paths_overlap(first, second):
    return (
        first == second
        or first in second.parents
        or second in first.parents
    )


def _reject_unsafe_training_tree(root, label):
    """Reject links and special files in sealed source or instance input."""
    for current, directories, filenames in os.walk(
            root, topdown=True, followlinks=False):
        current = Path(current)
        for name in (*directories, *filenames):
            path = current / name
            info = os.lstat(path)
            relative = path.relative_to(root).as_posix()
            if stat.S_ISLNK(info.st_mode):
                raise ValueError(
                    f"{label} entry {relative} must not be a symbolic link"
                )
            if name in directories and not stat.S_ISDIR(info.st_mode):
                raise ValueError(
                    f"{label} entry {relative} must be a real directory"
                )
            if name in filenames and not stat.S_ISREG(info.st_mode):
                raise ValueError(
                    f"{label} entry {relative} must be a regular file"
                )
            if name in filenames and info.st_nlink != 1:
                raise ValueError(
                    f"{label} entry {relative} must have exactly one hard link"
                )


def validate_training_sandbox_paths(
        source_dir, input_dir, output_dir, tmp_dir, runtime_roots):
    """Validate and canonicalize every training sandbox allowlist root.

    Broad roots and overlapping roots are rejected because either turns a
    seemingly narrow Seatbelt exception into access to unrelated host data.
    The writable roots must already exist so the validated inode cannot be
    replaced by an implicitly-created symlink during setup.
    """
    try:
        runtime_roots = tuple(runtime_roots)
    except TypeError as exc:
        raise ValueError("runtime_roots must be a non-empty iterable") from exc
    if not runtime_roots:
        raise ValueError("runtime_roots must contain an explicit runtime root")

    named = [
        ("source_dir", _resolved_directory(source_dir, "source_dir")),
        ("input_dir", _resolved_directory(input_dir, "input_dir")),
        ("output_dir", _resolved_directory(output_dir, "output_dir")),
        ("tmp_dir", _resolved_directory(tmp_dir, "tmp_dir")),
    ]
    for index, root in enumerate(runtime_roots):
        named.append((
            f"runtime_roots[{index}]",
            _resolved_directory(root, f"runtime_roots[{index}]"),
        ))

    for label, root in named:
        forbidden = _forbidden_training_root(root)
        if forbidden is not None:
            raise ValueError(
                f"{label} is an overly broad allowlist root containing "
                f"protected path {forbidden}"
            )
    for index, (first_label, first) in enumerate(named):
        for second_label, second in named[index + 1:]:
            if _paths_overlap(first, second):
                raise ValueError(
                    "training sandbox roots must not overlap: "
                    f"{first_label} and {second_label}"
                )

    _reject_unsafe_training_tree(named[0][1], "source_dir")
    _reject_unsafe_training_tree(named[1][1], "input_dir")

    return TrainingSandboxPaths(
        source_dir=named[0][1],
        input_dir=named[1][1],
        output_dir=named[2][1],
        tmp_dir=named[3][1],
        runtime_roots=tuple(root for _label, root in named[4:]),
    )


def _seatbelt_path_clause(operation, roots):
    conditions = []
    for root in roots:
        escaped = _seatbelt_escape(root)
        conditions.extend((
            f'    (literal "{escaped}")',
            f'    (subpath "{escaped}")',
        ))
    return "\n".join((f"(allow {operation}", *conditions, ")"))


def _training_sandbox_profile(paths):
    readable = (
        paths.source_dir,
        paths.input_dir,
        paths.output_dir,
        paths.tmp_dir,
        *paths.runtime_roots,
    )
    executable = (paths.source_dir, *paths.runtime_roots)
    writable = (paths.output_dir, paths.tmp_dir)

    # Parent-directory metadata is needed to traverse to an allowed root.  It
    # is granted only for literal ancestors, never as a broad subtree read.
    ancestors = {Path("/")}
    for root in (*readable, *writable, *TRAINING_SANDBOX_DEVICE_READS):
        ancestors.update(root.parents)
    metadata_literals = "\n".join(
        f'    (literal "{_seatbelt_escape(path)}")'
        for path in sorted(ancestors, key=str)
    )
    return "\n".join((
        "(version 1)",
        "(deny default)",
        "(deny network*)",
        "(allow process-fork)",
        "(allow signal (target self))",
        # Bootstrap/syscall mediation is required for dyld to start a process;
        # file and network operations remain governed by the rules below.
        "(allow mach-bootstrap)",
        "(allow syscall*)",
        # Keep descendants in the session/process group owned by the parent.
        "(deny syscall-unix (syscall-number SYS_setsid SYS_setpgid))",
        "(allow sysctl-read)",
        "(allow file-read-metadata",
        metadata_literals,
        *(f'    (subpath "{_seatbelt_escape(path)}")'
          for path in TRAINING_SANDBOX_DYLD_METADATA_ROOTS),
        ")",
        # dyld opens the root directory as an openat anchor during process
        # startup.  This leaks only the top-level directory names; it grants no
        # descendant file data.  Removing it prevents the trusted wrapper and
        # candidate runtime from starting on macOS.
        '(allow file-read-data (literal "/"))',
        _seatbelt_path_clause("file-read*", readable),
        "(allow file-read*",
        *(f'    (literal "{_seatbelt_escape(path)}")'
          for path in TRAINING_SANDBOX_DEVICE_READS),
        ")",
        _seatbelt_path_clause("process-exec", executable),
        _seatbelt_path_clause("file-write*", writable),
        '(allow file-write* (literal "/dev/null"))',
        "",
    ))


def build_training_sandbox_profile(
        source_dir, input_dir, output_dir, tmp_dir, runtime_roots):
    """Build the macOS default-deny profile for one training invocation."""
    paths = validate_training_sandbox_paths(
        source_dir, input_dir, output_dir, tmp_dir, runtime_roots,
    )
    return _training_sandbox_profile(paths)


def training_sandbox_environment(tmp_dir, runtime_roots):
    """Return a fixed environment without host credentials or user config."""
    tmp_dir = _resolved_directory(tmp_dir, "tmp_dir")
    runtime_roots = tuple(
        _resolved_directory(root, f"runtime_roots[{index}]")
        for index, root in enumerate(runtime_roots)
    )
    path_entries = []
    for root in runtime_roots:
        candidates = [root] if root.name in {"bin", "sbin"} else [
            root / "bin", root / "sbin",
        ]
        for candidate in candidates:
            if candidate.is_dir() and candidate not in path_entries:
                path_entries.append(candidate)
    return {
        "PATH": os.pathsep.join(str(path) for path in path_entries),
        "HOME": str(tmp_dir),
        "TMPDIR": str(tmp_dir),
        "TMP": str(tmp_dir),
        "TEMP": str(tmp_dir),
        "PYTHONHASHSEED": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "LANG": "C",
        "LC_ALL": "C",
        "TZ": "UTC",
        **NUMERIC_THREAD_ENV,
    }


TRAINING_LIMIT_WRAPPER = r"""
import os, resource, sys
limits = [
    (resource.RLIMIT_FSIZE, int(sys.argv[2]), "file-size"),
    (resource.RLIMIT_CPU, int(sys.argv[3]), "cpu-time"),
]
# macOS exposes RLIMIT_AS but rejects finite values after the Python runtime
# has mapped its shared-cache address space.  The trusted parent therefore
# enforces aggregate process-group RSS there (and on every platform as a
# defense in depth); kernels with a working RLIMIT_AS get both controls.
if sys.platform != "darwin":
    limits.insert(0, (resource.RLIMIT_AS, int(sys.argv[1]), "address-space"))
for key, value, label in limits:
    try:
        _soft, hard = resource.getrlimit(key)
        target = value if hard == resource.RLIM_INFINITY else min(value, hard)
        resource.setrlimit(key, (target, target))
        applied, _hard = resource.getrlimit(key)
        if applied == resource.RLIM_INFINITY or applied > value:
            raise RuntimeError("limit was not applied")
    except Exception as exc:
        print("training sandbox could not apply %s limit: %s" % (label, exc),
              file=sys.stderr)
        raise SystemExit(126)
os.chdir(sys.argv[4])
os.execv(sys.argv[5], sys.argv[5:])
"""


def _positive_limit(value, label, *, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a positive number")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be a positive finite number")
    return int(math.ceil(value)) if integer else value


def _training_limited_cmd(
        command, *, source_dir, cpu_seconds, memory_bytes, file_size_bytes):
    command = _normalize_training_command(command)
    cpu_seconds = _positive_limit(cpu_seconds, "cpu_seconds", integer=True)
    memory_bytes = _positive_limit(memory_bytes, "memory_bytes", integer=True)
    file_size_bytes = _positive_limit(
        file_size_bytes, "file_size_bytes", integer=True,
    )
    return [
        sys.executable,
        "-I",
        "-S",
        "-c",
        TRAINING_LIMIT_WRAPPER,
        str(memory_bytes),
        str(file_size_bytes),
        str(cpu_seconds),
        str(source_dir),
        *command,
    ]


def _normalize_training_command(command):
    if not isinstance(command, (list, tuple)) or not command:
        raise ValueError("training command must be a non-empty list")
    if any(
        not isinstance(item, (str, os.PathLike)) or "\0" in os.fspath(item)
        for item in command
    ):
        raise ValueError("training command entries must be NUL-free paths/text")
    return [os.fspath(item) for item in command]


def _training_sandboxed_cmd(paths, command, *, externally_isolated):
    if sys.platform == "darwin":
        profile = _training_sandbox_profile(paths)
        return ["/usr/bin/sandbox-exec", "-p", profile, *command], "seatbelt"
    if externally_isolated is not True:
        raise RuntimeError(
            "training sandbox fails closed without macOS Seatbelt; pass "
            "externally_isolated=True only after placing the process in an "
            "external container/VM with equivalent read, write, and network "
            "isolation"
        )
    return list(command), "external"


def _read_log_tail(stream, max_bytes=TRAINING_SANDBOX_LOG_BYTES):
    stream.flush()
    size = os.fstat(stream.fileno()).st_size
    stream.seek(max(0, size - max_bytes))
    data = stream.read(max_bytes)
    return data.decode("utf-8", errors="replace")


def _process_group_rss_bytes(process_group):
    """Return aggregate resident bytes for a process group via trusted ps."""
    try:
        result = subprocess.run(
            ["/bin/ps", "-axo", "pgid=,rss="],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=0.2,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"could not inspect training memory: {exc}") from exc
    if result.returncode != 0:
        raise RuntimeError(
            "could not inspect training memory: "
            + result.stderr.strip()[:300]
        )
    total_kib = 0
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) != 2:
            continue
        try:
            pgid, rss_kib = (int(field) for field in fields)
        except ValueError:
            continue
        if pgid == process_group:
            total_kib += max(0, rss_kib)
    return total_kib * 1024


def _training_tree_usage(roots, *, entry_limit, byte_limit):
    """Measure writable trees without following candidate-created links.

    The limits are checked while walking so a candidate cannot force the
    trusted monitor to retain or traverse an unbounded directory listing.
    Directories count as entries because millions of empty directories are as
    effective a denial of service as millions of empty files.
    """
    entries = 0
    total_bytes = 0
    pending = list(roots)
    while pending:
        directory = pending.pop()
        try:
            iterator = os.scandir(directory)
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise RuntimeError(
                f"could not inspect training writable tree: {exc}"
            ) from exc
        with iterator:
            for entry in iterator:
                try:
                    info = entry.stat(follow_symlinks=False)
                except FileNotFoundError:
                    continue
                except OSError as exc:
                    raise RuntimeError(
                        f"could not inspect training output entry: {exc}"
                    ) from exc
                entries += 1
                total_bytes += max(0, info.st_size)
                if entries > entry_limit or total_bytes > byte_limit:
                    return entries, total_bytes
                if stat.S_ISDIR(info.st_mode):
                    pending.append(Path(entry.path))
    return entries, total_bytes


def _wait_training_process(
        proc, timeout_s, memory_bytes, writable_roots,
        max_output_entries, max_total_output_bytes, cancel_event=None):
    """Poll wall time, cancellation, RSS, and aggregate writable output."""
    started = time.monotonic()
    usage = {
        "peak_memory_bytes": 0,
        "peak_writable_entries": 0,
        "peak_writable_bytes": 0,
    }
    while True:
        if cancel_event is not None and cancel_event.is_set():
            return "cancelled", "", usage
        remaining = timeout_s - (time.monotonic() - started)
        if remaining <= 0:
            return "timeout", "", usage
        try:
            rss_bytes = _process_group_rss_bytes(proc.pid)
        except RuntimeError as exc:
            return "monitor_error", str(exc), usage
        usage["peak_memory_bytes"] = max(
            usage["peak_memory_bytes"], rss_bytes,
        )
        if rss_bytes > memory_bytes:
            return "memory_limit", (
                "training process group exceeded memory limit "
                f"({rss_bytes} > {memory_bytes} bytes)"
            ), usage
        try:
            entries, output_bytes = _training_tree_usage(
                writable_roots,
                entry_limit=max_output_entries,
                byte_limit=max_total_output_bytes,
            )
        except RuntimeError as exc:
            return "monitor_error", str(exc), usage
        usage["peak_writable_entries"] = max(
            usage["peak_writable_entries"], entries,
        )
        usage["peak_writable_bytes"] = max(
            usage["peak_writable_bytes"], output_bytes,
        )
        if entries > max_output_entries or output_bytes > max_total_output_bytes:
            return "output_limit", (
                "training writable trees exceeded aggregate limit "
                f"({entries} entries, {output_bytes} bytes)"
            ), usage
        if proc.poll() is not None:
            return "completed", "", usage
        try:
            proc.wait(timeout=min(0.1, remaining))
            # Re-enter the loop once after exit so final writable-tree usage is
            # checked before a successful status can be returned.
            continue
        except subprocess.TimeoutExpired:
            pass


def run_training_sandbox(
        command, *, source_dir, input_dir, output_dir, tmp_dir,
        runtime_roots, timeout_s=60, cpu_seconds=None,
        memory_bytes=TRAINING_SANDBOX_DEFAULT_MEMORY_BYTES,
        file_size_bytes=TRAINING_SANDBOX_DEFAULT_FILE_SIZE_BYTES,
        max_total_output_bytes=TRAINING_SANDBOX_DEFAULT_TOTAL_OUTPUT_BYTES,
        max_output_entries=TRAINING_SANDBOX_DEFAULT_OUTPUT_ENTRIES,
        externally_isolated=False, cancel_event=None):
    """Run arbitrary candidate training code inside a hermetic boundary.

    This API does not evaluate or import candidate output.  It only runs one
    per-instance training command and returns process-level status; a trusted
    caller remains responsible for validating and freezing exported weights.
    """
    paths = validate_training_sandbox_paths(
        source_dir, input_dir, output_dir, tmp_dir, runtime_roots,
    )
    for label, root in (
        ("output_dir", paths.output_dir), ("tmp_dir", paths.tmp_dir),
    ):
        if any(root.iterdir()):
            raise ValueError(
                f"{label} must be empty for each training invocation"
            )
    timeout_s = _positive_limit(timeout_s, "timeout_s")
    command = _normalize_training_command(command)
    if cpu_seconds is None:
        cpu_seconds = max(1, int(math.ceil(timeout_s)))
    max_total_output_bytes = _positive_limit(
        max_total_output_bytes, "max_total_output_bytes", integer=True,
    )
    max_output_entries = _positive_limit(
        max_output_entries, "max_output_entries", integer=True,
    )
    # The first process must be the isolation boundary itself.  Starting the
    # Python limit wrapper from candidate cwd before Seatbelt would let a
    # candidate shadow imports such as ``resource.py`` and execute outside the
    # sandbox.  The isolated wrapper imports trusted stdlib with -I/-S, applies
    # inherited limits, then chdirs to the candidate source only for exec.
    limited = _training_limited_cmd(
        command,
        source_dir=paths.source_dir,
        cpu_seconds=cpu_seconds,
        memory_bytes=memory_bytes,
        file_size_bytes=file_size_bytes,
    )
    sandboxed, isolation = _training_sandboxed_cmd(
        paths, limited, externally_isolated=externally_isolated,
    )
    env = training_sandbox_environment(paths.tmp_dir, paths.runtime_roots)
    log_path = paths.tmp_dir / ".openhyra-training.log"
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        log_fd = os.open(log_path, flags, 0o600)
    except OSError as exc:
        raise ValueError(f"could not create private training log: {exc}") from exc

    started = time.monotonic()
    state = "completed"
    state_note = ""
    usage = {
        "peak_memory_bytes": 0,
        "peak_writable_entries": 0,
        "peak_writable_bytes": 0,
    }
    proc = None
    log_tail = ""
    try:
        with os.fdopen(log_fd, "w+b", buffering=0) as log_stream:
            try:
                proc = subprocess.Popen(
                    sandboxed,
                    cwd=paths.tmp_dir,
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=log_stream,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            except OSError as exc:
                return {
                    "status": "crash",
                    "returncode": None,
                    "log_tail": f"could not start training sandbox: {exc}",
                    "wall_seconds": time.monotonic() - started,
                    "isolation": isolation,
                    **usage,
                    "output_entries": 0,
                    "output_bytes": 0,
                }
            try:
                state, state_note, usage = _wait_training_process(
                    proc,
                    timeout_s,
                    int(memory_bytes),
                    (paths.output_dir, paths.tmp_dir),
                    int(max_output_entries),
                    int(max_total_output_bytes),
                    cancel_event,
                )
            finally:
                # Kill descendants even after a successful parent exit.  This
                # closes the output-mutation race before trusted collection.
                _kill_process_group(proc)
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    _kill_process_group(proc)
            log_tail = _read_log_tail(log_stream)
            output_entries, output_bytes = _training_tree_usage(
                (paths.output_dir,),
                entry_limit=int(max_output_entries),
                byte_limit=int(max_total_output_bytes),
            )
    finally:
        try:
            log_path.unlink()
        except FileNotFoundError:
            pass

    if state_note:
        log_tail = (log_tail + "\n" + state_note).strip()
    if state == "timeout":
        status = "timeout"
    elif state == "cancelled":
        status = "cancelled"
    elif state in {"memory_limit", "output_limit", "monitor_error"}:
        status = "resource_limit"
    else:
        status = "ok" if proc.returncode == 0 else "crash"
    return {
        "status": status,
        "returncode": proc.returncode,
        "log_tail": log_tail,
        "wall_seconds": time.monotonic() - started,
        "isolation": isolation,
        **usage,
        "output_entries": output_entries,
        "output_bytes": output_bytes,
    }


def _seatbelt_escape(path):
    return str(Path(path).resolve()).replace("\\", "\\\\").replace('"', '\\"')


def _is_sha256(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value).issubset(HEX_DIGITS)
    )


def _canonical_json_bytes(payload):
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode()


def validate_evaluation_request(request):
    """Validate the exact trusted evaluator request envelope."""
    if not isinstance(request, dict):
        raise ValueError("evaluation request must be an object")
    expected = {
        "schema", "stage", "task", "protocol", "seed", "suite_id", "config",
    }
    if set(request) != expected:
        raise ValueError(
            "evaluation request fields must be exactly: "
            + ", ".join(sorted(expected))
        )
    if request.get("schema") != EVALUATION_REQUEST_SCHEMA:
        raise ValueError("unsupported evaluation request schema")
    if request.get("stage") not in {"search", "audit"}:
        raise ValueError("evaluation request stage must be search or audit")
    for field in ("task", "protocol", "suite_id"):
        value = request.get(field)
        if not isinstance(value, str) or not value.strip() or len(value) > 256:
            raise ValueError(
                f"evaluation request {field} must be bounded non-empty text"
            )
    seed = request.get("seed")
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or not 0 <= seed <= (1 << 63) - 1
    ):
        raise ValueError("evaluation request seed must be a 63-bit non-negative integer")
    if not isinstance(request.get("config"), dict):
        raise ValueError("evaluation request config must be an object")
    try:
        encoded = _canonical_json_bytes(request)
    except (TypeError, ValueError) as exc:
        raise ValueError("evaluation request must be canonical-JSON serializable") from exc
    if len(encoded) > 1024 * 1024:
        raise ValueError("evaluation request exceeds the 1 MiB limit")
    return encoded


def write_evaluation_request(path, request):
    """Write one immutable, parent-controlled request and return its digest."""
    encoded = validate_evaluation_request(request)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        path.unlink()
    path.write_bytes(encoded)
    path.chmod(0o400)
    return hashlib.sha256(encoded).hexdigest()


def _sandboxed_cmd(sandbox_dir, evaluator, cmd):
    if sys.platform == "darwin":
        profile = SANDBOX_PROFILE.format(
            sandbox=_seatbelt_escape(sandbox_dir),
            evaluator=_seatbelt_escape(evaluator),
        )
        return ["sandbox-exec", "-p", profile] + cmd
    if os.environ.get("OPENHYRA_ALLOW_UNSANDBOXED") == "1":
        return cmd
    raise RuntimeError(
        "OpenHyra fails closed without macOS Seatbelt; set "
        "OPENHYRA_ALLOW_UNSANDBOXED=1 only inside an external container/VM"
    )


LIMIT_WRAPPER = r"""
import os, resource, sys
limits = (
    (resource.RLIMIT_AS, int(sys.argv[1])),
    (resource.RLIMIT_FSIZE, int(sys.argv[2])),
    (resource.RLIMIT_CPU, int(sys.argv[3])),
)
for key, value in limits:
    try:
        _soft, hard = resource.getrlimit(key)
        target = value if hard == resource.RLIM_INFINITY else min(value, hard)
        resource.setrlimit(key, (target, target))
    except (OSError, ValueError):
        pass
os.execvp(sys.argv[4], sys.argv[4:])
"""


def _limited_cmd(task, command):
    memory = int(getattr(task, "max_memory_mb", 1024)) * 1024 * 1024
    output = int(getattr(task, "max_output_mb", 64)) * 1024 * 1024
    return [
        sys.executable, "-c", LIMIT_WRAPPER,
        str(memory), str(output), str(int(task.timeout_s) + 5),
        *command,
    ]


def trusted_artifact_dir(sandbox_dir):
    """Return a parent-controlled directory outside the candidate write root."""
    sandbox_dir = Path(sandbox_dir)
    return sandbox_dir.parent / ".trusted_artifacts" / sandbox_dir.name


def read_regular_file(path, max_bytes, *, label=None):
    """Read one untrusted regular file once without following links."""
    path = Path(path)
    label = label or path.name
    try:
        before = os.lstat(path)
    except FileNotFoundError as exc:
        raise ValueError(f"{label} not found") from exc
    if stat.S_ISLNK(before.st_mode):
        raise ValueError(f"{label} must not be a symbolic link")

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f"could not safely open {label}: {exc}") from exc
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(f"{label} must be a regular file")
        if info.st_nlink != 1:
            raise ValueError(f"{label} must have exactly one hard link")
        if info.st_size > max_bytes:
            raise ValueError(
                f"{label} exceeds the {max_bytes}-byte limit"
            )
        chunks = []
        remaining = max_bytes + 1
        while remaining:
            chunk = os.read(fd, min(READ_CHUNK_BYTES, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        data = b"".join(chunks)
        if len(data) > max_bytes:
            raise ValueError(
                f"{label} exceeds the {max_bytes}-byte limit"
            )
        return data
    finally:
        os.close(fd)


def _source_tree_entries(source_dir, max_bytes):
    """Yield one bounded, symlink-free snapshot of a candidate source tree."""
    source_dir = Path(source_dir)
    try:
        root_info = os.lstat(source_dir)
    except FileNotFoundError as exc:
        raise ValueError("candidate source directory not found") from exc
    if stat.S_ISLNK(root_info.st_mode) or not stat.S_ISDIR(root_info.st_mode):
        raise ValueError("candidate source must be a real directory")

    total = 0
    for current, directories, filenames in os.walk(
            source_dir, topdown=True, followlinks=False):
        current = Path(current)
        kept_directories = []
        for name in sorted(directories):
            if name in SOURCE_TREE_IGNORES:
                continue
            child = current / name
            info = os.lstat(child)
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                relative = child.relative_to(source_dir).as_posix()
                raise ValueError(
                    f"candidate source directory {relative} must not be a link"
                )
            kept_directories.append(name)
        directories[:] = kept_directories

        for name in sorted(filenames):
            if name in SOURCE_TREE_IGNORES:
                continue
            path = current / name
            relative = path.relative_to(source_dir)
            remaining = max_bytes - total
            data = read_regular_file(
                path, max(0, remaining),
                label=f"candidate source file {relative.as_posix()}",
            )
            total += len(data)
            mode = os.lstat(path).st_mode & 0o777
            yield relative, data, mode


def _source_manifest_hash(hashes):
    payload = json.dumps(
        hashes, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def source_tree_hash(source_dir, max_bytes):
    """Hash exactly the source files that are eligible for execution/commit."""
    tree_hash, hashes, _files = read_source_tree(source_dir, max_bytes)
    return tree_hash, hashes


def read_source_tree(source_dir, max_bytes):
    """Read one complete source snapshot for hash validation or export."""
    hashes = {}
    files = {}
    for relative, data, _mode in _source_tree_entries(source_dir, max_bytes):
        name = relative.as_posix()
        hashes[name] = hashlib.sha256(data).hexdigest()
        files[name] = data
    return _source_manifest_hash(hashes), hashes, files


def snapshot_source_tree(source_dir, trusted_source_dir, max_bytes):
    """Seal candidate source bytes in a parent-controlled directory."""
    trusted_source_dir = Path(trusted_source_dir)
    if trusted_source_dir.exists():
        shutil.rmtree(trusted_source_dir)
    trusted_source_dir.mkdir(parents=True)
    hashes = {}
    try:
        for relative, data, mode in _source_tree_entries(source_dir, max_bytes):
            destination = trusted_source_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
            destination.chmod(mode & ~0o222)
            hashes[relative.as_posix()] = hashlib.sha256(data).hexdigest()
    except Exception:
        shutil.rmtree(trusted_source_dir, ignore_errors=True)
        raise
    return _source_manifest_hash(hashes), hashes


def _snapshot_artifact(artifact, trusted_dir, max_bytes):
    """Copy a validated candidate artifact into a fresh trusted directory."""
    data = read_regular_file(
        artifact, max_bytes, label="solution.json",
    )
    trusted_dir = Path(trusted_dir)
    trusted_dir.mkdir(parents=True, exist_ok=True)
    snapshot = trusted_dir / "solution.snapshot.json"
    if snapshot.exists() or snapshot.is_symlink():
        snapshot.unlink()
    snapshot.write_bytes(data)
    snapshot.chmod(0o444)
    return snapshot, data


def _kill_process_group(proc):
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _wait_process(proc, timeout_s, cancel_event=None):
    """Return completed, timeout, or cancelled while polling shared state."""
    started = time.monotonic()
    while True:
        if cancel_event is not None and cancel_event.is_set():
            return "cancelled"
        remaining = timeout_s - (time.monotonic() - started)
        if remaining <= 0:
            return "timeout"
        try:
            proc.wait(timeout=min(0.2, remaining))
            return "completed"
        except subprocess.TimeoutExpired:
            pass


def _trusted_score(
        task, snapshot_path, cancel_event=None, *, evaluation_request=None,
        trusted_dir=None):
    started = time.perf_counter()
    timeout_s = int(getattr(task, "evaluator_timeout_s", 300))
    memory_mb = int(getattr(task, "evaluator_max_memory_mb", 512))
    output_mb = int(getattr(task, "max_output_mb", 64))
    command = [sys.executable, str(task.evaluator), str(snapshot_path)]
    request_sha256 = None
    request_path = None
    if evaluation_request is not None:
        if trusted_dir is None:
            raise ValueError(
                "trusted_dir is required when passing an evaluation request"
            )
        request_path = Path(trusted_dir) / "evaluation_request.json"
        request_sha256 = write_evaluation_request(
            request_path, evaluation_request,
        )
        command.append(str(request_path))
    limited = [
        sys.executable, "-c", LIMIT_WRAPPER,
        str(memory_mb * 1024 * 1024),
        str(output_mb * 1024 * 1024),
        str(timeout_s + 5),
        *command,
    ]
    evaluator_env = os.environ.copy()
    evaluator_env.update(NUMERIC_THREAD_ENV)
    evaluator_env["PYTHONDONTWRITEBYTECODE"] = "1"
    proc = subprocess.Popen(
        limited, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, start_new_session=True, env=evaluator_env,
    )
    try:
        try:
            state = _wait_process(proc, timeout_s, cancel_event)
        finally:
            # Trusted code should not leave descendants behind either, even
            # when final audit is interrupted while waiting.
            _kill_process_group(proc)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        stdout, stderr = proc.communicate()
    finally:
        if request_path is not None:
            try:
                request_path.chmod(0o600)
                request_path.unlink()
            except OSError:
                pass
    if state == "timeout":
        return (
            None, "crash", {}, "evaluator timed out",
            time.perf_counter() - started, None, None, request_sha256,
        )
    if state == "cancelled":
        return (
            None, "cancelled", {}, "evaluator cancelled",
            time.perf_counter() - started, None, None, request_sha256,
        )
    elapsed = time.perf_counter() - started
    line = stdout.strip().splitlines()[-1] if stdout.strip() else ""
    try:
        result = json.loads(line)
    except ValueError:
        note = f"evaluator produced no verdict: {stderr.strip()[:300]}"
        return None, "crash", {}, note, elapsed, None, None, request_sha256
    if not isinstance(result, dict):
        return (
            None, "crash", {}, "evaluator verdict must be an object", elapsed,
            None, None, request_sha256,
        )
    if "error" in result:
        return (
            None, "crash", {},
            f"evaluator rejected solution: {result['error']}",
            elapsed, None, None, request_sha256,
        )
    normalized = result.get("normalized_solution")
    if normalized is None and result.get("normalized_A") is not None:
        # Backward compatibility for task evaluators using the original
        # normalized_A response contract.
        normalized = {"A": result["normalized_A"]}
    metrics = result.get("metrics", {})
    if not isinstance(metrics, dict):
        return (
            None, "crash", {}, "evaluator metrics must be an object", elapsed,
            None, None, request_sha256,
        )
    try:
        score = float(result["score"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return (
            None, "crash", {}, "evaluator score must be numeric", elapsed,
            None, None, request_sha256,
        )
    if not math.isfinite(score):
        return (
            None, "crash", {}, "evaluator score must be finite", elapsed,
            None, None, request_sha256,
        )
    return (
        score, "ok", metrics, "", elapsed,
        normalized, result.get("evidence"), request_sha256,
    )


def evaluate_trusted_artifact(
        task, artifact_path, trusted_dir, evaluation_request,
        cancel_event=None):
    """Evaluate an already-frozen artifact without running candidate code."""
    trusted_dir = Path(trusted_dir)
    if trusted_dir.exists():
        shutil.rmtree(trusted_dir)
    trusted_dir.mkdir(parents=True)
    max_artifact_bytes = int(getattr(
        task, "max_artifact_bytes", DEFAULT_MAX_ARTIFACT_BYTES,
    ))
    try:
        snapshot, snapshot_bytes = _snapshot_artifact(
            artifact_path, trusted_dir, max_artifact_bytes,
        )
    except (OSError, ValueError) as exc:
        return {
            "score": None,
            "status": "crash",
            "metrics": {},
            "note": f"could not freeze trusted artifact: {exc}",
            "artifact_sha256": None,
            "request_sha256": None,
        }
    artifact_sha256 = hashlib.sha256(snapshot_bytes).hexdigest()
    (
        score, status, metrics, note, evaluator_seconds, normalized, evidence,
        request_sha256,
    ) = _trusted_score(
        task, snapshot, cancel_event,
        evaluation_request=evaluation_request,
        trusted_dir=trusted_dir,
    )
    metrics = dict(metrics)
    metrics.update({
        "evaluator_seconds": evaluator_seconds,
        "artifact_sha256": artifact_sha256,
        "evaluation_request_sha256": request_sha256,
        "evaluation_stage": evaluation_request["stage"],
        "evaluation_suite_id": evaluation_request["suite_id"],
    })
    return {
        "score": score,
        "status": status,
        "metrics": metrics,
        "note": note,
        "artifact_sha256": artifact_sha256,
        "request_sha256": request_sha256,
        "normalized_solution": normalized,
        "evidence": evidence,
    }


def _apply_formalization_verdict(task, normalized, evidence, metrics):
    """Run an optional task-owned proof gate and promote only trusted claims."""
    research = (
        normalized.get("research")
        if isinstance(normalized, dict) else None
    )
    if not isinstance(research, dict):
        return
    request = research.get("formalization")
    if request is None:
        return
    formal_started = time.perf_counter()
    verifier = getattr(task, "verify_formalization", None)
    config = getattr(task, "formalization", {}) or {}
    if not callable(verifier):
        verdict = {
            "target": "lean4",
            "status": "unavailable",
            "reason": "task_formalizer_not_configured",
        }
    else:
        try:
            verdict = verifier(
                request,
                research.get("claims", []),
                runner=getattr(task, "formal_runner", None),
                trusted_files=getattr(task, "formal_spec_files", {}),
                command_prefix=tuple(
                    config.get(
                        "command_prefix", ["lake", "env", "lean"],
                    )
                ),
                toolchain=config.get("toolchain"),
                mathlib_revision=config.get("mathlib_revision"),
                allowed_axioms=tuple(config.get(
                    "allowed_axioms",
                    ["Classical.choice", "Quot.sound", "propext"],
                )),
            )
        except Exception as exc:
            verdict = {
                "target": "lean4",
                "status": "infrastructure_error",
                "reason": "formal_verifier_exception",
                "failure": {
                    "phase": "trusted_parent",
                    "detail": repr(exc)[:4000],
                },
            }
    metrics["formal_verifier_seconds"] = (
        time.perf_counter() - formal_started
    )

    if not isinstance(verdict, dict) or verdict.get("status") not in {
        "not_submitted",
        "unavailable",
        "rejected",
        "verified",
        "infrastructure_error",
    }:
        verdict = {
            "target": "lean4",
            "status": "infrastructure_error",
            "reason": "invalid_formal_verifier_verdict",
        }
    research_evidence = evidence.setdefault("research", {})
    claims = {
        item.get("id"): item
        for item in research_evidence.get("claims", [])
        if isinstance(item, dict)
    }
    requested_proofs = {}
    for item in request.get("proofs", []):
        if (
            isinstance(item, dict)
            and isinstance(item.get("claim_id"), str)
            and isinstance(item.get("term"), str)
        ):
            requested_proofs[item["claim_id"]] = hashlib.sha256(
                item["term"].strip().encode("utf-8")
            ).hexdigest()
    expected_binding = None
    binding_error = None
    validator = getattr(task, "validate_formalization_request", None)
    wrapper_builder = getattr(task, "build_formalization_wrapper", None)
    audit_builder = getattr(task, "build_formalization_audit", None)
    if (
        callable(validator)
        and callable(wrapper_builder)
        and callable(audit_builder)
    ):
        try:
            sealed_request, theorem_types = validator(
                request,
                research.get("claims", []),
                allow_sealed_hashes=True,
            )
            wrapper, theorem_names = wrapper_builder(
                sealed_request,
                theorem_types,
                trusted_spec_source=getattr(
                    task, "formal_spec_files", {}
                ).get("OpenHyraSumDiff/Spec.lean"),
            )
            audit = audit_builder(theorem_names)
            expected_binding = {
                "wrapper_sha256": hashlib.sha256(wrapper).hexdigest(),
                "audit_sha256": hashlib.sha256(audit).hexdigest(),
                "theorem_types": dict(theorem_types),
                "theorem_names": dict(theorem_names),
            }
        except Exception as exc:
            binding_error = repr(exc)[:4_000]
    if verdict.get("status") == "verified":
        promotion_error = None
        verified_payload = verdict.get("verified_claim_ids")
        verified_ids_valid = (
            isinstance(verified_payload, list)
            and all(isinstance(item, str) for item in verified_payload)
            and len(verified_payload) == len(set(verified_payload))
        )
        if (
            not requested_proofs
            or not verified_ids_valid
            or set(verified_payload) != set(requested_proofs)
        ):
            promotion_error = "formal_verifier_claim_set_mismatch"
        expected_request_hash = metrics.get(
            "formalization_request_sha256"
        )
        if (
            promotion_error is None
            and (
                not isinstance(expected_request_hash, str)
                or verdict.get("request_sha256") != expected_request_hash
            )
        ):
            promotion_error = "formal_verifier_request_hash_mismatch"
        verdict_proofs = verdict.get("proofs")
        verdict_proof_map = {}
        if isinstance(verdict_proofs, list):
            for item in verdict_proofs:
                if (
                    not isinstance(item, dict)
                    or not isinstance(item.get("claim_id"), str)
                    or not isinstance(item.get("proof_sha256"), str)
                    or item["claim_id"] in verdict_proof_map
                ):
                    verdict_proof_map = {}
                    break
                verdict_proof_map[item["claim_id"]] = item["proof_sha256"]
        if (
            promotion_error is None
            and verdict_proof_map != requested_proofs
        ):
            promotion_error = "formal_verifier_proof_hash_mismatch"
        require_environment_binding = bool(
            config.get("mathlib_revision")
        )
        if (
            promotion_error is None
            and require_environment_binding
            and expected_binding is None
        ):
            promotion_error = "formal_claim_binding_unavailable"
        if (
            promotion_error is None
            and expected_binding is not None
            and (
                verdict.get("wrapper_sha256")
                != expected_binding["wrapper_sha256"]
                or verdict.get("audit_sha256")
                != expected_binding["audit_sha256"]
                or verdict.get("theorem_types")
                != expected_binding["theorem_types"]
                or verdict.get("theorem_names")
                != expected_binding["theorem_names"]
            )
        ):
            promotion_error = "formal_claim_binding_mismatch"
        runner_identity = getattr(task, "formal_runner_identity", None)
        spec_sha256 = getattr(task, "formal_spec_sha256", None)
        runtime_attestation = verdict.get("runtime_attestation")
        if (
            promotion_error is None
            and require_environment_binding
            and (
                not isinstance(spec_sha256, str)
                or not _is_sha256(spec_sha256)
                or not isinstance(runner_identity, dict)
                or not _is_sha256(runner_identity.get("sha256"))
                or not isinstance(runtime_attestation, dict)
                or any(
                    not _is_sha256(runtime_attestation.get(field))
                    for field in (
                        "environment_sha256",
                        "lean_binary_sha256",
                        "mathlib_tree_sha256",
                    )
                )
                or runtime_attestation.get("toolchain")
                != config.get("toolchain")
                or runtime_attestation.get("mathlib_revision")
                != config.get("mathlib_revision")
            )
        ):
            promotion_error = "formal_environment_binding_mismatch"
        if (
            promotion_error is None
            and (
                any(claim_id not in claims for claim_id in requested_proofs)
                or any(
                    claims[claim_id].get("status") == "refuted"
                    for claim_id in requested_proofs
                )
            )
        ):
            promotion_error = (
                "formal_proof_conflicts_with_trusted_refutation"
            )
        if promotion_error is not None:
            verdict = {
                **verdict,
                "status": "infrastructure_error",
                "reason": promotion_error,
                **(
                    {
                        "failure": {
                            "phase": "trusted_parent_binding",
                            "detail": binding_error,
                        },
                    }
                    if binding_error is not None else {}
                ),
            }
    runner_identity = getattr(task, "formal_runner_identity", None)
    trusted_environment = {
        "spec_sha256": getattr(task, "formal_spec_sha256", None),
        "runner_sha256": (
            runner_identity.get("sha256")
            if isinstance(runner_identity, dict) else None
        ),
        "toolchain": (
            (getattr(task, "formalization", {}) or {}).get("toolchain")
        ),
        "mathlib_revision": (
            (getattr(task, "formalization", {}) or {}).get(
                "mathlib_revision"
            )
        ),
        "runtime_attestation": (
            verdict.get("runtime_attestation")
            if isinstance(verdict.get("runtime_attestation"), dict)
            else None
        ),
    }
    verdict = {
        **verdict,
        "trusted_environment": trusted_environment,
    }
    metrics["formal_spec_sha256"] = trusted_environment["spec_sha256"]
    metrics["formal_runner_sha256"] = trusted_environment["runner_sha256"]
    metrics["formal_toolchain"] = trusted_environment["toolchain"]
    metrics["formal_mathlib_revision"] = trusted_environment[
        "mathlib_revision"
    ]
    runtime_attestation = trusted_environment["runtime_attestation"] or {}
    metrics["formal_environment_sha256"] = runtime_attestation.get(
        "environment_sha256"
    )
    metrics["formal_lean_binary_sha256"] = runtime_attestation.get(
        "lean_binary_sha256"
    )
    metrics["formal_mathlib_tree_sha256"] = runtime_attestation.get(
        "mathlib_tree_sha256"
    )

    research_evidence["formalization"] = verdict
    metrics["formalization_status"] = verdict.get("status", "infrastructure_error")
    for source, destination in (
        ("request_sha256", "formalization_request_sha256"),
        ("wrapper_sha256", "formal_wrapper_sha256"),
        ("audit_sha256", "formal_audit_sha256"),
    ):
        if verdict.get(source):
            metrics[destination] = verdict[source]
    proof_hashes = sorted(requested_proofs.values())
    if proof_hashes:
        metrics["proof_sha256"] = hashlib.sha256(
            json.dumps(proof_hashes, separators=(",", ":")).encode()
        ).hexdigest()

    verified_ids = set(verdict.get("verified_claim_ids", []))
    has_refutation = (
        any(
            isinstance(metrics.get(field), int)
            and not isinstance(metrics.get(field), bool)
            and metrics.get(field) > 0
            for field in (
                "refuted_claim_count",
                "refuted_obligation_count",
                "refuted_certificate_count",
            )
        )
        or any(
            claim.get("status") == "refuted"
            for claim in claims.values()
        )
        or (
            isinstance(research_evidence.get("construction"), dict)
            and research_evidence["construction"].get("status")
            == "contains_refutation"
        )
        or any(
            isinstance(item, dict)
            and item.get("status") == "refuted"
            for item in research_evidence.get("certificates", [])
        )
    )
    if verdict.get("status") == "verified":
        # Promotion is atomic: all ids, hashes and refutation checks above
        # succeed before any claim status is changed.
        for claim_id in verified_ids:
            claim = claims.get(claim_id)
            claim["status"] = "formal_checked"
            claim["formalization_request_sha256"] = verdict.get(
                "request_sha256"
            )
        if has_refutation:
            research_evidence["status"] = (
                "formal_checked_with_refutation"
            )
            metrics["research_rank"] = max(
                metrics.get("research_rank", 0), 70
            )
            metrics["evidence_level"] = (
                "formal_checked_with_refutation"
            )
        else:
            research_evidence["status"] = "formal_checked"
            metrics["research_rank"] = max(
                metrics.get("research_rank", 0), 80
            )
            metrics["evidence_level"] = "formal_checked"
    elif verdict.get("status") == "infrastructure_error":
        if has_refutation:
            research_evidence["status"] = "contains_refutation"
        else:
            research_evidence["status"] = "infrastructure_error"
            metrics["research_rank"] = -1

    formally_checked = [
        item for item in claims.values()
        if item.get("status") == "formal_checked"
    ]
    refuted = [
        item for item in claims.values()
        if item.get("status") == "refuted"
    ]
    bounded = [
        item for item in claims.values()
        if item.get("status") == "bounded_supported"
    ]
    metrics["formally_checked_claim_count"] = len(formally_checked)
    metrics["refuted_claim_count"] = len(refuted)
    metrics["bounded_supported_claim_count"] = len(bounded)
    metrics["formal_checked_claim_templates"] = sorted({
        item.get("template")
        for item in formally_checked
        if isinstance(item.get("template"), str)
    })
    metrics["formal_checked_targets"] = [
        {
            "claim_id": item.get("id"),
            "template": item.get("template"),
            "target": item.get("target"),
        }
        for item in sorted(
            formally_checked,
            key=lambda claim: str(claim.get("id")),
        )
        if isinstance(item.get("target"), dict)
    ]


def run_solution(solution_dir: Path, sandbox_dir: Path, task):
    """Run a candidate, kill its process group, snapshot output, then score."""
    total_started = time.perf_counter()
    cancel_event = getattr(task, "cancel_event", None)
    if cancel_event is not None and cancel_event.is_set():
        return None, "cancelled", "candidate cancelled before solver launch", {
            "solver_seconds": 0.0,
        }
    sandbox_dir = Path(sandbox_dir)
    if sandbox_dir.exists():
        shutil.rmtree(sandbox_dir)
    trusted_dir = trusted_artifact_dir(sandbox_dir)
    if trusted_dir.exists():
        shutil.rmtree(trusted_dir)
    trusted_dir.mkdir(parents=True)
    max_source_bytes = int(
        getattr(task, "max_source_bytes", 0)
        or int(getattr(task, "max_output_mb", 64)) * 1024 * 1024
    )
    try:
        source_snapshot_sha256, _source_hashes = snapshot_source_tree(
            solution_dir, trusted_dir / "source", max_source_bytes,
        )
        shutil.copytree(trusted_dir / "source", sandbox_dir)
    except (OSError, ValueError) as exc:
        return None, "crash", f"could not seal candidate source: {exc}", {}
    tmp_dir = sandbox_dir / ".tmp"
    tmp_dir.mkdir()
    log_path = sandbox_dir / "run.log"

    env = {
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "HOME": str(sandbox_dir),
        "TMPDIR": str(tmp_dir),
        "OPENHYRA_PYTHON": task.python_bin,
        "PYTHONDONTWRITEBYTECODE": "1",
        **NUMERIC_THREAD_ENV,
    }
    try:
        command = _limited_cmd(task, _sandboxed_cmd(
            sandbox_dir, task.evaluator, ["bash", "solve.sh"],
        ))
    except RuntimeError as exc:
        return None, "crash", str(exc), {}

    solver_started = time.perf_counter()
    wait_state = "completed"
    with open(log_path, "w") as log_stream:
        proc = subprocess.Popen(
            command, cwd=sandbox_dir, env=env,
            stdout=log_stream, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            wait_state = _wait_process(proc, task.timeout_s, cancel_event)
        finally:
            # Also removes descendants deliberately left behind after a normal
            # parent exit, closing the artifact mutation race before snapshot.
            _kill_process_group(proc)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
    solver_seconds = time.perf_counter() - solver_started

    log = log_path.read_text(errors="replace") if log_path.exists() else ""
    log_tail = "\n".join(log.replace("\r", "\n").splitlines()[-15:])
    base_metrics = {
        "solver_seconds": solver_seconds,
        "source_snapshot_sha256": source_snapshot_sha256,
    }
    if wait_state == "cancelled":
        return None, "cancelled", (
            f"cancelled solver process group\n{log_tail}"
        ).strip(), base_metrics
    if wait_state == "timeout":
        return None, "timeout", (
            f"killed process group after {task.timeout_s}s\n{log_tail}"
        ).strip(), base_metrics
    if proc.returncode != 0:
        return None, "crash", log_tail, base_metrics

    artifact = sandbox_dir / "solution.json"
    max_artifact_bytes = int(getattr(
        task, "max_artifact_bytes", DEFAULT_MAX_ARTIFACT_BYTES,
    ))
    try:
        snapshot, snapshot_bytes = _snapshot_artifact(
            artifact, trusted_dir, max_artifact_bytes,
        )
    except (OSError, ValueError) as exc:
        return None, "crash", (log_tail + f"\n{exc}").strip(), base_metrics
    candidate_artifact_sha256 = hashlib.sha256(snapshot_bytes).hexdigest()

    (
        score, status, metrics, note, evaluator_seconds, normalized, evidence,
        request_sha256,
    ) = _trusted_score(
        task, snapshot, cancel_event,
        evaluation_request=getattr(task, "search_evaluation_request", None),
        trusted_dir=trusted_dir,
    )
    if normalized is not None and evidence is not None:
        _apply_formalization_verdict(
            task, normalized, evidence, metrics,
        )
    evaluated_artifact_sha256 = candidate_artifact_sha256
    if normalized is not None:
        evaluated_bytes = json.dumps(
            normalized,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode()
        evaluated = trusted_dir / "evaluated_solution.json"
        evaluated.write_bytes(evaluated_bytes)
        evaluated.chmod(0o444)
        evaluated_artifact_sha256 = hashlib.sha256(evaluated_bytes).hexdigest()
    if evidence is not None:
        evidence_bytes = json.dumps(
            evidence,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode()
        evidence_path = trusted_dir / "evidence.json"
        evidence_path.write_bytes(evidence_bytes)
        evidence_path.chmod(0o444)
        metrics["evidence_sha256"] = hashlib.sha256(evidence_bytes).hexdigest()
    metrics.update(base_metrics)
    metrics.update({
        "evaluator_seconds": evaluator_seconds,
        "total_seconds": time.perf_counter() - total_started,
        "candidate_artifact_sha256": candidate_artifact_sha256,
        "artifact_sha256": evaluated_artifact_sha256,
    })
    request = getattr(task, "search_evaluation_request", None)
    if request is not None:
        metrics.update({
            "evaluation_request_sha256": request_sha256,
            "evaluation_stage": request["stage"],
            "evaluation_suite_id": request["suite_id"],
        })
    if note:
        log_tail = (log_tail + "\n[evaluator] " + note).strip()
    return score, status, log_tail, metrics
