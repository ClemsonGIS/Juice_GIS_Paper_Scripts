"""
Run an exported ArcGIS ModelBuilder Python script while logging GPU usage per
TrainDeepLearningModel call.

Reproducibility notes:
- Input: one ModelBuilder-exported Python script path passed on the command
  line.
- Outputs under RESULTS_ROOT/RUN_ID:
  results_master.csv records timing, status, and GPU summaries for each tracked
  tool call; gpu_log_*.csv records sampled GPU utilization and memory; and
  gpu_proc_*.log records sampled compute-process memory.
- Configure RESULTS_ROOT, or PROJECT_ROOT for the default results folder, before
  running:
  python train_mb_monkey_wrapper.py PythonTest_ModelBuilder.py
"""
import ast
import contextlib
import csv
import datetime
import os
import re
import runpy
import subprocess
import sys
import time
from pathlib import Path

# CONFIG
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path.cwd()))
RESULTS_ROOT = Path(os.environ.get("RESULTS_ROOT", PROJECT_ROOT / "results"))
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
RUN_ID = datetime.datetime.now(datetime.UTC).strftime("Run_%Y-%m-%d_T%H-%M-%SZ")
RUN_DIR = RESULTS_ROOT / RUN_ID
RUN_DIR.mkdir(parents=True, exist_ok=True)
MASTER_CSV = RUN_DIR / "results_master.csv"
ALLOWED_TOOL_NAMES = {"TrainDeepLearningModel", "TrainDeepLearningModel_ia"}

POSSIBLES = [
    str(
        Path(os.environ.get("ProgramFiles", "Program Files"))
        / "NVIDIA Corporation"
        / "NVSMI"
        / "nvidia-smi.exe"
    ),
    str(Path(os.environ.get("SystemRoot", "Windows")) / "System32" / "nvidia-smi.exe"),
    "nvidia-smi",
]
NVIDIA_SMI = None
for p in POSSIBLES:
    try:
        if p.lower() == "nvidia-smi":
            subprocess.run(
                [p, "-h"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
            NVIDIA_SMI = p
            break
        if Path(p).exists():
            NVIDIA_SMI = p
            break
    except Exception:
        pass
print("Using nvidia-smi ->", NVIDIA_SMI)
print("Run output directory ->", RUN_DIR)


def _safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._-") or "tool"


def _extract_tool_paths(exported_script: Path):
    src = exported_script.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(exported_script))
    tool_paths = set()

    def _call_path(node):
        parts = []
        cur = node
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            parts.append(cur.id)
            parts.reverse()
            if parts and parts[0] == "arcpy":
                return ".".join(parts)
        return None

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            path = _call_path(node.func)
            if path:
                tool_paths.add(path)
    return sorted(tool_paths)


def _start_gpu_logging(exp_name: str):
    """Start two nvidia-smi loggers. One writes global GPU stats to a csv file.
    The other logs compute-apps to a rotating log file to avoid PIPE buffer issues.
    Returns handles dict with file handles so stop can close/flush cleanly.
    """
    gpu_csv = RUN_DIR / f"gpu_log_{exp_name}.csv"
    proc_log = RUN_DIR / f"gpu_proc_{exp_name}.log"
    baseline_pids = _get_compute_app_pids()
    if NVIDIA_SMI is None:
        return {
            "p_gpu": None,
            "p_proc": None,
            "gpu_csv": gpu_csv,
            "proc_log": proc_log,
            "baseline_pids": baseline_pids,
        }

    cmd_gpu = [
        NVIDIA_SMI,
        "--query-gpu=timestamp,utilization.gpu,memory.used",
        "--format=csv",
        "-l",
        "1",
    ]
    # make sure file is line-buffered and encoded
    fh_gpu = open(gpu_csv, "w", buffering=1, encoding="utf-8", newline="")
    # compute-apps go to a logfile (so the child never blocks on a PIPE)
    fh_proc = open(proc_log, "w", buffering=1, encoding="utf-8", newline="")
    cmd_proc = [
        NVIDIA_SMI,
        "--query-compute-apps=timestamp,pid,process_name,used_memory",
        "--format=csv",
        "-l",
        "1",
    ]
    p_gpu = subprocess.Popen(cmd_gpu, stdout=fh_gpu, stderr=subprocess.STDOUT)
    p_proc = subprocess.Popen(cmd_proc, stdout=fh_proc, stderr=subprocess.DEVNULL, text=True)
    return {
        "p_gpu": p_gpu,
        "p_proc": p_proc,
        "fh_gpu": fh_gpu,
        "fh_proc": fh_proc,
        "gpu_csv": gpu_csv,
        "proc_log": proc_log,
        "baseline_pids": baseline_pids,
    }


def _stop_gpu_logging(handles):
    """Stop subprocesses and return proc log contents as a string."""
    proc_output = ""
    if not handles:
        return proc_output
    # terminate processes if running
    if handles.get("p_gpu") is not None:
        try:
            handles["p_gpu"].terminate()
        except Exception:
            pass
    if handles.get("p_proc") is not None:
        try:
            handles["p_proc"].terminate()
        except Exception:
            pass
    # give processes a moment to exit and flush
    time.sleep(0.5)
    # close any open fh's (which forces child to flush)
    if handles.get("fh_gpu") is not None:
        try:
            handles["fh_gpu"].close()
        except Exception:
            pass
    if handles.get("fh_proc") is not None:
        try:
            handles["fh_proc"].close()
        except Exception:
            pass
    # try to join processes cleanly; if they refuse, kill them
    if handles.get("p_proc") is not None:
        try:
            handles["p_proc"].wait(timeout=1)
        except Exception:
            try:
                handles["p_proc"].kill()
            except Exception:
                pass
    if handles.get("p_gpu") is not None:
        try:
            handles["p_gpu"].wait(timeout=1)
        except Exception:
            try:
                handles["p_gpu"].kill()
            except Exception:
                pass
    # read proc log file (if there is one)
    proc_log = handles.get("proc_log")
    if proc_log and proc_log.exists():
        try:
            proc_output = proc_log.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            proc_output = ""
    return proc_output


def _parse_gpu_csv(gpu_csv: Path):
    """Read the GPU CSV file defensively and compute average util and max mem."""
    util_values = []
    mem_values = []
    if not gpu_csv.exists():
        return "", ""
    try:
        with open(gpu_csv, newline="", encoding="utf-8", errors="ignore") as fh:
            reader = csv.reader(fh)
            for row in reader:
                try:
                    if not row:
                        continue
                    joined = ",".join(row).lower()
                    if "timestamp" in joined:
                        continue
                    if len(row) >= 2:
                        util_text = "".join(ch for ch in row[1] if ch.isdigit() or ch == ".")
                        if util_text:
                            util_values.append(float(util_text))
                    if len(row) >= 3:
                        mem_text = "".join(ch for ch in row[2] if ch.isdigit() or ch == ".")
                        if mem_text:
                            mem_values.append(float(mem_text))
                except Exception:
                    # skip malformed line but keep parsing
                    continue
    except Exception:
        return "", ""
    avg_util = round(sum(util_values) / len(util_values), 2) if util_values else ""
    max_mem = int(max(mem_values)) if mem_values else ""
    return avg_util, max_mem


def _get_compute_app_pids():
    if NVIDIA_SMI is None:
        return set()
    cmd = [
        NVIDIA_SMI,
        "--query-compute-apps=pid",
        "--format=csv,noheader,nounits",
    ]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if p.returncode != 0:
            return set()
        out = set()
        for line in p.stdout.splitlines():
            s = line.strip()
            if s.isdigit():
                out.add(int(s))
        return out
    except Exception:
        return set()


def _parse_avg_filtered_tool_memory(proc_output: str, baseline_pids):
    """Parse compute-apps proc log defensively; ignore malformed lines."""
    if not proc_output:
        return ""
    current_pid = os.getpid()
    per_timestamp_sum = {}
    try:
        reader = csv.reader(proc_output.splitlines())
        for row in reader:
            try:
                if len(row) < 4:
                    continue
                joined = ",".join(row).lower()
                if "timestamp" in joined and "pid" in joined:
                    continue
                ts = row[0].strip()
                pid_text = "".join(ch for ch in row[1] if ch.isdigit())
                mem_text = "".join(ch for ch in row[3] if ch.isdigit() or ch == ".")
                if not pid_text or not mem_text:
                    continue
                pid = int(pid_text)
                mem = float(mem_text)
                include = (pid == current_pid) or (pid not in baseline_pids)
                if include:
                    per_timestamp_sum[ts] = per_timestamp_sum.get(ts, 0.0) + mem
            except Exception:
                continue
    except Exception:
        return ""
    if not per_timestamp_sum:
        return ""
    return round(sum(per_timestamp_sum.values()) / len(per_timestamp_sum), 2)


def _append_master_row(row):
    header = [
        "run_id",
        "seq",
        "exp_name",
        "tool_path",
        "start_time_utc",
        "end_time_utc",
        "wall_time_seconds",
        "avg_global_gpu_util_percent",
        "avg_filtered_tool_gpu_mem_mb",
        "status",
        "notes",
        "gpu_log",
    ]
    write_header = not MASTER_CSV.exists()
    with open(MASTER_CSV, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=header)
        if write_header:
            w.writeheader()
        w.writerow(row)


def _resolve_attr_path(root_obj, full_path: str):
    parts = full_path.split(".")
    if not parts or parts[0] != "arcpy":
        return None, None

    obj = root_obj
    for part in parts[1:-1]:
        if not hasattr(obj, part):
            return None, None
        obj = getattr(obj, part)
    return obj, parts[-1]


class _FilteredConsole:
    """Filter duplicate epoch-detail lines while preserving normal training progress output."""

    def __init__(self, target):
        self.target = target
        self._buffer = ""
        self._drop_re = re.compile(r"^\d+\s+[-+]?\d+\.\d+\s+[-+]?\d+\.\d+.*\d{2}:\d{2}:\d{2}\s*$")

    def write(self, text):
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._emit(line + "\n")
        return len(text)

    def flush(self):
        if self._buffer:
            self._emit(self._buffer)
            self._buffer = ""
        self.target.flush()

    def _emit(self, line):
        if self._drop_re.match(line.strip()):
            return
        self.target.write(line)
        self.target.flush()


def main(exported_script_path: str):
    exported = Path(exported_script_path)
    if not exported.exists():
        print("Exported script not found:", exported)
        sys.exit(1)

    try:
        import arcpy
        arcpy.env.overwriteOutput = True
    except Exception as e:
        print("Could not import arcpy. Run with ArcGIS Pro python. Exception:", e)
        sys.exit(1)

    tool_paths = _extract_tool_paths(exported)
    if not tool_paths:
        print("No arcpy calls found in exported script.")
    else:
        print("Discovered arcpy call targets:")
        for p in tool_paths:
            print(" -", p)

    tracked_tool_paths = []
    for p in tool_paths:
        leaf = p.split(".")[-1]
        if leaf in ALLOWED_TOOL_NAMES:
            tracked_tool_paths.append(p)
    if not tracked_tool_paths:
        print("No matching tracked tools found. Current filter:", sorted(ALLOWED_TOOL_NAMES))
    else:
        print("Tracked tool targets:")
        for p in tracked_tool_paths:
            print(" -", p)

    originals = {}
    call_state = {"seq": 0}
    skip_paths = {"arcpy.ImportToolbox", "arcpy.EnvManager"}

    def _make_wrapper(tool_path, original_func):
        def _wrapped(*args, **kwargs):
            call_state["seq"] += 1
            seq = call_state["seq"]
            base_name = tool_path.split(".")[-1]
            exp_name = f"{seq:03d}_{_safe_name(base_name)}"

            print(f"[{exp_name}] Start {tool_path}")
            t0 = time.time()
            handles = _start_gpu_logging(exp_name)
            status = "OK"
            notes = ""
            try:
                if os.environ.get("WRAPPER_NO_FILTER", ""):
                    result = original_func(*args, **kwargs)
                else:
                    filtered_stdout = _FilteredConsole(sys.stdout)
                    filtered_stderr = _FilteredConsole(sys.stderr)
                    with contextlib.redirect_stdout(
                        filtered_stdout
                    ), contextlib.redirect_stderr(filtered_stderr):
                        result = original_func(*args, **kwargs)
            except Exception as exc:
                status = "ERROR"
                notes = str(exc)
                proc_output = _stop_gpu_logging(handles)
                t1 = time.time()
                avg_util, _ = _parse_gpu_csv(handles["gpu_csv"])
                avg_filtered_mem = _parse_avg_filtered_tool_memory(
                    proc_output, handles.get("baseline_pids", set())
                )
                _append_master_row(
                    {
                        "run_id": RUN_ID,
                        "seq": seq,
                        "exp_name": exp_name,
                        "tool_path": tool_path,
                        "start_time_utc": datetime.datetime.fromtimestamp(
                            t0, datetime.UTC
                        ).isoformat().replace("+00:00", "Z"),
                        "end_time_utc": datetime.datetime.fromtimestamp(
                            t1, datetime.UTC
                        ).isoformat().replace("+00:00", "Z"),
                        "wall_time_seconds": int(t1 - t0),
                        "avg_global_gpu_util_percent": avg_util,
                        "avg_filtered_tool_gpu_mem_mb": avg_filtered_mem,
                        "status": status,
                        "notes": notes,
                        "gpu_log": str(handles["gpu_csv"]),
                    }
                )
                print(f"[{exp_name}] ERROR {notes}")
                raise

            proc_output = _stop_gpu_logging(handles)
            t1 = time.time()
            avg_util, _ = _parse_gpu_csv(handles["gpu_csv"])
            avg_filtered_mem = _parse_avg_filtered_tool_memory(
                proc_output, handles.get("baseline_pids", set())
            )
            _append_master_row(
                {
                    "run_id": RUN_ID,
                    "seq": seq,
                    "exp_name": exp_name,
                    "tool_path": tool_path,
                    "start_time_utc": datetime.datetime.fromtimestamp(
                        t0, datetime.UTC
                    ).isoformat().replace("+00:00", "Z"),
                    "end_time_utc": datetime.datetime.fromtimestamp(
                        t1, datetime.UTC
                    ).isoformat().replace("+00:00", "Z"),
                    "wall_time_seconds": int(t1 - t0),
                    "avg_global_gpu_util_percent": avg_util,
                    "avg_filtered_tool_gpu_mem_mb": avg_filtered_mem,
                    "status": status,
                    "notes": notes,
                    "gpu_log": str(handles["gpu_csv"]),
                }
            )
            print(f"[{exp_name}] Done {tool_path}")
            return result

        return _wrapped

    def _install_wrappers():
        installed = 0
        for tool_path in tracked_tool_paths:
            if tool_path in skip_paths:
                continue
            owner, attr_name = _resolve_attr_path(arcpy, tool_path)
            if owner is None:
                continue
            if not hasattr(owner, attr_name):
                continue
            current = getattr(owner, attr_name)
            if not callable(current):
                continue
            if tool_path not in originals:
                originals[tool_path] = current
            setattr(owner, attr_name, _make_wrapper(tool_path, originals[tool_path]))
            installed += 1
        return installed

    original_import_toolbox = arcpy.ImportToolbox

    def _import_toolbox_wrapper(*args, **kwargs):
        result = original_import_toolbox(*args, **kwargs)
        installed = _install_wrappers()
        print(f"Re-applied wrappers after ImportToolbox ({installed} installed).")
        return result

    arcpy.ImportToolbox = _import_toolbox_wrapper
    initially_installed = _install_wrappers()
    print(f"Installed tool wrappers: {initially_installed}")
    print("Executing exported ModelBuilder script...")

    try:
        runpy.run_path(str(exported), run_name="__main__")
    except SystemExit as se:
        print("Exported script called sys.exit()", se)
    except Exception as e:
        print("Error executing exported script:", e)
        raise

    print("Done. Master CSV at:", MASTER_CSV)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: <ArcGIS Python exe> train_mb_monkey_wrapper.py "
            "<exported_modelbuilder_script.py>"
        )
        sys.exit(1)
    main(sys.argv[1])
