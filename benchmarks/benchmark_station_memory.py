# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Characterize DGX Station GB300 memory tiers for weight offloading.

The benchmark compiles and runs a native CUDA probe, then combines its
measurements with host and NVIDIA driver inventory. It intentionally refuses
to run on GPUs other than SM103 so results cannot be mistaken for DGX Station
measurements.

Run with the repository virtual environment:

    .venv/bin/python benchmarks/benchmark_station_memory.py \
        --output station-memory.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any

SCHEMA_VERSION = 1
EXPECTED_COMPUTE_CAPABILITY = (10, 3)
_NVIDIA_SMI_FIELDS = (
    "index",
    "name",
    "uuid",
    "pci.bus_id",
    "driver_version",
    "memory.total",
    "memory.free",
    "memory.used",
)
_MEMINFO_UNITS = {
    "B": 1,
    "kB": 1024,
    "MB": 1024**2,
    "GB": 1024**3,
}


class HarnessError(RuntimeError):
    """Raised when the host cannot produce a valid Station result."""


def parse_meminfo(text: str) -> dict[str, int]:
    """Parse Linux meminfo text into byte counts or unitless integers."""
    values: dict[str, int] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, raw_value = line.split(":", maxsplit=1)
        fields = raw_value.split()
        if not fields:
            continue
        try:
            value = int(fields[0])
        except ValueError:
            continue
        if len(fields) > 1:
            multiplier = _MEMINFO_UNITS.get(fields[1])
            if multiplier is None:
                continue
            value *= multiplier
        values[key.strip()] = value
    return values


def parse_numa_meminfo(text: str) -> dict[str, int]:
    """Parse one Linux NUMA node's meminfo into byte counts."""
    normalized_lines = []
    for line in text.splitlines():
        fields = line.split(maxsplit=2)
        if len(fields) == 3 and fields[0] == "Node":
            normalized_lines.append(fields[2])
        else:
            normalized_lines.append(line)
    return parse_meminfo("\n".join(normalized_lines))


def parse_nvidia_smi_csv(
    text: str,
    fields: Sequence[str] = _NVIDIA_SMI_FIELDS,
) -> list[dict[str, str]]:
    """Parse ``nvidia-smi --format=csv,noheader,nounits`` output."""
    rows = []
    for values in csv.reader(text.splitlines(), skipinitialspace=True):
        if not values:
            continue
        if len(values) != len(fields):
            raise HarnessError(
                "nvidia-smi returned "
                f"{len(values)} columns; expected {len(fields)}: {values!r}"
            )
        rows.append(dict(zip(fields, (value.strip() for value in values))))
    return rows


def nearest_rank_percentile(samples: Sequence[float], percentile: float) -> float:
    """Return a nearest-rank percentile for a non-empty sample."""
    if not samples:
        raise ValueError("samples must not be empty")
    if not 0 < percentile <= 100:
        raise ValueError("percentile must be in (0, 100]")
    ordered = sorted(samples)
    rank = max(1, math.ceil(percentile / 100 * len(ordered)))
    return ordered[rank - 1]


def summarize_samples(
    samples_ms: Sequence[float],
    bytes_per_iteration: int,
    memory_traffic_factor: float,
) -> dict[str, float]:
    """Summarize timings and derive decimal GB/s rates."""
    if not samples_ms:
        raise ValueError("samples_ms must not be empty")
    if any(sample <= 0 for sample in samples_ms):
        raise ValueError("samples_ms values must be positive")
    if bytes_per_iteration <= 0:
        raise ValueError("bytes_per_iteration must be positive")
    if memory_traffic_factor <= 0:
        raise ValueError("memory_traffic_factor must be positive")

    mean_ms = fmean(samples_ms)
    payload_gb_s = bytes_per_iteration / (mean_ms * 1_000_000)
    return {
        "mean_ms": mean_ms,
        "min_ms": min(samples_ms),
        "max_ms": max(samples_ms),
        "p50_ms": nearest_rank_percentile(samples_ms, 50),
        "p95_ms": nearest_rank_percentile(samples_ms, 95),
        "payload_gb_s": payload_gb_s,
        "memory_traffic_gb_s": payload_gb_s * memory_traffic_factor,
    }


def build_overlap_assessment(
    h2d_mean_ms: float,
    hbm_read_mean_ms: float,
    observed_mean_ms: float,
) -> dict[str, float]:
    """Describe copy/compute overlap relative to serial and ideal timing."""
    if min(h2d_mean_ms, hbm_read_mean_ms, observed_mean_ms) <= 0:
        raise ValueError("overlap timings must be positive")
    serial_ms = h2d_mean_ms + hbm_read_mean_ms
    ideal_ms = max(h2d_mean_ms, hbm_read_mean_ms)
    return {
        "isolated_serial_ms": serial_ms,
        "ideal_fully_overlapped_ms": ideal_ms,
        "observed_ms": observed_mean_ms,
        "speedup_vs_serial": serial_ms / observed_mean_ms,
        "fraction_of_ideal_overlap": min(
            1.0,
            max(0.0, (serial_ms - observed_mean_ms) / (serial_ms - ideal_ms)),
        )
        if serial_ms > ideal_ms
        else 0.0,
    }


def validate_native_report(report: dict[str, Any]) -> None:
    """Ensure the native probe ran on the intended GPU."""
    try:
        capability = report["device"]["compute_capability"]
        actual = (int(capability["major"]), int(capability["minor"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise HarnessError(
            "native probe output omitted device compute capability"
        ) from exc
    if actual != EXPECTED_COMPUTE_CAPABILITY:
        expected = ".".join(map(str, EXPECTED_COMPUTE_CAPABILITY))
        found = ".".join(map(str, actual))
        raise HarnessError(
            f"unsupported GPU compute capability {found}; "
            f"DGX Station GB300 validation requires SM{expected.replace('.', '')}"
        )


def enrich_native_report(report: dict[str, Any]) -> None:
    """Add timing summaries and overlap analysis to native probe output."""
    by_name: dict[str, dict[str, Any]] = {}
    for benchmark in report.get("benchmarks", []):
        summary = summarize_samples(
            benchmark["samples_ms"],
            int(benchmark["bytes_per_iteration"]),
            float(benchmark["memory_traffic_factor"]),
        )
        benchmark["summary"] = summary
        by_name[benchmark["name"]] = benchmark

    required = (
        "registered_host_to_hbm_async",
        "hbm_kernel_read",
        "registered_host_to_hbm_plus_hbm_read",
    )
    if all(name in by_name for name in required):
        report["copy_compute_overlap"] = build_overlap_assessment(
            by_name[required[0]]["summary"]["mean_ms"],
            by_name[required[1]]["summary"]["mean_ms"],
            by_name[required[2]]["summary"]["mean_ms"],
        )


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text()
    except OSError:
        return None


def _run_command(command: Sequence[str], cwd: Path | None = None) -> str:
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise HarnessError(f"required command not found: {command[0]}") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout).strip()
        raise HarnessError(
            f"command failed ({' '.join(command)}): {detail or exc.returncode}"
        ) from exc
    return completed.stdout.strip()


def collect_host_inventory() -> dict[str, Any]:
    """Collect CPU, kernel, system-memory, NUMA, and huge-page inventory."""
    proc_meminfo = _read_text(Path("/proc/meminfo"))
    meminfo = parse_meminfo(proc_meminfo) if proc_meminfo else {}

    cpu_model = None
    cpuinfo = _read_text(Path("/proc/cpuinfo"))
    if cpuinfo:
        for line in cpuinfo.splitlines():
            if line.lower().startswith(("model name", "hardware")) and ":" in line:
                cpu_model = line.split(":", maxsplit=1)[1].strip()
                break

    numa_nodes = []
    node_root = Path("/sys/devices/system/node")
    for node_path in sorted(node_root.glob("node[0-9]*")):
        node_meminfo = _read_text(node_path / "meminfo")
        numa_nodes.append(
            {
                "node": node_path.name,
                "memory": (parse_numa_meminfo(node_meminfo) if node_meminfo else {}),
                "cpulist": _read_text(node_path / "cpulist"),
            }
        )

    return {
        "architecture": platform.machine(),
        "cpu_model": cpu_model,
        "logical_cpu_count": os.cpu_count(),
        "kernel": platform.release(),
        "platform": platform.platform(),
        "page_size_bytes": os.sysconf("SC_PAGE_SIZE"),
        "memory": meminfo,
        "numa_nodes": numa_nodes,
        "transparent_hugepage_enabled": _read_text(
            Path("/sys/kernel/mm/transparent_hugepage/enabled")
        ),
        "kernel_command_line": _read_text(Path("/proc/cmdline")),
    }


def collect_nvidia_inventory() -> dict[str, Any]:
    """Collect stable nvidia-smi fields and the topology matrix."""
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        raise HarnessError(
            "nvidia-smi not found; run this benchmark in an NVIDIA CUDA container"
        )
    query = ",".join(_NVIDIA_SMI_FIELDS)
    gpu_csv = _run_command(
        [
            nvidia_smi,
            f"--query-gpu={query}",
            "--format=csv,noheader,nounits",
        ]
    )
    topology = _run_command([nvidia_smi, "topo", "-m"])
    return {
        "gpus": parse_nvidia_smi_csv(gpu_csv),
        "topology_matrix": topology,
    }


def _git_inventory(repo_root: Path) -> dict[str, Any]:
    try:
        commit = _run_command(["git", "rev-parse", "HEAD"], cwd=repo_root)
        status = _run_command(["git", "status", "--porcelain"], cwd=repo_root)
    except HarnessError as exc:
        return {"error": str(exc)}
    return {"commit": commit, "dirty": bool(status)}


def _resolve_nvcc(value: str | None) -> str:
    candidate = value or os.environ.get("NVCC") or shutil.which("nvcc")
    if not candidate:
        raise HarnessError(
            "nvcc not found; use a CUDA 13.1 development image or pass --nvcc"
        )
    return candidate


def build_native_probe(
    nvcc: str,
    source: Path,
    output: Path,
) -> dict[str, Any]:
    """Compile the native probe for architecture-specific SM103 execution."""
    version = _run_command([nvcc, "--version"])
    command = [
        nvcc,
        "-O3",
        "-std=c++17",
        "-arch=sm_103a",
        str(source),
        "-o",
        str(output),
    ]
    try:
        _run_command(command)
    except HarnessError as exc:
        raise HarnessError(
            f"failed to compile the SM103a probe with {nvcc}; "
            "CUDA 13.1 or newer is recommended. "
            f"Compiler detail: {exc}"
        ) from exc
    return {"nvcc": nvcc, "version": version, "command": command}


def run_native_probe(
    executable: Path,
    device: int,
    bytes_mib: int,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    """Run the CUDA probe and parse its JSON output."""
    command = [
        str(executable),
        "--device",
        str(device),
        "--bytes-mib",
        str(bytes_mib),
        "--warmup",
        str(warmup),
        "--iterations",
        str(iterations),
    ]
    raw_report = _run_command(command)
    try:
        report = json.loads(raw_report)
    except json.JSONDecodeError as exc:
        raise HarnessError(
            f"native probe returned invalid JSON: {raw_report[:500]!r}"
        ) from exc
    validate_native_report(report)
    enrich_native_report(report)
    return report


def addressing_assessment(native_report: dict[str, Any]) -> dict[str, Any]:
    """Summarize the CUDA properties relevant to coherent host access."""
    attrs = native_report["device"]["attributes"]
    registered = native_report["registered_host_allocation"]
    required = (
        "unified_addressing",
        "can_map_host_memory",
        "can_use_host_pointer_for_registered_memory",
        "pageable_memory_access",
        "pageable_memory_access_uses_host_page_tables",
    )
    return {
        "ats_related_attributes_available": all(key in attrs for key in required),
        "direct_registered_host_access_supported": all(
            bool(attrs.get(key)) for key in required[:3]
        )
        and bool(registered.get("device_pointer_available")),
        "pageable_host_page_table_access_supported": bool(
            attrs.get("pageable_memory_access")
            and attrs.get("pageable_memory_access_uses_host_page_tables")
        ),
        "registered_host_and_device_virtual_addresses_match": registered.get(
            "host_and_device_pointer_match"
        ),
        "note": (
            "CUDA does not expose a single 'ATS enabled' flag. The probe records "
            "the unified-addressing, host-mapping, and host-page-table "
            "capabilities that make the Station's coherent access path usable."
        ),
    }


def _render_report(report: dict[str, Any], pretty: bool) -> str:
    return json.dumps(
        report,
        indent=2 if pretty else None,
        sort_keys=pretty,
    )


def _write_or_print_report(
    report: dict[str, Any],
    output: Path | None,
    pretty: bool,
) -> None:
    rendered = _render_report(report, pretty)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(f"{rendered}\n")
    print(rendered)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        help="Also write the complete JSON report to this path",
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--bytes-mib",
        type=int,
        default=1024,
        help="Bytes per transfer/read in MiB (default: 1024)",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--nvcc", help="Path to the CUDA compiler")
    parser.add_argument(
        "--probe-binary",
        type=Path,
        help="Use a precompiled station_memory_probe instead of invoking nvcc",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON instead of emitting compact JSON",
    )
    args = parser.parse_args(argv)
    if args.device < 0:
        parser.error("--device must be non-negative")
    if args.bytes_mib <= 0:
        parser.error("--bytes-mib must be greater than zero")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.iterations <= 0:
        parser.error("--iterations must be greater than zero")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[1]
    source = (
        repo_root / "benchmarks" / "station_weight_tiering" / "station_memory_probe.cu"
    )
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "error",
        "host": collect_host_inventory(),
        "repository": _git_inventory(repo_root),
    }

    try:
        report["nvidia"] = collect_nvidia_inventory()
        if args.probe_binary is not None:
            if not args.probe_binary.is_file():
                raise HarnessError(
                    f"--probe-binary does not exist: {args.probe_binary}"
                )
            native_report = run_native_probe(
                args.probe_binary,
                args.device,
                args.bytes_mib,
                args.warmup,
                args.iterations,
            )
            report["build"] = {"precompiled_probe": str(args.probe_binary)}
        else:
            nvcc = _resolve_nvcc(args.nvcc)
            with tempfile.TemporaryDirectory(
                prefix="vllm-station-memory-"
            ) as temporary_directory:
                probe = Path(temporary_directory) / "station_memory_probe"
                report["build"] = build_native_probe(nvcc, source, probe)
                native_report = run_native_probe(
                    probe,
                    args.device,
                    args.bytes_mib,
                    args.warmup,
                    args.iterations,
                )
        report["native"] = native_report
        report["addressing_assessment"] = addressing_assessment(native_report)
        report["status"] = "ok"
    except (HarnessError, KeyError, TypeError, ValueError) as exc:
        report["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        _write_or_print_report(report, args.output, args.pretty)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    _write_or_print_report(report, args.output, args.pretty)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
