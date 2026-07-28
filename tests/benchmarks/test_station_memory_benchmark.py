# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from benchmarks.benchmark_station_memory import (
    HarnessError,
    build_overlap_assessment,
    parse_meminfo,
    parse_numa_meminfo,
    parse_nvidia_smi_csv,
    summarize_samples,
    validate_native_report,
)


def test_meminfo_parsers_preserve_bytes_and_unitless_counts():
    proc = parse_meminfo(
        "MemTotal:       1000 kB\nMemAvailable:    500 kB\nHugePages_Total:   12\n"
    )
    numa = parse_numa_meminfo(
        "Node 0 MemTotal:       750 kB\nNode 0 MemFree:        250 kB\n"
    )

    assert proc == {
        "MemTotal": 1000 * 1024,
        "MemAvailable": 500 * 1024,
        "HugePages_Total": 12,
    }
    assert numa == {
        "MemTotal": 750 * 1024,
        "MemFree": 250 * 1024,
    }


def test_nvidia_smi_parser_respects_quoted_names():
    fields = ("index", "name", "memory.total")
    rows = parse_nvidia_smi_csv('0, "NVIDIA GB300, Engineering", 258048', fields)

    assert rows == [
        {
            "index": "0",
            "name": "NVIDIA GB300, Engineering",
            "memory.total": "258048",
        }
    ]


def test_timing_summary_uses_nearest_rank_and_decimal_bandwidth():
    summary = summarize_samples(
        samples_ms=[4.0, 1.0, 3.0, 2.0],
        bytes_per_iteration=1_000_000_000,
        memory_traffic_factor=2,
    )

    assert summary["mean_ms"] == pytest.approx(2.5)
    assert summary["p50_ms"] == 2.0
    assert summary["p95_ms"] == 4.0
    assert summary["payload_gb_s"] == pytest.approx(400.0)
    assert summary["memory_traffic_gb_s"] == pytest.approx(800.0)


def test_overlap_assessment_reports_perfect_overlap():
    assessment = build_overlap_assessment(
        h2d_mean_ms=4.0,
        hbm_read_mean_ms=6.0,
        observed_mean_ms=6.0,
    )

    assert assessment["speedup_vs_serial"] == pytest.approx(10 / 6)
    assert assessment["fraction_of_ideal_overlap"] == pytest.approx(1.0)


def test_native_report_rejects_non_sm103_results():
    with pytest.raises(HarnessError, match="requires SM103"):
        validate_native_report(
            {
                "device": {
                    "compute_capability": {
                        "major": 12,
                        "minor": 1,
                    }
                }
            }
        )
