"""Reproducible annual native-kernel timings and numerical regression checks.

Run with the same interpreter and dependencies before and after a change. Set
PYTHONPATH to a saved package to measure a baseline without replacing the install.
"""

import argparse
import hashlib
import inspect
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from epsm_rc._core_rs import run_coupled_domains_hourly, run_single_domain_hourly

from epsm_rc import Building5R1C
from epsm_rc.heating import DirectElectric


def domain_arguments(model, count):
    """Dynamic hourly forcing, including disabled heating and active cooling."""
    hours = np.arange(count, dtype=float)
    model._prepare_deterministic_hourly()
    values = (
        model,
        np.ascontiguousarray(5 + 20 * np.sin(hours / 100)),
        np.full(count, 400.0),
        np.full(count, 200.0),
        np.full(count, 20.0),
        np.full(count, 26.0),
        np.full(count, 2.0),
        np.full(count, 0.3),
        np.zeros(count),
        np.full(count, 0.5),
        np.maximum(0.0, 2000 * np.sin(hours / 12)),
        np.zeros(count),
        None,
        (1.0, 0.01, 0.02, 0.0),
        False,
        np.where(hours % 24 < 6, 0.0, np.inf),
        np.full(count, -1000.0),
    )
    # Prepared platform inputs use immutable byte storage. Include that path in
    # both baseline and candidate measurements, independently of package version.
    return tuple(
        np.frombuffer(value.tobytes(), dtype=value.dtype)
        if isinstance(value, np.ndarray)
        else value
        for value in values
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--reference-directory", type=Path)
    parser.add_argument("--repeats", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=20)
    args = parser.parse_args()
    if args.repeats < 3 or args.batch_size < 1:
        parser.error("Use at least three repeats and a positive batch size")
    args.directory.mkdir(parents=True, exist_ok=False)
    native_module = sys.modules[run_single_domain_hourly.__module__]
    (args.directory / "environment.json").write_text(
        json.dumps(
            {
                "python": sys.version,
                "numpy": np.__version__,
                "python_source": inspect.getfile(Building5R1C),
                "native_extension": native_module.__file__,
                "native_sha256": hashlib.sha256(
                    Path(native_module.__file__).read_bytes()
                ).hexdigest(),
            },
            indent=2,
        )
        + "\n"
    )

    count = 8760
    outdoor = np.full(count, -5.0)
    standalone = Building5R1C()
    heater = DirectElectric(building_model=Building5R1C())
    single = domain_arguments(Building5R1C(), count)
    first = domain_arguments(Building5R1C(), count)
    second = domain_arguments(Building5R1C(), count)
    coupled = (first[0], second[0], *first[1:], *second[1:], 50.0)

    def run_domain():
        single[0].reset()
        return run_single_domain_hourly(*single)

    def run_coupled():
        coupled[0].reset()
        coupled[1].reset()
        return run_coupled_domains_hourly(*coupled)

    cases = {
        "standalone": lambda: standalone.simulate(outdoor),
        "heating": lambda: heater.heat_demand_w(outdoor, minutes_per_slot=60),
        "single_domain": run_domain,
        "coupled_domains": run_coupled,
    }
    report = {}
    for name, run in cases.items():
        result = run()
        channels = (
            result
            if isinstance(result, dict)
            else {
                str(i): array
                for i, array in enumerate(result if isinstance(result, tuple) else (result,))
            }
        )
        np.savez(args.directory / f"{name}.npz", **channels)
        if args.reference_directory:
            with np.load(args.reference_directory / f"{name}.npz") as reference:
                assert set(channels) == set(reference.files)
                for channel, values in channels.items():
                    np.testing.assert_allclose(values, reference[channel], rtol=1e-10, atol=1e-9)
        samples = []
        for _ in range(args.repeats):
            started = time.perf_counter_ns()
            for _ in range(args.batch_size):
                run()
            samples.append((time.perf_counter_ns() - started) / args.batch_size / 1e6)
        report[name] = {"median_ms": statistics.median(samples), "samples_ms": samples}
        if args.reference_directory:
            old = json.loads((args.reference_directory / "report.json").read_text())[name]
            report[name]["relative_time"] = report[name]["median_ms"] / old["median_ms"]
        print(name, report[name]["median_ms"], report[name].get("relative_time"), flush=True)
    (args.directory / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
