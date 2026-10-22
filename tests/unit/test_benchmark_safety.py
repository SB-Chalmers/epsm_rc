"""Benchmark output directories must never replace existing caller-owned data."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks import run as benchmark


def benchmark_arguments(directory):
    return SimpleNamespace(
        directory=directory,
        reference_directory=None,
        reference_python=None,
        repeats=2,
        policy=1,
        no_output=True,
        warm=False,
        batch=False,
    )


@pytest.mark.parametrize("existing", ["directory", "symlink", "file"])
def test_existing_output_is_refused_before_weather_or_calculations(tmp_path, monkeypatch, existing):
    output = tmp_path / "benchmark-output"
    retained = tmp_path / "retained"
    retained.mkdir()
    sentinel = retained / "result.txt"
    sentinel.write_bytes(b"existing calculation result\x00")
    if existing == "directory":
        output.mkdir()
        sentinel = output / "result.txt"
        sentinel.write_bytes(b"existing calculation result\x00")
    elif existing == "symlink":
        output.symlink_to(retained, target_is_directory=True)
    else:
        output.write_bytes(b"existing benchmark report")
    output_was_link = output.is_symlink()
    original = sentinel.read_bytes()

    def forbidden(*args, **kwargs):
        pytest.fail("Existing benchmark output was accepted before weather preparation")

    monkeypatch.setattr(benchmark, "controlled_weather", forbidden)
    monkeypatch.setattr(benchmark, "simulate_building", forbidden)
    with pytest.raises(FileExistsError):
        benchmark.run(benchmark_arguments(output), None)
    assert output.exists()
    assert output.is_symlink() == output_was_link
    assert sentinel.read_bytes() == original
    if existing == "file":
        assert output.read_bytes() == b"existing benchmark report"


def test_dangling_output_symlink_is_refused(tmp_path, monkeypatch):
    output = tmp_path / "benchmark-output"
    target = tmp_path / "absent-target"
    output.symlink_to(target, target_is_directory=True)
    monkeypatch.setattr(
        benchmark,
        "controlled_weather",
        lambda *args: pytest.fail("Dangling output link reached weather preparation"),
    )
    with pytest.raises(FileExistsError):
        benchmark.run(benchmark_arguments(output), None)
    assert output.is_symlink()
    assert output.readlink() == Path(target)
    assert not target.exists()
