from pathlib import Path

import numpy as np
import pytest
import torch as th

from stable_baselines3.common.logger import (
    DEBUG,
    ScopedConfigure,
    configure,
    debug,
    dump,
    error,
    info,
    make_output_format,
    read_csv,
    read_json,
    record,
    record_dict,
    record_mean,
    reset,
    set_level,
    warn,
    Video
)
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

KEY_VALUES = {
    "test": 1,
    "b": -3.14,
    "8": 9.9,
    "l": [1, 2],
    "a": np.array([1, 2, 3]),
    "f": np.array(1),
    "g": np.array([[[1]]]),
}

KEY_EXCLUDED = {}
for key in KEY_VALUES.keys():
    KEY_EXCLUDED[key] = None


def test_main(tmp_path):
    """
    tests for the logger module
    """
    info("hi")
    debug("shouldn't appear")
    set_level(DEBUG)
    debug("should appear")
    configure(folder=str(tmp_path))
    record("a", 3)
    record("b", 2.5)
    dump()
    record("b", -2.5)
    record("a", 5.5)
    dump()
    info("^^^ should see a = 5.5")
    record_mean("b", -22.5)
    record_mean("b", -44.4)
    record("a", 5.5)
    dump()
    with ScopedConfigure(None, None):
        info("^^^ should see b = 33.3")

    with ScopedConfigure(str(tmp_path / "test-logger"), ["json"]):
        record("b", -2.5)
        dump()

    reset()
    record("a", "longasslongasslongasslongasslongasslongassvalue")
    dump()
    warn("hey")
    error("oh")
    record_dict({"test": 1})


@pytest.mark.parametrize("_format", ["stdout", "log", "json", "csv", "tensorboard"])
def test_make_output(tmp_path, _format):
    """
    test make output

    :param _format: (str) output format
    """
    if _format == "tensorboard":
        # Skip if no tensorboard installed
        pytest.importorskip("tensorboard")

    writer = make_output_format(_format, tmp_path)
    writer.write(KEY_VALUES, KEY_EXCLUDED)
    if _format == "csv":
        read_csv(tmp_path / "progress.csv")
    elif _format == "json":
        read_json(tmp_path / "progress.json")
    writer.close()


def test_make_output_fail(tmp_path):
    """
    test value error on logger
    """
    with pytest.raises(ValueError):
        make_output_format("dummy_format", tmp_path)


@pytest.mark.parametrize("_format", ["stdout", "log", "json", "csv", "tensorboard"])
def test_report_video(tmp_path, _format):
    """
    test reporting a video to tensorboard, other formats are not supported

    :param _format: (str) output format
    """
    if _format == "tensorboard":
        # Skip if no tensorboard installed
        pytest.importorskip("tensorboard")

    video = Video(frames=th.rand(1, 20, 3, 16, 16), fps=20)
    writer = make_output_format(_format, tmp_path)
    writer.write({"video": video}, KEY_EXCLUDED)
    if _format == "tensorboard":
        assert_has_tb_tag(tmp_path, "video")
    elif _format == "csv":
        assert read_csv(tmp_path / "progress.csv").empty
    elif _format == "json":
        assert read_json(tmp_path / "progress.json").empty
    elif _format == "log":
        assert_is_empty_file(tmp_path / "log.txt")
    writer.close()


def assert_has_tb_tag(log_dir: Path, tag: str):
    acc = EventAccumulator(str(log_dir))
    acc.Reload()
    images_tags = set(acc.Tags()["images"])
    assert tag in images_tags


def assert_is_empty_file(file: Path):
    assert file.read_text() == ""
