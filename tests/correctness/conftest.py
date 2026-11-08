import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def pytest_addoption(parser):
    parser.addoption("--results", action="store", default=None,
                     help="results dir produced by `python -m bench run`")
    parser.addoption("--update-golden", action="store_true",
                     help="(re)write golden/<model>.json from this run's cache_off outputs")


@pytest.fixture(scope="session")
def results_dir(request) -> Path:
    r = request.config.getoption("--results")
    if not r:
        pytest.skip("pass --results <dir> to run correctness checks")
    p = Path(r)
    if not p.is_absolute():
        p = ROOT / p
    return p


@pytest.fixture(scope="session")
def cfg(results_dir) -> dict:
    return json.loads((results_dir / "config.json").read_text())


@pytest.fixture(scope="session")
def tol(cfg) -> dict:
    return cfg["correctness"]["thresholds"]


@pytest.fixture(scope="session")
def runs(results_dir) -> dict:
    out = {}
    for mode in ("cache_on", "cache_off"):
        f = results_dir / f"correctness_{mode}.json"
        if f.exists():
            out[mode] = json.loads(f.read_text())
    if not out:
        pytest.skip(f"no correctness_*.json in {results_dir}")
    return out


@pytest.fixture(scope="session")
def golden_path(cfg) -> Path:
    model = cfg["server"].get("served_model_name") or cfg["server"]["model"]
    return ROOT / "golden" / (model.replace("/", "__") + ".json")
