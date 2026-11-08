# RoboGuardian

RoboGuardian verifies physical effects, maintains trusted memory, and supports recovery from execution faults in embodied agent harness frameworks.

Simulator integrations are available for AdaClearGrasp, LIBERO, and LaViRA / Habitat.

[Project website](index.html) · [Paper](assets/papers/RoboGuardian.pdf) · [Demo video](assets/videos/demo.mp4)

## Installation

The core package requires Python 3.10+ and [uv](https://docs.astral.sh/uv/):

```bash
uv sync --locked
```

The core package has no third-party runtime dependencies. Each simulator integration requires its own Python environment, models, datasets, and assets; these are not included in the core lockfile. The LaViRA worker requires Python 3.9.

To initialize the AdaClearGrasp and LaViRA submodules:

```bash
git submodule update --init third_party/AdaClearGrasp third_party/LaViRA
python scripts/setup_adacleargrasp.py
```

## Usage

Run the following from the repository root to see the available options:

```bash
uv run python -m integrations.adacleargrasp --help
uv run python -m integrations.libero --help
uv run python -m integrations.habitat --help
```

Prepare the appropriate simulator environment, models, and datasets before running an integration.

## Tests

```bash
uv run python -m unittest discover -s tests -v
```

The tests cover core behavior and integration contracts; running simulations requires the corresponding external environments.
