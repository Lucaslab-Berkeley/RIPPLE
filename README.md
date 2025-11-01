# RIPPLE

[![License](https://img.shields.io/pypi/l/RIPPLE.svg?color=green)](https://github.com/Lucaslab-Berkeley/RIPPLE/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/RIPPLE.svg?color=green)](https://pypi.org/project/RIPPLE)
[![Python Version](https://img.shields.io/pypi/pyversions/RIPPLE.svg?color=green)](https://python.org)
[![CI](https://github.com/Lucaslab-Berkeley/RIPPLE/actions/workflows/ci.yml/badge.svg)](https://github.com/Lucaslab-Berkeley/RIPPLE/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/Lucaslab-Berkeley/RIPPLE/branch/main/graph/badge.svg)](https://codecov.io/gh/Lucaslab-Berkeley/RIPPLE)

Cryo-EM movie frame alignment and polishing. RIPPLE: Realigning image patches and polishing local environment

## Development

The easiest way to get started is to use the [github cli](https://cli.github.com)
and [uv](https://docs.astral.sh/uv/getting-started/installation/):

```sh
gh repo fork Lucaslab-Berkeley/RIPPLE --clone
# or just
# gh repo clone Lucaslab-Berkeley/RIPPLE
cd RIPPLE
uv sync
```

Run tests:

```sh
uv run pytest
```

Lint files:

```sh
uv run pre-commit run --all-files
```
