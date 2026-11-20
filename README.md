# TokToxStability

[![Build Status](https://github.com/adrianaghiozzi/TokToxStability.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/adrianaghiozzi/TokToxStability.jl/actions/workflows/CI.yml?query=branch%3Amain)

## Setup

No local OFT source build and no local GPEC checkout are required -- GPEC is pulled
automatically as a normal Julia dependency (see `Project.toml`'s `[sources]`), and OFT
comes from a Python virtualenv you set up once:

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate()'
./setup_venv.sh
```

Then, before `using TokToxStability` (e.g. in your notebook's first cell), point
PythonCall at that venv instead of letting it manage its own Python:

```julia
ENV["JULIA_CONDAPKG_BACKEND"] = "Null"
ENV["JULIA_PYTHONCALL_EXE"] = "/absolute/path/to/TokToxStability/.venv/bin/python"
using TokToxStability
```

`setup_venv.sh` prints the exact path to use. Python versions are pinned in
`requirements.txt`: stable OpenFUSIONToolkit from PyPI, and TORAX held at 1.4.x because
OFT's TokaMaker+TORAX coupling does not yet support TORAX 2.x.

GPEC v2 requires Julia 1.11 or newer.

GPEC is not in the General registry, so `Project.toml`'s `[sources]` installs it from GitHub,
following `main`, GPEC's release branch. Day-to-day GPEC development lands on `develop`, and
`main` only moves when a release is cut. The `[compat]` bound (`GeneralizedPerturbedEquilibrium = "2"`)
keeps it on 2.x, and Pkg enforces that bound for URL sources too. `Pkg.update()` picks up new GPEC
releases. Once GPEC is registered, delete the `[sources]` entry.

## Running the tests

The tests don't call into Python, but PythonCall still has to load, so point it at the venv
(or any Python 3.12+) the same way:

```sh
JULIA_CONDAPKG_BACKEND=Null JULIA_PYTHONCALL_EXE="$PWD/.venv/bin/python" \
    julia --project=. -e 'using Pkg; Pkg.test()'
```
