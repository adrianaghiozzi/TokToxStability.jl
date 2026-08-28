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

`setup_venv.sh` prints the exact path to use. See `requirements.txt` for why this can't
just be a `CondaPkg.toml` -- the OFT build this package needs is a pre-release only
published to TestPyPI, and neither `pip`'s standard flags nor CondaPkg's pixi backend
can express that combination through `CondaPkg.toml` alone.
