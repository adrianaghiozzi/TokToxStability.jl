using PythonCall
using DelimitedFiles
using Printf

default_tag(fraction::Real) = "f" * replace(@sprintf("%.4f", fraction), "." => "p")

"""
    coarse_scan(mygs, fractions, equilibrium, profiles, gpec_settings::GPECSettings;
                working_directory=nothing, verbose=false)

Runs the TokaMaker equilibrium solve for each fraction, then the GPEC stability
solve for each resulting equilibrium, sequentially.

`gpec_settings` is a `GPECSettings` you build and configure yourself (toroidal
mode number, wall, kinetic settings) -- see its docstring. It's read, never
mutated -- see `evaluate_factor`.

`working_directory`: `nothing` (default) runs each fraction in its own
temporary directory, deleted right after that fraction's GPEC call returns --
see `evaluate_factor`'s docstring for what that costs you (no `plot_equilibrium`
afterward). Pass a real path to persist every fraction's gEQDSK/GPEC outputs
under it instead, as before.

Both stages must stay sequential:
- TokaMaker solves mutate the single `mygs` instance in place, so they can't be
  reordered or overlapped.
- GPEC itself uses `Threads.@threads :static` extensively in its own hot loops
  (flux-surface reconstruction, ballooning/Mercier scans, vacuum kernels, ...),
  completely independent of the `parallel_threads`/`use_parallel` toml knobs
  (those only gate the inner Delta' BVP path). Julia's `:static` schedule
  cannot be nested or run concurrently across tasks/threads, so calling
  `run_GPEC` from more than one task at a time throws
  `"@threads :static cannot be used concurrently or nested"`. A single GPEC
  run already uses every thread the process has (`-t auto`/`-t N`); that's the
  available parallelism here. Running multiple fractions' GPEC solves at once
  would require separate OS processes (e.g. `Distributed.jl` workers), not
  Julia threads.
"""
function coarse_scan(mygs::Py, fractions::Vector{<:Real}, equilibrium::Equilibrium, profiles::Profiles,
                      gpec_settings::GPECSettings; working_directory::Union{String,Nothing}=nothing, verbose::Bool = false)
    coarse_scan_results = []
    for fraction in fractions
        try
            result = evaluate_factor(mygs, deepcopy(profiles), deepcopy(equilibrium), fraction, gpec_settings; working_directory, verbose)
            push!(coarse_scan_results, result)
        catch e
            println(e)
            continue
        end
    end
    return coarse_scan_results
end

"""
    evaluate_factor(mygs, profiles, equilibrium, fraction, gpec_settings::GPECSettings;
                     working_directory=nothing, tag=nothing, verbose=false)

Solves TokaMaker at `fraction`, writes its gEQDSK, then runs GPEC on it using
the settings in `gpec_settings` (toroidal mode number, wall, kinetic) -- read
only, never mutated (it holds no per-fraction state; see `GPECSettings`/
`run_GPEC`).

GPEC itself only knows how to read its equilibrium off disk (it's invoked by
pointing it at a directory, not through an in-memory API), so *some* gEQDSK
file always has to exist temporarily for it to read -- that part isn't
optional. Whether that file (and GPEC's own output artifacts alongside it --
F/G/K matrices, etc., since `mat_flag=true`) survives afterward is:

- `working_directory = nothing` (default): runs in a fresh `mktempdir()`,
  deleted in a `finally` block right after GPEC returns (even on error).
  `geqdsk_path`/`run_dir` are `nothing` in the returned row, since the files
  they'd point to no longer exist -- notably, `plots.jl`'s `plot_equilibrium`
  won't work for a result row from this mode, since it has nothing left to
  read back.
- `working_directory = "some/path"`: unchanged from before -- writes to
  `some/path/<tag>/`, kept after the call returns.
"""
function evaluate_factor(mygs::Py, profiles::Profiles, equilibrium::Equilibrium, fraction::Real, gpec_settings::GPECSettings;
                          working_directory::Union{String,Nothing}=nothing, tag::Union{String,Nothing}=nothing, verbose::Bool=false)
    tag = something(tag, default_tag(fraction))
    ephemeral = working_directory === nothing
    base_dir = ephemeral ? mktempdir() : working_directory
    run_directory = joinpath(base_dir, tag)

    try
        nl_its, stats, beta_n, beta_t, l_i = quietly(; verbose) do
            solve_at_pressure_factor(mygs, profiles, equilibrium, fraction)
        end

        # Full q(psi) profile at this fraction's converged equilibrium -- kept
        # alongside the scalar q0/qmax (from GPEC, below) for per-iteration
        # plotting (see plots.jl). `*_` absorbs any extra return values (e.g.
        # flux-surface-averaged geometry terms) so this doesn't break if
        # `get_q`'s return shape grows in a future TokaMaker version.
        psi_q, q_profile, _ravgs, _rest... = pyconvert(Tuple, mygs.get_q(npsi=length(equilibrium.Psi), psi_pad=0.001))

        geqdsk_path = write_geqdsk_for_gpec(mygs, run_directory, tag)

        gpec_raw = quietly(; verbose) do
            run_GPEC(gpec_settings, basename(geqdsk_path), run_directory)
        end
        gp = read_output(gpec_raw)

        return (fraction=fraction, tag=tag, beta_n=beta_n, beta_t=beta_t, l_i=l_i,
                p_axis=profiles.pressure[1], dW_total=gp.dW_total, dW_plasma=gp.dW_plasma,
                dW_vacuum=gp.dW_vacuum, q0=gp.q0, qmax=gp.qmax, msing=gp.msing,
                run_dir=(ephemeral ? nothing : run_directory), geqdsk_path=(ephemeral ? nothing : geqdsk_path),
                # Per-iteration profile/q snapshots for plots.jl. `psi` is the
                # grid `profiles`/`equilibrium` are already sampled on (post
                # rescale_profile!), so these all line up index-for-index.
                psi=copy(equilibrium.Psi), Te=copy(profiles.Te), Ti=copy(profiles.Ti),
                ne=copy(profiles.ne), pressure=copy(profiles.pressure),
                psi_q=pyconvert(Vector{Float64}, psi_q), q_profile=pyconvert(Vector{Float64}, q_profile))
    finally
        ephemeral && rm(base_dir; recursive=true, force=true)
    end
end

"""
    coarse_scan_until_unstable(mygs, start_fraction, step, equilibrium, profiles,
                                gpec_settings::GPECSettings;
                                working_directory=nothing, extra_past_crossing=1,
                                max_fractions=8, verbose=false)

`gpec_settings` is a `GPECSettings` you build and configure yourself (toroidal mode number,
wall, kinetic settings) -- see its docstring. It's read, never mutated -- see
`evaluate_factor`.

`working_directory`: `nothing` (default) runs each fraction in its own
temporary directory, deleted right after that fraction's GPEC call returns --
see `evaluate_factor`'s docstring for what that costs you (no `plot_equilibrium`
afterward). Pass a real path to persist every fraction's gEQDSK/GPEC outputs
under it instead, as before.

Runs `evaluate_factor` (TokaMaker + GPEC) sequentially in ascending pressure
fraction, starting at `start_fraction` and incrementing by `step` each time,
stopping once GPEC's `dW_total` actually crosses from stable (>=0) to
unstable (<0) -- the real ideal-MHD marginal-stability criterion. This
replaces an earlier attempt to locate the crossing cheaply via TokaMaker's
canonical `beta_N = 4*l_i` limit before ever calling GPEC: that limit is
only a rough empirical operational guideline and was found (empirically, on
a real case) to sit well above where GPEC's `dW_total` actually crosses
zero for a given equilibrium/mode/wall, so using it to pick which fractions
get a GPEC solve pointed the search at the wrong region of fraction space.
This version checks the real quantity at every step instead, at the cost of
spending GPEC time on every fraction visited (not just a pre-guessed few) --
still cheaper than a long fixed sweep, since it stops as soon as the
crossing is bracketed rather than running to a preset end fraction.

Continues for `extra_past_crossing` fractions past the first negative
`dW_total` so the returned scan brackets the crossing with a bit of slope
context (matching the shape of a typical hand-picked reference scan).
`max_fractions` is a hard cap on the number of fractions attempted (whether
or not each one converges), in case `dW_total` never goes negative (e.g.
`start_fraction`/`step` too small/large for this equilibrium) -- a warning
is issued rather than looping forever.
"""
function coarse_scan_until_unstable(mygs::Py, start_fraction::Real, step::Real, equilibrium::Equilibrium, profiles::Profiles,
                                     gpec_settings::GPECSettings; working_directory::Union{String,Nothing}=nothing,
                                     extra_past_crossing::Int=1, max_fractions::Int=8, verbose::Bool=false)
    step > 0 || error("coarse_scan_until_unstable: step must be positive (ascending scan)")

    results = []
    crossed_at = nothing
    fraction = start_fraction
    attempts = 0
    while attempts < max_fractions
        attempts += 1
        result = try
            evaluate_factor(mygs, deepcopy(profiles), deepcopy(equilibrium), fraction, gpec_settings; working_directory, verbose)
        catch e
            println(e)
            nothing
        end

        if result !== nothing
            push!(results, result)
            if result.dW_total < 0
                crossed_at = something(crossed_at, length(results))
                length(results) - crossed_at >= extra_past_crossing && break
            end
        end

        fraction += step
    end

    crossed_at === nothing && @warn "coarse_scan_until_unstable: dW_total never went negative within " *
                                     "max_fractions=$max_fractions (last fraction tried = $(fraction - step)) " *
                                     "-- results may not bracket the true beta limit"

    return results
end

"""
    _interp_zero_crossing(xs, ys) -> (crossing, lo, hi, bracketed::Bool)

Generic zero-crossing helper shared by `estimate_beta_limit` (over
`beta_n`/`dW_total`) and `plots.jl`'s `plot_beta_scan_summary` (over
`fraction`/`dW_total`, to place the same estimated crossing on a
fraction-space x-axis). See `estimate_beta_limit`'s docstring for the
bracketed-vs-extrapolated behavior this implements; this just operates on
plain vectors instead of `coarse_scan` result rows.
"""
function _interp_zero_crossing(xs::AbstractVector{<:Real}, ys::AbstractVector{<:Real})
    order = sortperm(xs)
    xs, ys = xs[order], ys[order]

    for i in 1:length(xs)-1
        if ys[i] >= 0 && ys[i+1] < 0
            t = ys[i] / (ys[i] - ys[i+1])
            crossing = xs[i] + t * (xs[i+1] - xs[i])
            return (crossing=crossing, lo=xs[i], hi=xs[i+1], bracketed=true)
        end
    end

    n = length(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    denom = sum((xs .- mean_x) .^ 2)
    denom == 0 && error("_interp_zero_crossing: all x values are identical -- cannot fit a line.")
    slope = sum((xs .- mean_x) .* (ys .- mean_y)) / denom
    slope == 0 && error("_interp_zero_crossing: y values are flat (zero slope) -- cannot extrapolate a crossing.")
    intercept = mean_y - slope * mean_x
    crossing = -intercept / slope

    return (crossing=crossing, lo=first(xs), hi=last(xs), bracketed=false)
end

"""
    estimate_beta_limit(results; verbose=true) -> (beta_n_crossing, beta_n_lo, beta_n_hi, bracketed::Bool)

Estimates the ideal-MHD marginal-stability beta_N limit (where `dW_total`
crosses zero) from `results` -- the vector of NamedTuples returned by
`coarse_scan`/`coarse_scan_until_unstable`, each expected to have `beta_n`
and `dW_total` fields.

If the scan actually BRACKETS a stable (`dW_total >= 0`) -> unstable
(`dW_total < 0`) transition between two adjacent points (sorted by
`beta_n`), `beta_n_crossing` is linearly interpolated strictly between
those two points, and `beta_n_lo`/`beta_n_hi` are their exact, calculated
beta_N values (`bracketed = true`) -- i.e. the real bounds the scan
achieved, not an extrapolation.

If no such transition is found (e.g. `coarse_scan_until_unstable` hit
`max_fractions` while still stable everywhere, or the scan started already
unstable), this issues a `@warn` and instead fits a straight line through
ALL scanned (beta_n, dW_total) points -- ordinary least squares for 3+
points, the exact two-point line for exactly 2 -- and extrapolates it to
`dW_total = 0`. `beta_n_lo`/`beta_n_hi` in this case report the min/max
beta_N actually scanned (NOT the crossing bounds), so the gap between them
and `beta_n_crossing` shows how far the extrapolation reaches beyond real
data. `bracketed = false` flags that `beta_n_crossing` here is a rough
extrapolated guess, not a calculated bound.
"""
function estimate_beta_limit(results::AbstractVector; verbose::Bool=true)
    length(results) >= 2 || error("estimate_beta_limit needs at least 2 scanned points, got $(length(results))")

    beta_n = [r.beta_n for r in results]
    dW = [r.dW_total for r in results]
    res = _interp_zero_crossing(beta_n, dW)

    if res.bracketed
        verbose && println(
            "estimate_beta_limit: dW_total crossing bracketed between the scanned points " *
            "beta_N=$(round(res.lo; digits=4)) and beta_N=$(round(res.hi; digits=4)) -- " *
            "interpolated crossing at beta_N=$(round(res.crossing; digits=4))."
        )
        return (beta_n_crossing=res.crossing, beta_n_lo=res.lo, beta_n_hi=res.hi, bracketed=true)
    end

    @warn "estimate_beta_limit: no stable->unstable dW_total transition found across the " *
          "$(length(results)) scanned points (beta_n in [$(res.lo), $(res.hi)]) -- " *
          "extrapolating a line through the scanned points instead of interpolating between " *
          "bracketing points. Treat the resulting beta_N as a rough guess, not a calculated bound; " *
          "scan further to actually bracket the crossing."

    verbose && println(
        "estimate_beta_limit: extrapolated (unbracketed) crossing at beta_N=$(round(res.crossing; digits=4)), " *
        "from a line fit through beta_N in [$(round(res.lo; digits=4)), $(round(res.hi; digits=4))]."
    )

    return (beta_n_crossing=res.crossing, beta_n_lo=res.lo, beta_n_hi=res.hi, bracketed=false)
end

function tokamaker_run(mygs::Py, profiles::Profiles, equilibrium::Equilibrium; n_settle::Int = 3)
    nl_its = nothing

    for _ in 1:n_settle
        psi_bounds = mygs.psi_bounds
        psi_b0 = pyconvert(Float64, psi_bounds[0])
        psi_b1 = pyconvert(Float64, psi_bounds[1])

        profiles.pprime = np_gradient(profiles.pressure) ./
                           (np_gradient(equilibrium.Psi) .* (psi_b1 - psi_b0))

        mygs.set_targets(Ip=equilibrium.Ip_target, pax=profiles.pressure[1])

        mygs.set_profiles(
            pp_prof  = Dict("type" => "linterp",      "y" => profiles.pprime,  "x" => equilibrium.Psi),
            ffp_prof = Dict("type" => "jphi-linterp", "y" => equilibrium.Jphi,    "x" => equilibrium.Psi),
        )

        try
            _eq, its = mygs.solve(return_its=true) # this is the TokaMaker solve step
            nl_its = pyconvert(Int, its)
        catch exc
            error("TokaMaker did not converge: $exc")
        end
    end

    stats = mygs.get_stats()
    return (
        nl_its = nl_its,
        stats  = stats,
        beta_n = pyconvert(Float64, stats.get("beta_n")),
        beta_t = pyconvert(Float64, stats.get("beta_tor")),
        l_i    = pyconvert(Float64, stats.get("l_i")),
    )
end

function solve_at_pressure_factor(mygs::Py, profiles::Profiles, equilibrium::Equilibrium, fraction::Real)
    rescale_profile!(profiles, fraction)
    nl_its, stats, beta_n, beta_t, l_i = tokamaker_run(mygs, profiles, equilibrium)
    return nl_its, stats, beta_n, beta_t, l_i
end

function write_geqdsk_for_gpec(mygs::Py, run_directory::String, tag, nr::Int=257, nz::Int=257)
    mkpath(run_directory)
    out = joinpath(run_directory, "tokamaker_$(tag).geqdsk")

    mygs.save_eqdsk(out; nr=nr, nz=nz, truncate_eq=false, lcfs_pad=0.001)
    return out
end
