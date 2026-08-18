using PythonCall
using DelimitedFiles
using Printf

default_tag(fraction::Real) = "f" * replace(@sprintf("%.4f", fraction), "." => "p")

"""
    coarse_scan(mygs, fractions, equilibrium, profiles, toroidal_mode_number, wall, kinetic, working_directory; verbose=false)

Runs the TokaMaker equilibrium solve for each fraction, then the GPEC stability
solve for each resulting equilibrium, sequentially.

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
function coarse_scan(mygs::Py, fractions::Vector{<:Real}, equilibrium::Equilibrium, profiles::Profiles, toroidal_mode_number, wall, kinetic, working_directory; verbose::Bool = false)
    coarse_scan_results = []
    for fraction in fractions
        try
            result = evaluate_factor(mygs, deepcopy(profiles), deepcopy(equilibrium), fraction, toroidal_mode_number, wall, kinetic, working_directory; verbose)
            push!(coarse_scan_results, result)
        catch e
            println(e)
            continue
        end
    end
    return coarse_scan_results
end

function evaluate_factor(mygs::Py, profiles::Profiles, equilibrium::Equilibrium, fraction::Real, toroidal_mode_number::Int,
                          wall::Dict{String,<:Any}, kinetic::Dict{String,<:Any},
                          working_directory::String; tag::Union{String,Nothing}=nothing,
                          parallel_threads::Int=2, verbose::Bool=false)
    tag = something(tag, default_tag(fraction))
    run_directory = joinpath(working_directory, tag)

    nl_its, stats, beta_n, beta_t, l_i = quietly(; verbose) do
        solve_at_pressure_factor(mygs, profiles, equilibrium, fraction)
    end

    geqdsk_path = write_geqdsk_for_gpec(mygs, run_directory, tag)
    equilibrium_filename = basename(geqdsk_path)

    gi = GPECInput(equilibrium_filename=equilibrium_filename, toroidal_mode_number=toroidal_mode_number, wall=wall,
                    kinetic=kinetic, parallel_threads=parallel_threads)
    gpec_raw = quietly(; verbose) do
        run_GPEC(gi, run_directory)
    end
    gp = read_output(gpec_raw)

    return (fraction=fraction, tag=tag, beta_n=beta_n, beta_t=beta_t, l_i=l_i,
            p_axis=profiles.pressure[1], dW_total=gp.dW_total, dW_plasma=gp.dW_plasma,
            dW_vacuum=gp.dW_vacuum, q0=gp.q0, qmax=gp.qmax, msing=gp.msing,
            run_dir=run_directory)
end

"""
    coarse_scan_until_unstable(mygs, start_fraction, step, equilibrium, profiles,
                                toroidal_mode_number, wall, kinetic, working_directory;
                                extra_past_crossing=1, max_fractions=20, verbose=false)

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
`max_fractions` is a hard cap in case `dW_total` never goes negative (e.g.
`start_fraction`/`step` too small/large for this equilibrium) -- a warning
is issued rather than looping forever.
"""
function coarse_scan_until_unstable(mygs::Py, start_fraction::Real, step::Real, equilibrium::Equilibrium, profiles::Profiles,
                                     toroidal_mode_number, wall::Dict{String,<:Any}, kinetic::Dict{String,<:Any},
                                     working_directory; extra_past_crossing::Int=1, max_fractions::Int=20, verbose::Bool=false)
    step > 0 || error("coarse_scan_until_unstable: step must be positive (ascending scan)")

    results = []
    crossed_at = nothing
    fraction = start_fraction
    while length(results) < max_fractions
        result = try
            evaluate_factor(mygs, deepcopy(profiles), deepcopy(equilibrium), fraction, toroidal_mode_number, wall, kinetic, working_directory; verbose)
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
