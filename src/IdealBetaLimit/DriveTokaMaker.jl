using PythonCall
using DelimitedFiles
using Printf

default_tag(fraction::Real) = "f" * replace(@sprintf("%.4f", fraction), "." => "p")

function coarse_scan(mygs::Py, fractions::Vector{<:Real}, equilibrium::Equilibrium, profiles::Profiles,
                      gpec_settings::GPECSettings; working_directory::Union{String,Nothing}=nothing, verbose::Bool = false)
    psi_baseline = mygs.get_psi(normalized=false)
    coarse_scan_results = []
    for fraction in fractions
        try
            mygs.set_psi(psi_baseline, update_bounds=true)
            result = evaluate_factor(mygs, deepcopy(profiles), deepcopy(equilibrium), fraction, gpec_settings; working_directory, verbose)
            push!(coarse_scan_results, result)
        catch e
            println(e)
            continue
        end
    end
    return coarse_scan_results
end

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

function coarse_scan_until_unstable(mygs::Py, start_fraction::Real, step::Real, equilibrium::Equilibrium, profiles::Profiles,
                                     gpec_settings::GPECSettings; working_directory::Union{String,Nothing}=nothing,
                                     extra_past_crossing::Int=1, max_fractions::Int=8, verbose::Bool=false)
    step > 0 || error("coarse_scan_until_unstable: step must be positive (ascending scan)")

    psi_baseline = mygs.get_psi(normalized=false)
    results = []
    crossed_at = nothing
    fraction = start_fraction
    attempts = 0
    while attempts < max_fractions
        attempts += 1
        mygs.set_psi(psi_baseline, update_bounds=true)
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

        profiles.pprime = numpy_like_gradient(profiles.pressure) ./
                           (numpy_like_gradient(equilibrium.Psi) .* (psi_b1 - psi_b0))

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

function setup_tokamaker_from_file!(::Val{:D3D}, profiles::TokToxStability.Profiles, equilibrium::TokToxStability.Equilibrium, myOFT::Py; oft_root::String)
    kin_nr = length(equilibrium.Psi)
    d3d_case_dir = joinpath(pkgdir(TokToxStability), "cases", "D3D")
    mesh_file = joinpath(d3d_case_dir, "DIIID_mesh.h5")
    geqdsk_file = joinpath(d3d_case_dir, "g147131.02300_DIIID_KEFIT.eqdsk")

    result = @pyexec (
        OFT_ROOT=oft_root, MESH_FILE=mesh_file, GEQDSK_FILE=geqdsk_file,
        psi_prof=equilibrium.Psi, kin_nr=kin_nr, myOFT=myOFT,
    ) => """
    import sys, os
    import numpy as np

    oft_py = os.path.join(OFT_ROOT, "python")
    sys.path = [p for p in sys.path
                if not (p and p != oft_py
                        and os.path.isfile(os.path.join(p, "OpenFUSIONToolkit", "__init__.py")))]
    sys.path.insert(0, oft_py)

    import OpenFUSIONToolkit as _oft_pkg
    assert _oft_pkg.__file__.startswith(OFT_ROOT), (
        f"Wrong OpenFUSIONToolkit loaded: {_oft_pkg.__file__} (wanted {OFT_ROOT})")

    from OpenFUSIONToolkit.TokaMaker import TokaMaker
    from OpenFUSIONToolkit.TokaMaker.meshing import load_gs_mesh
    from OpenFUSIONToolkit.TokaMaker.util import read_eqdsk

    eqdsk = read_eqdsk(GEQDSK_FILE)

    mygs  = TokaMaker(myOFT)
    mygs.settings.maxits = 200

    mesh_pts, mesh_lc, mesh_reg, coil_dict, cond_dict = load_gs_mesh(MESH_FILE)
    mygs.setup_mesh(mesh_pts, mesh_lc, mesh_reg)
    mygs.setup_regions(cond_dict=cond_dict, coil_dict=coil_dict)
    mygs.setup(order=2, F0=abs(eqdsk["rcentr"] * eqdsk["bcentr"]))

    mygs.set_coil_vsc({"F9A": 1.0, "F9B": -1.0})

    mygs.set_targets(Ip=abs(eqdsk["ip"]), pax=eqdsk["pres"][0])

    isoflux_pts = eqdsk["rzout"].copy()
    weights     = np.ones(isoflux_pts.shape[0]) * 50.0
    mygs.set_isoflux(isoflux_pts, weights=weights)

    coil_names = [
        "ECOILA", "ECOILB",
        "F1A", "F2A", "F3A", "F4A", "F5A", "F6A", "F7A", "F8A", "F9A",
        "F1B", "F2B", "F3B", "F4B", "F5B", "F6B", "F7B", "F8B", "F9B",
    ]
    reg_terms = [mygs.coil_reg_term({c: 1.0}, target=0.0, weight=1.0) for c in coil_names]
    reg_terms.append(mygs.coil_reg_term({"#VSC": 1.0}, target=0.0, weight=1.0e-2))
    mygs.set_coil_reg(reg_terms=reg_terms)

    ffprim  = eqdsk["ffprim"]
    pprime0 = eqdsk["pprime"]
    psi_eqdsk = np.linspace(0.0, 1.0, np.size(ffprim))

    ffp_interp = np.interp(psi_prof, psi_eqdsk, ffprim)
    pp_interp  = np.interp(psi_prof, psi_eqdsk, pprime0)

    mygs.set_profiles(
        ffp_prof={"type": "linterp", "y": ffp_interp, "x": psi_prof},
        pp_prof ={"type": "linterp", "y": pp_interp,  "x": psi_prof},
    )

    mygs.init_psi(1.7, 0.0, 0.5, 1.3, -0.4)
    mygs.solve()
    mygs.print_info()

    psi_tkmkr, f_bl, fp_bl, p_bl, pp_bl = mygs.get_profiles(npsi=kin_nr)
    psi_q, q_bl, ravgs_bl, *_ = mygs.get_q(npsi=kin_nr, psi_pad=0.001)

    mu0 = 4.0e-7 * np.pi
    R_avg, oneOverR_avg = ravgs_bl['<R>'], ravgs_bl['<1/R>']
    jphi_baseline = R_avg * pp_bl + oneOverR_avg * (f_bl * fp_bl) / mu0

    ip_target = abs(eqdsk["ip"])
    """ => (mygs, psi_tkmkr::Vector{Float64}, jphi_baseline::Vector{Float64}, ip_target::Float64)

    equilibrium.Psi     = result.psi_tkmkr
    equilibrium.Jphi = result.jphi_baseline
    equilibrium.Ip_target     = result.ip_target
    return result.mygs
end

function setup_tokamaker_from_file!(device::Symbol, profiles::TokToxStability.Profiles, equilibrium::TokToxStability.Equilibrium, myOFT::Py; oft_root::String)
    return setup_tokamaker_from_file!(Val(device), profiles, equilibrium, myOFT; oft_root)
end
