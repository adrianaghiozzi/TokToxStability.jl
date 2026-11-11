using PythonCall
using DelimitedFiles
using Printf

default_tag(fraction::Real) = "f" * replace(@sprintf("%.4f", fraction), "." => "p")

_is_tokamaker_solve_failure(e) = e isa ErrorException && startswith(e.msg, "TokaMaker did not converge")

# A scan skips points that fail (e.g. TokaMaker not converging at high pressure), but if *every*
# point fails the cause is almost always systematic (bad gpec.toml, API mismatch), so surface it.
function _warn_scan_failure(fn::String, fraction, e)
    @warn "$fn: fraction = $fraction failed and was skipped" exception = e
end
function _check_scan_not_all_failed(fn::String, results, n_failed::Int, last_error)
    if isempty(results) && n_failed > 0
        error("$fn: all $n_failed scan points failed; the last error was:\n" * sprint(showerror, last_error))
    end
end

function coarse_scan(mygs::Py, fractions::Vector{<:Real}, equilibrium::Equilibrium, profiles::Profiles,
                      gpec_settings::GPECSettings; working_directory::Union{String,Nothing}=nothing, verbose::Bool = false)
    psi_baseline = mygs.get_psi(normalized=false)
    coarse_scan_results = []
    mygs_possibly_corrupted = false
    n_failed, last_error = 0, nothing
    for fraction in fractions
        try
            mygs.set_psi(psi_baseline, update_bounds=true)
            result = evaluate_factor(mygs, deepcopy(profiles), deepcopy(equilibrium), fraction, gpec_settings; working_directory, verbose)
            push!(coarse_scan_results, result)
        catch e
            _is_tokamaker_solve_failure(e) && (mygs_possibly_corrupted = true)
            _warn_scan_failure("coarse_scan", fraction, e)
            n_failed += 1; last_error = e
            continue
        end
    end
    mygs_possibly_corrupted && @warn "coarse_scan: a TokaMaker solve failed to converge during this scan, " *
                                      "leaving `mygs`'s internal solver state mid-iteration. Rebuild `mygs` " *
                                      "(e.g. via `setup_from_tmtx!`/`setup_tokamaker_from_file!`) before running " *
                                      "another scan on it -- reusing this `mygs` risks the failure cascading into " *
                                      "future scans."
    _check_scan_not_all_failed("coarse_scan", coarse_scan_results, n_failed, last_error)
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

        psi_q, q_profile, _ravgs, _rest... = pyconvert(Tuple, mygs.get_q(npsi=length(equilibrium.Psi), psi_pad=0.001))

        geqdsk_path = write_geqdsk_for_gpec(mygs, run_directory, tag)

        gpec_raw = quietly(; verbose) do
            run_GPEC(gpec_settings, basename(geqdsk_path), run_directory)
        end
        gp = read_output(gpec_raw)
        pressure = pressure_profile(profiles)

        # Glasser/inner-layer terms (D_R, Delta_crit, ...) need profiles.omega (E x B rotation),
        # only populated when the kinetic file carries wexb(rad/s) -- i.e. the tearing workflow.
        # gpec_raw (and its ffs.equil/ffs.surfaces) isn't kept past this function, so compute this
        # here rather than deferring it to a later readout step like delta_prime_for_mode.
        glasser_terms = profiles.omega === nothing ? nothing :
                        quietly(; verbose) do
                            compute_glasser_terms(gpec_raw, profiles, gpec_settings)
                        end

        return (fraction=fraction, tag=tag, beta_n=beta_n, beta_t=beta_t, l_i=l_i,
                p_axis=pressure[1], dW_total=gp.dW_total, dW_plasma=gp.dW_plasma,
                dW_vacuum=gp.dW_vacuum, q0=gp.q0, qmax=gp.qmax, msing=gp.msing,
                delta_prime_matrix=gp.delta_prime_matrix, singular_surfaces=gp.singular_surfaces,
                glasser_terms=glasser_terms,
                run_dir=(ephemeral ? nothing : run_directory), geqdsk_path=(ephemeral ? nothing : geqdsk_path),
                # Per-iteration profile/q snapshots for plots.jl. `psi` is the
                # grid `profiles`/`equilibrium` are already sampled on (post
                # rescale_profile!), so these all line up index-for-index.
                psi=copy(equilibrium.Psi), Te=copy(profiles.Te), Ti=copy(profiles.Ti),
                ne=copy(profiles.ne), pressure=pressure,
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
    mygs_possibly_corrupted = false
    n_failed, last_error = 0, nothing
    while attempts < max_fractions
        attempts += 1
        mygs.set_psi(psi_baseline, update_bounds=true)
        result = try
            evaluate_factor(mygs, deepcopy(profiles), deepcopy(equilibrium), fraction, gpec_settings; working_directory, verbose)
        catch e
            _is_tokamaker_solve_failure(e) && (mygs_possibly_corrupted = true)
            _warn_scan_failure("coarse_scan_until_unstable", fraction, e)
            n_failed += 1; last_error = e
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

    _check_scan_not_all_failed("coarse_scan_until_unstable", results, n_failed, last_error)

    crossed_at === nothing && @warn "coarse_scan_until_unstable: dW_total never went negative within " *
                                     "max_fractions=$max_fractions (last fraction tried = $(fraction - step)) " *
                                     "-- results may not bracket the true beta limit"

    mygs_possibly_corrupted && @warn "coarse_scan_until_unstable: a TokaMaker solve failed to converge during " *
                                      "this scan, leaving `mygs`'s internal solver state mid-iteration. Rebuild " *
                                      "`mygs` (e.g. via `setup_from_tmtx!`/`setup_tokamaker_from_file!`) before " *
                                      "running another scan on it -- reusing this `mygs` risks the failure " *
                                      "cascading into future scans."

    return results
end

function tokamaker_run(mygs::Py, profiles::Profiles, equilibrium::Equilibrium; n_settle::Int = 3)
    nl_its = nothing

    for _ in 1:n_settle
        psi_bounds = mygs.psi_bounds
        psi_b0 = pyconvert(Float64, psi_bounds[0])
        psi_b1 = pyconvert(Float64, psi_bounds[1])

        pressure = pressure_profile(profiles)
        pprime = numpy_like_gradient(pressure) ./
                 (numpy_like_gradient(equilibrium.Psi) .* (psi_b1 - psi_b0))

        mygs.set_targets(Ip=equilibrium.Ip_target, pax=pressure[1])

        mygs.set_profiles(
            pp_prof  = Dict("type" => "linterp",      "y" => pprime,  "x" => equilibrium.Psi),
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

function setup_tokamaker_from_file!(::Val{:D3D}, profiles::TokToxStability.Profiles, equilibrium::TokToxStability.Equilibrium, myOFT::Py)
    kin_nr = length(profiles.Psi)
    d3d_case_dir = joinpath(pkgdir(TokToxStability), "cases", "D3D")
    mesh_file = joinpath(d3d_case_dir, "DIIID_mesh.h5")

    result = @pyexec (
        MESH_FILE=mesh_file,
        rcentr=equilibrium.rcentr, bcentr=equilibrium.bcentr, ip=equilibrium.ip,
        pax=equilibrium.pax, rzout=equilibrium.rzout, ffprim=equilibrium.ffprim, pprime=equilibrium.pprime,
        psi_prof=profiles.Psi, kin_nr=kin_nr, myOFT=myOFT,
    ) => """
    import numpy as np

    from OpenFUSIONToolkit.TokaMaker import TokaMaker
    from OpenFUSIONToolkit.TokaMaker.meshing import load_gs_mesh

    mygs  = TokaMaker(myOFT)
    mygs.settings.maxits = 200

    mesh_pts, mesh_lc, mesh_reg, coil_dict, cond_dict = load_gs_mesh(MESH_FILE)
    mygs.setup_mesh(mesh_pts, mesh_lc, mesh_reg)
    mygs.setup_regions(cond_dict=cond_dict, coil_dict=coil_dict)
    mygs.setup(order=2, F0=abs(rcentr * bcentr))

    mygs.set_coil_vsc({"F9A": 1.0, "F9B": -1.0})

    mygs.set_targets(Ip=abs(ip), pax=pax)

    isoflux_pts = np.asarray(rzout)
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

    psi_eqdsk = np.linspace(0.0, 1.0, len(ffprim))

    ffp_interp = np.interp(psi_prof, psi_eqdsk, ffprim)
    pp_interp  = np.interp(psi_prof, psi_eqdsk, pprime)

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

    ip_target = abs(ip)
    """ => (mygs, psi_tkmkr::Vector{Float64}, jphi_baseline::Vector{Float64}, ip_target::Float64)

    equilibrium.Psi     = result.psi_tkmkr
    equilibrium.Jphi = result.jphi_baseline
    equilibrium.Ip_target     = result.ip_target
    return result.mygs
end

function setup_tokamaker_from_file!(device::Symbol, profiles::TokToxStability.Profiles, equilibrium::TokToxStability.Equilibrium, myOFT::Py)
    return setup_tokamaker_from_file!(Val(device), profiles, equilibrium, myOFT)
end
