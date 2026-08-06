using PythonCall
using DelimitedFiles
using Printf

# function read_profiles_from_toktox and write those out to the Profiles object (this is the alternative to getting the profiles from a KEFIT)


default_tag(frac::Real) = "f" * replace(@sprintf("%.4f", frac), "." => "p")

function coarse_scan(fracs::Vector{<:Real}, profiles::Profiles, n_tor, wall, kinetic, work_dir; verbose::Bool = false)
    coarse_scan_results = []
    for frac in fracs
        try
            result = evaluate_factor(profiles, frac, n_tor, wall, kinetic, work_dir; verbose)
            @show typeof(result)
            @show result.frac
            push!(coarse_scan_results, result)
        catch e 
            println(e)
            continue # put a retry in here
        end
    end
    return coarse_scan_results

end

function evaluate_factor(profiles::Profiles, frac::Real, n_tor::Int,
                          wall::Dict{String,<:Any}, kinetic::Dict{String,<:Any},
                          work_dir::String; tag::Union{String,Nothing}=nothing,
                          parallel_threads::Int=2, verbose::Bool=false)
    tag = something(tag, default_tag(frac))
    run_directory = joinpath(work_dir, tag)

    tkm = quietly(; verbose) do
        solve_at_pressure_factor!(profiles, frac)
    end

    geqdsk_path = write_geqdsk_for_gpec(run_directory, tag)
    eq_filename = basename(geqdsk_path)

    gi = GPECInput(eq_filename=eq_filename, n_tor=n_tor, wall=wall,
                    kinetic=kinetic, parallel_threads=parallel_threads)
    gpec_raw = quietly(; verbose) do
        run_GPEC(gi, run_directory)
    end
    gp = read_output(gpec_raw)

    return (frac=frac, tag=tag, beta_n=tkm.beta_n, beta_t=tkm.beta_t,
            p_axis=profiles.pressure[1], dW_total=gp.dW_total, dW_plasma=gp.dW_plasma,
            dW_vacuum=gp.dW_vacuum, q0=gp.q0, qmax=gp.qmax, msing=gp.msing,
            run_dir=run_directory)
end

# something like tokamaker_run(profiles::Vector(Profiles)) to do the looping part 
function tokamaker_run(profiles::Vector{Profiles}; n_settle::Int = 3)
    # do the profile rescaling 
    # write the intermediate file 
    # run tokamaker for that one profile
    # do it again 

end

function tokamaker_run(profiles::Profiles; n_settle::Int = 3)
    profiles.pressure === nothing && error("profiles.pressure is nothing -- call rescale_profile first")
    isassigned(_mygs) || error("no TokaMaker instance set up -- run the setup_test_case cell first")
 
    mygs          = _mygs[]
    psi_tkmkr     = _psi_tkmkr[]
    jphi_baseline = _jphi_baseline[]
    ip_target     = _ip_target[]
 
    nl_its = nothing
    for _ in 1:n_settle
        psi_bounds = mygs.psi_bounds
        psi_b0 = pyconvert(Float64, psi_bounds[0])
        psi_b1 = pyconvert(Float64, psi_bounds[1])
 
        profiles.pprime = np_gradient(profiles.pressure) ./
                           (np_gradient(psi_tkmkr) .* (psi_b1 - psi_b0))
 
        mygs.set_targets(Ip=ip_target, pax=profiles.pressure[1])
 
        mygs.set_profiles(
            pp_prof  = Dict("type" => "linterp",      "y" => profiles.pprime,  "x" => psi_tkmkr),
            ffp_prof = Dict("type" => "jphi-linterp", "y" => jphi_baseline,    "x" => profiles.Psi),
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
    )
end

function solve_at_pressure_factor!(profiles::Profiles, frac::Real)
    new_profiles = rescale_profile!(profiles, frac)
    nl_its, stats, beta_n, beta_t = tokamaker_run(new_profiles)
end

function write_geqdsk_for_gpec(run_directory::String, tag, nr::Int=257, nz::Int=257)
    mygs = _mygs[]

    mkpath(run_directory)
    out = joinpath(run_directory, "tokamaker_$(tag).geqdsk")

    mygs.save_eqdsk(out; nr=nr, nz=nz, truncate_eq=false, lcfs_pad=0.001)
    return out 
end
