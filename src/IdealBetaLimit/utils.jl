using PythonCall 
using Logging

const boltzmann_constant = 1.6022e-19

# just for testing, get rid of this later #
const _mygs          = Ref{Py}() 
###########################################

Base.@kwdef mutable struct Profiles 
    Ti::Union{Vector{Float64},Nothing}
    Te::Union{Vector{Float64},Nothing}
    ni::Union{Vector{Float64},Nothing}
    ne::Union{Vector{Float64},Nothing}
    pressure::Union{Vector{Float64},Nothing}
    pprime::Union{Vector{Float64},Nothing}
end

Base.@kwdef mutable struct Equilibrium 
    Psi::Union{Vector{Float64},Nothing}
    Jphi::Union{Vector{Float64},Nothing}
    Ip_target::Union{Float64,Nothing}
end

function read_kinetic_profiles_from_file(path::String)
    data, header = readdlm(path, header=true)
    cols = Dict(strip(String(h)) => i for (i, h) in enumerate(header))
 
    col(name) = Vector{Float64}(data[:, cols[name]])
    Ti = col("ti(eV)")
    Te = col("te(eV)")
    ni = col("ni(m^-3)")
    ne = col("ne(m^-3)")
    pressure = boltzmann_constant .* (ne .* Te .+ ni .* Ti)
 
    return Profiles(
        Ti,
        Te,
        ni,
        ne,
        pressure,
        nothing
    )
end

function read_equilibrium_from_file(path::String)
    data, header = readdlm(path, header=true)
    cols = Dict(strip(String(h)) => i for (i, h) in enumerate(header))
 
    col(name) = Vector{Float64}(data[:, cols[name]])

    return Equilibrium(
        Psi = col("psi"),
        Jphi = nothing,
        Ip_target = nothing

    )
end

function rescale_profile!(profiles::Profiles, frac::Real) # ! because this is editing in place 
    profiles.Te .*= frac 
    profiles.Ti .*= frac 
    profiles.pressure = boltzmann_constant .* (profiles.ne .* profiles.Te .+ profiles.ni .* profiles.Ti)
    return profiles
end

# function setup_from_tmtx and write TMTX results to the Profiles and Equilibrium objects (this is the alternative to getting the profiles from a KEFIT)


function setup_tokamaker_from_file!(::Val{:D3D}, profiles::TokToxStability.Profiles, equilibrium::TokToxStability.Equilibrium; oft_root::String)
    kin_nr = length(equilibrium.Psi)
    d3d_case_dir = joinpath(pkgdir(TokToxStability), "cases", "D3D")
    mesh_file = joinpath(d3d_case_dir, "DIIID_mesh.h5")
    geqdsk_file = joinpath(d3d_case_dir, "g147131.02300_DIIID_KEFIT")

    result = @pyexec (
        OFT_ROOT=oft_root, MESH_FILE=mesh_file, GEQDSK_FILE=geqdsk_file,
        psi_prof=equilibrium.Psi, kin_nr=kin_nr,
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

    from OpenFUSIONToolkit import OFT_env
    from OpenFUSIONToolkit.TokaMaker import TokaMaker
    from OpenFUSIONToolkit.TokaMaker.meshing import load_gs_mesh
    from OpenFUSIONToolkit.TokaMaker.util import read_eqdsk

    eqdsk = read_eqdsk(GEQDSK_FILE)

    myOFT = OFT_env(nthreads=2)
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

    TokToxStability._mygs[]          = result.mygs # get rid of this global thing
    equilibrium.Psi     = result.psi_tkmkr
    equilibrium.Jphi = result.jphi_baseline
    equilibrium.Ip_target     = result.ip_target
    return nothing
end 

# in the future could add setup_tokamaker_from_file(::Val{:ITER})

function np_gradient(f::AbstractVector{<:Real}) # replace this later with native Julia gradient 
    n = length(f)
    n < 2 && throw(ArgumentError("np_gradient needs at least 2 points"))
    g = zeros(Float64, n)
    g[1]   = f[2] - f[1]                # forward difference at the start
    g[end] = f[end] - f[end-1]          # backward difference at the end
    @inbounds for i in 2:n-1
        g[i] = (f[i+1] - f[i-1]) / 2    # central difference, interior
    end
    return g
end

function quietly(f::Function; verbose::Bool=false)
    verbose && return f()
    return redirect_stdout(devnull) do
        redirect_stderr(devnull) do
            with_logger(ConsoleLogger(stderr, Logging.Warn)) do
                f()
            end
        end
    end
end