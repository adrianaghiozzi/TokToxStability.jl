using PythonCall

using PythonCall

Base.@kwdef mutable struct TMTXInput
    seed_geqdsk::String                          # path to a gEQDSK file seeding the coupled solve -- see `default_d3d_163303_seed_geqdsk()` for a ready-made D3D default
    Ip::Float64                                  # A, flattop target current
    x_points::Matrix{Float64}                    # Nx2, [[R,Z], ...]
    psi_sample::Vector{Float64}
    ne_init::Vector{Float64}                     # m^-3
    Te_init::Vector{Float64}                     # keV (TORAX convention)
    Ti_init::Vector{Float64}                     # keV
    ne_edge::Float64                              # m^-3
    Te_edge::Float64                              # keV
    Ti_edge::Float64                              # keV
    pedestal::Dict{String,<:Any}
    plasma_composition::Dict{String,<:Any}        # e.g. Dict("main_ion"=>Dict("D"=>0.5,"T"=>0.5), "impurity"=>"Ne", "Zeff"=>1.7)
    heating::Dict{String,<:Any} = Dict{String,Any}("ecrh" => 0.0)
    fueling::Dict{String,<:Any} = Dict{String,Any}()
end

Base.@kwdef mutable struct TMTXSettings
    max_loop::Int                = 2
    t_ave_window::Float64        = 25.0
    kin_nr::Union{Int,Nothing}   = nothing         # # of psi points for output profiles; defaults to length(psi_sample)
    last_surface_factor::Float64 = 0.995
    t_final::Float64            = 100.0           # s, long enough to equilibrate
    tx_dt::Float64              = 0.5
    torax_n_rho::Int            = 51
end

"""
    make_oft_env(; nthreads=2)

Build the OFT environment used by every TokaMaker call in this package.
`OpenFUSIONToolkit` is resolved the normal way (`pyimport`) -- see `CondaPkg.toml`
for how it's pinned/installed. No local OFT checkout or path setup is required.
"""
function make_oft_env(; nthreads::Int=2)
    oft_env_mod = pyimport("OpenFUSIONToolkit")
    return oft_env_mod.OFT_env(nthreads=nthreads)
end

"""
    default_d3d_163303_seed_geqdsk()

Path to a packaged D3D seed gEQDSK, generated from shot 163303 (the standard DIII-D
modeling reference discharge) at t=5.0s -- a clean flattop point (Ip steady at
1.13-1.17 MA over t=4.8-6.7s). The equilibrium itself was solved from scratch (mesh,
D3D coil set, isoflux/saddle/psi constraints, `init_psi` guess) using shape and target
data pulled once via `FUSE.case_parameters(:D3D, 163303; omfit_host="omega.gat.com")`
+ `FUSE.init(ini, act)`, off `dd.equilibrium.time_slice[208]`; only the physically
relevant (lower, on-boundary) X-point was used -- IMAS also reports a second null far
outside the LCFS outline that isn't part of the actual diverted shape.

Use as `TMTXInput(; seed_geqdsk=default_d3d_163303_seed_geqdsk(), <rest of your inputs>)`
for a quick D3D run; pass a different gEQDSK path by hand for a different shot/scenario.
"""
function default_d3d_163303_seed_geqdsk()
    return joinpath(pkgdir(TokToxStability), "cases", "D3D", "g163303.05000_DIIID_seed.eqdsk")
end

function build_seed_tokamaker(::Val{:D3D}, tmtx_input::TMTXInput, myOFT::Py)
    TokaMaker = pyimport("OpenFUSIONToolkit.TokaMaker").TokaMaker
    meshing   = pyimport("OpenFUSIONToolkit.TokaMaker.meshing")
    util      = pyimport("OpenFUSIONToolkit.TokaMaker.util")

    d3d_case_dir = joinpath(pkgdir(TokToxStability), "cases", "D3D")
    mesh_file = joinpath(d3d_case_dir, "DIIID_mesh.h5")

    eqdsk = util.read_eqdsk(tmtx_input.seed_geqdsk)

    mygs  = TokaMaker(myOFT)
    mygs.settings.maxits = 200

    mesh_pts, mesh_lc, mesh_reg, coil_dict, cond_dict = meshing.load_gs_mesh(mesh_file)
    mygs.setup_mesh(mesh_pts, mesh_lc, mesh_reg)
    mygs.setup_regions(cond_dict=cond_dict, coil_dict=coil_dict)

    rcentr = pyconvert(Float64, eqdsk["rcentr"])
    bcentr = pyconvert(Float64, eqdsk["bcentr"])
    mygs.setup(order=2, F0=abs(rcentr * bcentr))

    return mygs, eqdsk
end

function build_seed_tokamaker(device::Symbol, tmtx_input::TMTXInput, myOFT::Py)
    return build_seed_tokamaker(Val(device), tmtx_input, myOFT)
end

function build_tmtx(tmtx_input::TMTXInput, tmtx_settings::TMTXSettings, mygs::Py)
    pulse_design = pyimport("OpenFUSIONToolkit.TokaMaker.pulse_design")
    np           = pyimport("numpy")

    tm_times = np.linspace(0.0, tmtx_settings.t_final, 3)

    tmtx = pulse_design.TokaMaker_TORAX(
        t_init=0.0,
        t_final=tmtx_settings.t_final,
        tx_dt=tmtx_settings.tx_dt,
        eqtimes=np.array([0.0, tmtx_settings.t_final]),
        g_eqdsk_arr=[tmtx_input.seed_geqdsk, tmtx_input.seed_geqdsk],  # fixed equilibrium -> time-independent
        last_surface_factor=tmtx_settings.last_surface_factor,
        tm_times=tm_times,
        tokamaker_obj=mygs,
    )
    return tmtx
end

function configure_tmtx!(tmtx::Py, mygs::Py, tmtx_input::TMTXInput, tmtx_settings::TMTXSettings)
    tmtx.set_TORAX_grid(grid_type="n_rho", grid=tmtx_settings.torax_n_rho)

    tmtx.set_heating(; (Symbol(k) => v for (k, v) in tmtx_input.heating)...)
    tmtx.set_fueling(; (Symbol(k) => v for (k, v) in tmtx_input.fueling)...)

    array_to_profile_dict(arr, grid) = pydict(Dict(Float64(p) => Float64(v) for (p, v) in zip(grid, arr)))
    t0 = 0.0
    tmtx.set_ne(pydict(Dict(t0 => array_to_profile_dict(tmtx_input.ne_init, tmtx_input.psi_sample))), right_bc=tmtx_input.ne_edge)
    tmtx.set_Te(pydict(Dict(t0 => array_to_profile_dict(tmtx_input.Te_init, tmtx_input.psi_sample))), right_bc=tmtx_input.Te_edge)
    tmtx.set_Ti(pydict(Dict(t0 => array_to_profile_dict(tmtx_input.Ti_init, tmtx_input.psi_sample))), right_bc=tmtx_input.Ti_edge)

    tmtx.load_pedestal_config(pydict(tmtx_input.pedestal))

    tmtx._tm = mygs
    coil_names  = pyconvert(Vector{String}, mygs.coil_sets)
    coil_bounds = pydict(Dict(name => [-50.0e6, 50.0e6] for name in coil_names))
    tmtx.set_TokaMaker_coil_reg(coil_bounds=coil_bounds, updownsym=false)

    tmtx.set_Ip(pydict(Dict(t0 => tmtx_input.Ip)))
    tmtx.set_plasma_composition(; (Symbol(k) => v for (k, v) in tmtx_input.plasma_composition)...)
    tmtx.set_evolve(density=true, Ti=true, Te=true, current=true)

    tmtx.set_diverted_shape_targets(
        diverted_times=(0.0, tmtx_settings.t_final),
        x_point_targets=tmtx_input.x_points,
        x_point_weight=100,
    )

    return tmtx
end

function _patch_torax_progress_bar!()
    @pyexec """
    import torax
    if not getattr(torax.run_simulation, "_tts_progress_bar_patched", False):
        # _orig is a default arg (bound at def-time) rather than a closed-over name,
        # because exec()'d code with separate globals/locals (which is what pyexec
        # uses) gives top-level defs class-body scoping: a nested function can't see
        # names assigned at that same exec top level via closure/__globals__ lookup.
        def _tts_run_simulation(*args, _orig=torax.run_simulation, **kwargs):
            kwargs.setdefault("progress_bar", False)
            return _orig(*args, **kwargs)
        _tts_run_simulation._tts_progress_bar_patched = True
        torax.run_simulation = _tts_run_simulation

    from torax._src.orchestration import run_loop as _tts_run_loop_mod
    if not getattr(_tts_run_loop_mod.run_loop, "_tts_progress_bar_patched", False):
        def _tts_run_loop(*args, _orig=_tts_run_loop_mod.run_loop, **kwargs):
            kwargs["progress_bar"] = False  # force, not setdefault -- caller passes it explicitly
            return _orig(*args, **kwargs)
        _tts_run_loop._tts_progress_bar_patched = True
        _tts_run_loop_mod.run_loop = _tts_run_loop
    """
    return nothing
end

function run_tmtx_steady_state!(tmtx::Py, tmtx_settings::TMTXSettings; verbose::Bool=false)
    _patch_torax_progress_bar!()
    quietly(; verbose) do
        tmtx.fly(
            run_name="tmtx_seed",
            max_loop=tmtx_settings.max_loop,
            output_mode="normal",
            t_ave_toggle="on",
            t_ave_window=tmtx_settings.t_ave_window,
            steady_state_mode=true,
        )
    end
    return tmtx
end

function _psi_on_grid(dt::Py, rho_dim::String, rho_vals::Vector{Float64})
    np = pyimport("numpy")
    for name in ("psi_norm", "psi_hat", "psi")
        if pyconvert(Bool, pyhasattr(dt.profiles, name))
            da = getproperty(dt.profiles, name).isel(time=-1)
            psi_vals = pyconvert(Vector{Float64}, da.values)
            if name == "psi"  # raw psi, not yet normalized
                psi_vals = (psi_vals .- psi_vals[1]) ./ (psi_vals[end] - psi_vals[1])
            end
            psi_rho_dim = pyconvert(String, da.dims[0])
            psi_rho_dim == rho_dim && return psi_vals
            psi_rho_vals = pyconvert(Vector{Float64}, da.coords[psi_rho_dim].values)
            return np.interp(rho_vals, psi_rho_vals, psi_vals) |> x -> pyconvert(Vector{Float64}, x)
        end
    end
    error("Could not find a psi variable on tmtx._data_tree.profiles under any of the " *
          "expected names. In Python, run `list(tmtx._data_tree.profiles.data_vars)` " *
          "and update `_psi_on_grid` in DriveTokTox.jl with the correct name.")
end

function _profile_on_psi_out(dt::Py, varname::String, psi_out::Vector{Float64})
    np = pyimport("numpy")
    da = getproperty(dt.profiles, varname).isel(time=-1)
    rho_dim = pyconvert(String, da.dims[0])
    rho_vals = pyconvert(Vector{Float64}, da.coords[rho_dim].values)
    psi_at_grid = _psi_on_grid(dt, rho_dim, rho_vals)
    values = pyconvert(Vector{Float64}, da.values)
    return np.interp(psi_out, psi_at_grid, values) |> x -> pyconvert(Vector{Float64}, x)
end

function extract_profiles_and_equilibrium!(profiles::Profiles, equilibrium::Equilibrium, tmtx::Py, mygs::Py, tmtx_input::TMTXInput, tmtx_settings::TMTXSettings)
    kin_nr = something(tmtx_settings.kin_nr, length(tmtx_input.psi_sample))

    dt = tmtx._data_tree
    psi_out = collect(range(0.0, 1.0; length=kin_nr))

    ne = _profile_on_psi_out(dt, "n_e", psi_out)
    ni = _profile_on_psi_out(dt, "n_i", psi_out)
    Te = _profile_on_psi_out(dt, "T_e", psi_out) .* 1e3  # keV -> eV
    Ti = _profile_on_psi_out(dt, "T_i", psi_out) .* 1e3  # keV -> eV
    pressure = boltzmann_constant .* (ne .* Te .+ ni .* Ti)

    psi_tkmkr, f_bl, fp_bl, p_bl, pp_bl = mygs.get_profiles(npsi=kin_nr)
    ip_target = pyconvert(Float64, mygs.get_stats().get("Ip"))

    mu0 = 4.0e-7 * pi
    psi_q, q_bl, ravgs_bl = mygs.get_q(npsi=kin_nr, psi_pad=0.001)
    R_avg        = pyconvert(Vector{Float64}, ravgs_bl["<R>"])
    oneOverR_avg = pyconvert(Vector{Float64}, ravgs_bl["<1/R>"])
    f_bl_j  = pyconvert(Vector{Float64}, f_bl)
    fp_bl_j = pyconvert(Vector{Float64}, fp_bl)
    pp_bl_j = pyconvert(Vector{Float64}, pp_bl)
    jphi = R_avg .* pp_bl_j .+ oneOverR_avg .* (f_bl_j .* fp_bl_j) ./ mu0

    profiles.Ti = Ti
    profiles.Te = Te
    profiles.ni = ni
    profiles.ne = ne
    profiles.pressure = pressure
    profiles.pprime = nothing  # recomputed each settle iteration in tokamaker_run

    equilibrium.Psi = pyconvert(Vector{Float64}, psi_tkmkr)
    equilibrium.Jphi = jphi
    equilibrium.Ip_target = ip_target

    return nothing
end

function setup_from_tmtx!(::Val{:D3D}, tmtx_input::TMTXInput, tmtx_settings::TMTXSettings, myOFT::Py; verbose::Bool=false)
    profiles = Profiles(Ti=nothing, Te=nothing, ni=nothing, ne=nothing,
    pressure=nothing, pprime=nothing)
    equilibrium = Equilibrium(Psi=nothing, Jphi=nothing, Ip_target=nothing)
    
    mygs, _eqdsk = build_seed_tokamaker(Val(:D3D), tmtx_input, myOFT)
    tmtx = build_tmtx(tmtx_input, tmtx_settings, mygs)
    configure_tmtx!(tmtx, mygs, tmtx_input, tmtx_settings)
    run_tmtx_steady_state!(tmtx, tmtx_settings; verbose)
    extract_profiles_and_equilibrium!(profiles, equilibrium, tmtx, mygs, tmtx_input, tmtx_settings)

    return mygs, profiles, equilibrium
end

function setup_from_tmtx!(device::Symbol, tmtx_input::TMTXInput, tmtx_settings::TMTXSettings, myOFT::Py; verbose::Bool=false)
    return setup_from_tmtx!(Val(device), tmtx_input, tmtx_settings, myOFT; verbose)
end