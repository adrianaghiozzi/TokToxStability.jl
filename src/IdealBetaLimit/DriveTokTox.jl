using PythonCall

using PythonCall

"""
    TMTXInput

Bundles everything needed to build and run a single time-independent
(steady_state_mode) TokaMaker_TORAX solve, analogous to `GPECInput` in
DriveGPEC.jl.

`seed_geqdsk` is used as BOTH endpoints of `g_eqdsk_arr` (i.e. the
equilibrium is held fixed over [0, t_final]) -- this is what makes the run
time-independent. `psi_sample`/`ne_init`/`Te_init`/`Ti_init` are your
initial-guess kinetic profiles at t=0 (shape doesn't matter much, TORAX
relaxes them -- see the ITER notebook's Â§1.7 "Initial kinetic profiles").
"""
Base.@kwdef mutable struct TMTXInput
    oft_root::String
    mesh_file::String
    seed_geqdsk::String
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
    t_final::Float64            = 100.0           # s, long enough to equilibrate
    tx_dt::Float64              = 0.5
    torax_n_rho::Int            = 51
    heating::Dict{String,<:Any} = Dict{String,Any}("ecrh" => 0.0)
    fueling::Dict{String,<:Any} = Dict{String,Any}()
    max_loop::Int                = 2
    t_ave_window::Float64        = 25.0
    kin_nr::Union{Int,Nothing}   = nothing         # # of psi points for output profiles; defaults to length(psi_sample)
    last_surface_factor::Float64 = 0.995
end

"""
    _ensure_oft_on_path!(oft_root)

Same sys.path hygiene as the D3D-from-file path in utils.jl (avoid picking
up a stray OpenFUSIONToolkit install elsewhere on the path). Factored out
here so both setup paths can eventually share it.
"""
function _ensure_oft_on_path!(oft_root::String)
    @pyexec (OFT_ROOT=oft_root,) => """
    import sys, os
    oft_py = os.path.join(OFT_ROOT, "python")
    sys.path = [p for p in sys.path if p != oft_py]  # dedupe only, no destructive filtering
    sys.path.insert(0, oft_py)

    import OpenFUSIONToolkit as _oft_pkg
    assert _oft_pkg.__file__.startswith(OFT_ROOT), (
        f"Wrong OpenFUSIONToolkit loaded: {_oft_pkg.__file__} (wanted {OFT_ROOT})")
    """ => ()
    return nothing
end

"""
    make_oft_env(oft_root; nthreads=2) -> myOFT::Py

Constructs the single `OFT_env` instance for this Python kernel/session.
`OFT_env` is a hard process-wide singleton (OpenFUSIONToolkit raises
`RuntimeError: Only one instance of OFT_env can be created per python
kernel` on a second construction) -- since PythonCall embeds Python in the
same process as this Julia session, that means one `OFT_env` per Julia
session, not per case.

Call this once per session and thread the returned `myOFT` through to
`build_seed_tokamaker`/`setup_tokamaker_from_file!` for every case, instead
of restarting the whole kernel between runs. `TokaMaker` objects themselves
carry no such restriction (see `OpenFUSIONToolkit/TokaMaker/_core.py`'s
plain `__init__`, vs `OFT_env`'s guarded `__new__`), so many `TokaMaker(myOFT)`
instances can be built against the one shared environment.
"""
function make_oft_env(oft_root::String; nthreads::Int=2)
    _ensure_oft_on_path!(oft_root)
    oft_env_mod = pyimport("OpenFUSIONToolkit")
    return oft_env_mod.OFT_env(nthreads=nthreads)
end

"""
    build_seed_tokamaker(inp, myOFT) -> (mygs::Py, eqdsk::Py)

Loads the D3D mesh and builds a fresh TokaMaker object (against the given,
already-constructed `myOFT` -- see `make_oft_env`) at the seed eqdsk's F0,
mirroring the mesh/setup portion of `setup_tokamaker_from_file!` (utils.jl)
and cell 1.4 of the ITER notebook. Does NOT solve an equilibrium here --
TokaMaker_TORAX drives the solve itself once `fly()` is called.
"""
function build_seed_tokamaker(tmtx_input::TMTXInput, myOFT::Py)
    _ensure_oft_on_path!(tmtx_input.oft_root)

    TokaMaker = pyimport("OpenFUSIONToolkit.TokaMaker").TokaMaker
    meshing   = pyimport("OpenFUSIONToolkit.TokaMaker.meshing")
    util      = pyimport("OpenFUSIONToolkit.TokaMaker.util")

    eqdsk = util.read_eqdsk(tmtx_input.seed_geqdsk)

    mygs  = TokaMaker(myOFT)
    mygs.settings.maxits = 200

    mesh_pts, mesh_lc, mesh_reg, coil_dict, cond_dict = meshing.load_gs_mesh(tmtx_input.mesh_file)
    mygs.setup_mesh(mesh_pts, mesh_lc, mesh_reg)
    mygs.setup_regions(cond_dict=cond_dict, coil_dict=coil_dict)

    rcentr = pyconvert(Float64, eqdsk["rcentr"])
    bcentr = pyconvert(Float64, eqdsk["bcentr"])
    mygs.setup(order=2, F0=abs(rcentr * bcentr))

    return mygs, eqdsk
end

"""
    build_tmtx(inp, mygs) -> tmtx::Py

Constructs the TokaMaker_TORAX object with the equilibrium pinned to the
seed geqdsk at both ends of the window (time-independent).
"""
function build_tmtx(tmtx_input::TMTXInput, mygs::Py)
    pulse_design = pyimport("OpenFUSIONToolkit.TokaMaker.pulse_design")
    np           = pyimport("numpy")

    tm_times = np.linspace(0.0, tmtx_input.t_final, 3)

    tmtx = pulse_design.TokaMaker_TORAX(
        t_init=0.0,
        t_final=tmtx_input.t_final,
        tx_dt=tmtx_input.tx_dt,
        eqtimes=np.array([0.0, tmtx_input.t_final]),
        g_eqdsk_arr=[tmtx_input.seed_geqdsk, tmtx_input.seed_geqdsk],  # fixed equilibrium -> time-independent
        last_surface_factor=tmtx_input.last_surface_factor,
        tm_times=tm_times,
        tokamaker_obj=mygs,
    )
    return tmtx
end

"""
    configure_tmtx!(tmtx, mygs, inp) -> tmtx

Applies the tmtx.set* calls (ITER notebook Â§1.7-1.8). Coil bounds are built
generically from whatever coil sets the D3D mesh actually defines, rather
than hardcoding ITER's CS/PF/VS naming convention.
"""
function configure_tmtx!(tmtx::Py, mygs::Py, tmtx_input::TMTXInput)
    tmtx.set_TORAX_grid(grid_type="n_rho", grid=tmtx_input.torax_n_rho)

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
        diverted_times=(0.0, tmtx_input.t_final),
        x_point_targets=tmtx_input.x_points,
        x_point_weight=100,
    )

    return tmtx
end

"""
    _patch_torax_progress_bar!()

Monkey-patches `torax.run_simulation` AND `torax._src.orchestration.run_loop.run_loop`
so `progress_bar` is forced to `False` -- without touching TokaMaker's installed
`pulse_design.py` at all, so this keeps working against an unmodified TokaMaker/TORAX
install. A `tqdm` progress bar writes to stdout on every step; under IJulia's piped
(non-TTY) stdout, that can fill the pipe buffer faster than it's drained and block the
underlying `write()` syscall indefinitely -- this was observed directly: the whole
process going idle (0% CPU, every thread parked, main thread stuck in
`_io_TextIOWrapper_write`) rather than a slow or compiling simulation.

Originally (pre TokaMaker+TORAX `#318`) `pulse_design.py`'s `_run_tx()` called the
public `torax.run_simulation(myconfig, log_timestep_info=False)`, leaving
`progress_bar` at TORAX's own default (`True`); patching that one entry point (via
`kwargs.setdefault`) was enough. `#318` rewired the internal call path to
`torax._src.orchestration.run_loop.run_loop(..., progress_bar=True)` directly -- a
private module `pulse_design.py` now calls instead of the public API, with
`progress_bar=True` passed as an explicit literal at the call site (not a default),
so the old patch became silent dead code and no longer intercepts anything. This
second patch targets that actual call site, force-overriding `progress_bar` to
`False` unconditionally (a plain assignment, not `setdefault`, since the caller always
passes it explicitly) rather than relying on it being left at some default.

Idempotent (checks a marker attribute before wrapping each target), so it's safe to
call on every `run_tmtx_steady_state!` invocation, and safe to keep both patches
regardless of which internal path a given TokaMaker/TORAX version actually uses.
"""
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

"""
    run_tmtx_steady_state!(tmtx, inp; verbose=false) -> tmtx

Runs `fly()` in steady_state_mode -- takes the initial-guess profiles,
relaxes them via TORAX's time-dependent solver, and re-solves TokaMaker
each loop, but since the equilibrium (g_eqdsk_arr) doesn't vary in time
there's no genuine time evolution -- just convergence to a stationary state.
"""
function run_tmtx_steady_state!(tmtx::Py, tmtx_input::TMTXInput; verbose::Bool=false)
    _patch_torax_progress_bar!()
    quietly(; verbose) do
        tmtx.fly(
            run_name="tmtx_seed",
            max_loop=tmtx_input.max_loop,
            output_mode="normal",
            t_ave_toggle="on",
            t_ave_window=tmtx_input.t_ave_window,
            steady_state_mode=true,
        )
    end
    return tmtx
end

"""
    _psi_on_grid(dt, rho_dim, rho_vals) -> Vector{Float64}

Best-effort extraction of TORAX's normalized poloidal flux at the final
time, resampled onto the rho coordinate `rho_vals` (whose xarray dim name is
`rho_dim`). TORAX doesn't put every profile on the same grid -- it packs
each output variable's dim (`rho_norm`, `rho_face_norm`, or
`rho_cell_norm`) by matching array length at write time, so e.g. `psi_norm`
lives on the N+1-point face grid while `n_e`/`T_e`/`T_i` may live on a
different, shorter cell grid. Resolving psi generically against whichever
grid the caller's profile is actually on (rather than assuming they match)
avoids both a hard length-mismatch crash in `np.interp` and, more subtly,
misaligning psi[j] with profile[j] when the lengths happen to coincide.

NOT independently verified against a live TORAX session -- if this raises,
run `list(tmtx._data_tree.profiles.data_vars)` in Python and swap in
whatever the psi variable is actually called there.
"""
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

"""
    _profile_on_psi_out(dt, varname, psi_out) -> Vector{Float64}

Extracts `dt.profiles.<varname>` at the final time and resamples it onto
`psi_out`, using psi resampled onto THIS variable's own rho grid (see
`_psi_on_grid`) rather than a single psi array assumed to fit every profile.
"""
function _profile_on_psi_out(dt::Py, varname::String, psi_out::Vector{Float64})
    np = pyimport("numpy")
    da = getproperty(dt.profiles, varname).isel(time=-1)
    rho_dim = pyconvert(String, da.dims[0])
    rho_vals = pyconvert(Vector{Float64}, da.coords[rho_dim].values)
    psi_at_grid = _psi_on_grid(dt, rho_dim, rho_vals)
    values = pyconvert(Vector{Float64}, da.values)
    return np.interp(psi_out, psi_at_grid, values) |> x -> pyconvert(Vector{Float64}, x)
end

"""
    extract_profiles_and_equilibrium!(profiles, equilibrium, tmtx, mygs)

Writes the converged steady-state solution into the shared `Profiles`/
`Equilibrium` objects: Ti/Te/ni/ne/pressure come from TORAX (each resampled
from its own native rho grid, via `_profile_on_psi_out`), Psi/Jphi/Ip_target
come from the re-solved TokaMaker object -- mirroring the extraction at the
end of `setup_tokamaker_from_file!`.
"""
function extract_profiles_and_equilibrium!(profiles::Profiles, equilibrium::Equilibrium, tmtx::Py, mygs::Py, tmtx_input::TMTXInput)
    kin_nr = something(tmtx_input.kin_nr, length(tmtx_input.psi_sample))

    dt = tmtx._data_tree
    psi_out = collect(range(0.0, 1.0; length=kin_nr))

    ne = _profile_on_psi_out(dt, "n_e", psi_out)
    Te = _profile_on_psi_out(dt, "T_e", psi_out) .* 1e3  # keV -> eV
    Ti = _profile_on_psi_out(dt, "T_i", psi_out) .* 1e3  # keV -> eV
    # TODO: if TORAX exposes n_i directly (dt.profiles.n_i) prefer that over
    # assuming ni == ne; leaving as ne here mirrors the simplification pattern
    # already visible elsewhere (single-ion-species D3D case).
    ni = ne
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

"""
    setup_from_tmtx!(::Val{:D3D}, profiles, equilibrium, inp::TMTXInput, myOFT::Py; verbose=false) -> mygs::Py

Runs the full time-independent TMTX pipeline: build the seed TokaMaker
object from the existing D3D mesh/eqdsk (against the given, already-built
`myOFT` -- see `make_oft_env`), construct and configure the TokaMaker_TORAX
object, solve once in steady_state_mode, then write the converged
profiles/equilibrium into `profiles`/`equilibrium` so the existing
coarse-bracketing scan (DriveTokaMaker.jl) can pick up unchanged. Returns
the re-solved TokaMaker instance for `tokamaker_run`/`coarse_scan`.
"""
function setup_from_tmtx!(::Val{:D3D}, tmtx_input::TMTXInput, myOFT::Py; verbose::Bool=false)
    profiles = Profiles(Ti=nothing, Te=nothing, ni=nothing, ne=nothing,
    pressure=nothing, pprime=nothing)
    equilibrium = Equilibrium(Psi=nothing, Jphi=nothing, Ip_target=nothing)
    
    mygs, _eqdsk = build_seed_tokamaker(tmtx_input, myOFT)
    tmtx = build_tmtx(tmtx_input, mygs)
    configure_tmtx!(tmtx, mygs, tmtx_input)
    run_tmtx_steady_state!(tmtx, tmtx_input; verbose)
    extract_profiles_and_equilibrium!(profiles, equilibrium, tmtx, mygs, tmtx_input)

    return mygs, profiles, equilibrium
end

function setup_from_tmtx!(device::Symbol, tmtx_input::TMTXInput, myOFT::Py; verbose::Bool=false)
    return setup_from_tmtx!(Val(device), tmtx_input, myOFT; verbose)
end