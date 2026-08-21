using GeneralizedPerturbedEquilibrium

"""
    GPECSettings(; toroidal_mode_number, wall=false, a=nothing,
                kinetic_source="fixed", kinetic_factor=0.0, parallel_threads=2)

Build one of these yourself, set the options you want, and pass the whole
object into `coarse_scan`/`coarse_scan_until_unstable`/`evaluate_factor` --
they read `toroidal_mode_number`/`wall`/`a`/`kinetic_source`/`kinetic_factor`/
`parallel_threads` straight off it, so there's one place to see and change
your GPEC settings instead of five loose parameters threaded through every
call site.

No `equilibrium_filename` here: that's the gEQDSK for whichever fraction is
currently being solved, an internal detail of `evaluate_factor`/`run_GPEC`,
not a setting -- there's nothing to choose and nothing to look at, so it
isn't part of the object you configure. It's passed as a plain argument to
`run_GPEC` instead.

`wall` and `kinetic` are exposed as simple, individually-overridable options
rather than raw GPEC-toml-shaped Dicts:

- `wall = false` (default): no conducting wall.
- `wall = true`: a conformal wall at standoff `a`, which you must also set
  (e.g. `GPECSettings(..., wall=true, a=0.3)`) -- there's no sensible default
  standoff distance, so this errors with a reminder rather than silently
  picking one.
- `wall = Dict(...)`: advanced/full manual control for non-conformal shapes
  (`"aw"`, `"bw"`, `"cw"`, `"dw"`, `"tw"`, `"equal_arc_wall"`) -- used as-is,
  unchanged from the raw GPEC-toml convention; must already be complete
  (including `"a"` if the shape needs it).
- `kinetic_source`/`kinetic_factor`: default to `"fixed"`/`0.0` (no kinetic
  corrections); override either independently to enable them.
"""
Base.@kwdef mutable struct GPECSettings
    toroidal_mode_number::Int                           # toroidal mode number (nn_low = nn_high)
    wall::Union{Bool,Dict{String,<:Any}} = false   # false = nowall; true = conformal (needs `a`); Dict = advanced/manual
    a::Union{Real,Nothing} = nothing               # wall standoff, only used when wall=true
    kinetic_source::String = "fixed"               # "fixed" or "calculated"
    kinetic_factor::Real = 0.0
    parallel_threads::Int = 2
end

"""
    _resolve_wall(wall, a) -> Dict{String,<:Any}

Turns `GPECSettings`'s friendly `wall`/`a` options into the Dict `_wall_toml`
expects. See `GPECSettings`'s docstring for the `false`/`true`/`Dict` cases.
"""
function _resolve_wall(wall, a)
    wall === false && return Dict("shape" => "nowall")
    if wall === true
        a === nothing && error(
            "GPECSettings: wall=true requires the wall standoff `a` (plasma-boundary-to-wall " *
            "distance) to also be set, e.g. GPECSettings(..., wall=true, a=0.3)."
        )
        return Dict("shape" => "conformal", "a" => a)
    end
    wall isa Dict && return wall  # advanced/manual control, used as-is
    error("GPECSettings: `wall` must be `false` (no wall), `true` (conformal wall, requires `a`), " *
          "or a Dict for advanced shapes (got $(typeof(wall))).")
end

"""
    run_GPEC(gpec_settings::GPECSettings, equilibrium_filename::String, test_dir::String)

Runs GPEC on `equilibrium_filename` (the gEQDSK already staged in
`test_dir`, e.g. by `write_geqdsk_for_gpec`) using the settings in `gpec_settings`.
"""
function run_GPEC(gpec_settings::GPECSettings, equilibrium_filename::String, test_dir::String)
    mkpath(test_dir)
    wall = _resolve_wall(gpec_settings.wall, gpec_settings.a)
    kinetic = Dict("kinetic_source" => gpec_settings.kinetic_source, "kinetic_factor" => gpec_settings.kinetic_factor)
    toml_str = make_gpec_toml(equilibrium_filename, gpec_settings.toroidal_mode_number, wall, kinetic;
                               parallel_threads=gpec_settings.parallel_threads)
    write(joinpath(test_dir, "gpec.toml"), toml_str)

    GPEC_output = GeneralizedPerturbedEquilibrium.main([test_dir]);
    return GPEC_output
end

function read_output(output::NamedTuple)
    ffs = output.ffs
    vd = ffs.free_boundary
    vd === nothing && error("read_output: no free_boundary result -- was vac_flag=true in gpec.toml?")

    return(
        dW_total = real(vd.et[1]),
        dW_plasma = real(vd.ep[1]),
        dW_vacuum = real(vd.ev[1]),
        et_full = vd.et,
        q0 = ffs.equil.params.q0,
        qmax = something(ffs.equil.params.qmax, ffs.qlim),
        msing = length(ffs.surfaces))
end

"""
    make_gpec_toml(equilibrium_filename, toroidal_mode_number, wall, kinetic; parallel_threads=2)

Return the full gpec.toml as a string, with inline comments preserved.

# Arguments
- `equilibrium_filename` — basename of the gEQDSK staged in the run directory.
- `toroidal_mode_number`       — toroidal mode number (nn_low = nn_high = toroidal_mode_number).
- `wall`        — must contain `"shape"`, plus shape-specific params.
- `kinetic`     — `"kinetic_source"`, `"kinetic_factor"`.

All non-obvious keys are annotated; defaults match shaped_beta_scan
and the DIII-D 147131 settings called out in CTM-processing/CONVENTIONS.md.

No `output_h5`/`HDF5_filename` here — HDF5 output is disabled entirely
(`write_outputs_to_HDF5 = false`) since results come back directly from
`GeneralizedPerturbedEquilibrium.main([test_dir])`'s return value now.
"""
function make_gpec_toml(equilibrium_filename, toroidal_mode_number, wall, kinetic; parallel_threads::Integer=2) # eventually get rid of file writing in favor of direct calls to GPEC functions
    wall_block      = _wall_toml(wall) # wall/no wall also exposed to the user 
    n_tor_int       = Int(toroidal_mode_number) # the toroidal mode number should be exposed to the user 
    kinetic_source  = kinetic["kinetic_source"]
    kinetic_factor  = Float64(kinetic["kinetic_factor"])
    pthreads        = Int(parallel_threads)

    return """# Generated by TokToxStability
# Settings audited against shaped_beta_scan/inputs/julia/gpec.toml and
# CTM-processing/CONVENTIONS.md (DIII-D 147131 reference settings).

[Equilibrium]
eq_filename = "$equilibrium_filename"
eq_type     = "efit"          # gEQDSK reader (vs. "chease", "tj_analytic", ...)
jac_type    = "hamada"        # Hamada (straight-field-line) coordinates
# NOTE: power_bp / power_b / power_r were REMOVED (deprecated) on develop.
# EquilibriumConfig's inner constructor now derives them from `jac_type`, and
# jac_type="hamada" forces power_* = 0 -- numerically identical to what this
# notebook used to set by hand, so this is a no-op beyond silencing the warning.
grid_type   = "ldp"           # log-density-power radial packing near rationals
psilow      = 1e-4            # inner ψ bound (avoid the axis singularity)
psihigh     = 0.995           # outer ψ bound (CONVENTIONS-recommended for D3D 147131)
mpsi        = 257             # # of radial grid points (Oak: bumped from auto)
mtheta      = 514             # # of poloidal grid points
newq0       = 0               # 0 = trust the input q0; >0 overrides
etol        = 1e-10           # equilibrium solver convergence tol
force_termination = false     # true = stop after equilibrium (skip stability)

$wall_block
[ForceFreeStates]
# --- Pipeline flags ---------------------------------------------------------
local_stability_flag = true   # Mercier D_I + ballooning (was mer_flag/bal_flag)
mat_flag = true               # save F, G, K matrices for diagnostics
ode_flag = true               # integrate Newcomb's eqn (REQUIRED for GPEC)
vac_flag = true               # compute free-boundary energies (populates FreeBoundaryStability/)
write_outputs_to_HDF5 = false # results are consumed directly from main_from_inputs's return value

# --- Edge / rational-surface truncation -------------------------------------
set_psilim_via_dmlim = true
dmlim                = 0.2
qhigh                = 1e3    # effectively no q-cap; dmlim governs psilim
qlow                 = 1.02   # skip the q ≈ 1 region (internal kink)
sing_start           = 0      # start integration at the sing_start'th rational
psiedge              = 0.99   # edge-dW diagnostic band lower bound

# --- Toroidal mode number ---------------------------------------------------
nn_low      = $n_tor_int
nn_high     = $n_tor_int
delta_mlow  = 8               # extend Fourier basis 8 harmonics below resonant m
delta_mhigh = 8               # ... and 8 above

# --- Numerical tolerances / integrator --------------------------------------
mthvac                   = 960     # # of pts in plasma-vacuum spline (≫ mtheta)
eulerlagrange_tolerance  = 1e-10   # ODE integrator relative tol
singfac_min              = 1e-4    # fractional distance from rational where ideal jump enforced
ucrit                    = 1e4     # max basis growth before re-normalisation
sing_order               = 6       # order of Frobenius expansion at rationals (STRIDE convention)
save_interval            = 3       # keep every 3rd ODE step (always saves near rationals)

# --- Concurrency / Δ' BVP ---------------------------------------------------
use_parallel             = false
parallel_threads         = $pthreads       # ignored while use_parallel = false; applies to the Δ' BVP path

# --- Kinetic terms ----------------------------------------------------------
kinetic_source = "$kinetic_source"
kinetic_factor = $kinetic_factor
"""
end

"""
    _wall_toml(wall)

Emit the [Wall] section. Only the keys relevant for the chosen `shape`
are written, to avoid drowning the file in stale defaults.
"""
function _wall_toml(wall)
    if wall["shape"] == "nowall"
        return "[Wall]\nshape = \"nowall\"\n"
    end

    lines = ["[Wall]",
              "shape = \"$(wall["shape"])\"",
              "a = $(wall["a"])    # plasma-boundary-to-wall standoff"]

    for k in ("aw", "bw", "cw", "dw", "tw")
        if haskey(wall, k)
            push!(lines, "$k = $(wall[k])")
        end
    end

    if get(wall, "equal_arc_wall", false)
        push!(lines, "equal_arc_wall = true")
    end

    return join(lines, "\n") * "\n"
end