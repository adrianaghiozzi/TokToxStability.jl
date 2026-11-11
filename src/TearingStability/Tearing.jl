"""
    load_default_settings_tearing(toroidal_mode_number; wall=false, a=nothing,
                                   chi_perp=1.0, zeff=1.5, dc_type=:fitzpatrick, mu_i=2.0, kwargs...)

`GPECSettings` set up the way the tearing-stability workflow needs them: ideal-MHD
(`kinetic_factor=0.0`, matching Leuthold et al.'s Delta_eff = Delta' - Delta_crit split, since the
Glasser/chi_parallel-matching correction is applied on top of the classical ideal-MHD Delta'), plus
the inner-layer inputs (`chi_perp`, `zeff`, `dc_type`, `mu_i`) `compute_glasser_terms` needs. Extra
`kwargs` are forwarded to `GPECSettings`.
"""
function load_default_settings_tearing(toroidal_mode_number::Int;
                                        wall::Union{Bool,Dict{String,<:Any}}=false,
                                        a::Union{Real,Nothing}=nothing,
                                        chi_perp::Real=1.0, zeff::Real=1.5,
                                        dc_type::Symbol=:fitzpatrick, mu_i::Real=2.0, kwargs...)
    return GPECSettings(; toroidal_mode_number, wall, a, kinetic_factor=0.0,
                         chi_perp, zeff, dc_type, mu_i, kwargs...)
end

"""
    delta_prime_for_mode(result, m::Int, n::Int)

Classical linear tearing Δ' [1/m] for the rational surface resonant with poloidal/toroidal
mode numbers `(m, n)` (e.g. `(2, 1)` for the 2/1 mode), picked out of `result.delta_prime_matrix`
using `result.singular_surfaces` to find the matching row/column. `result` can be the NamedTuple
returned by `read_output` or a per-fraction row from a scan (anything exposing those two fields).

Returns `missing` if GPEC found no Δ' matrix (e.g. a kinetic run, or no rational surfaces), or if
`(m, n)` doesn't resonate with any surface in this equilibrium (common at low pressure/current,
before that surface has entered the plasma).
"""
function delta_prime_for_mode(result, m::Int, n::Int)
    result.delta_prime_matrix === nothing && return missing
    surfaces = result.singular_surfaces
    if size(result.delta_prime_matrix, 1) != length(surfaces)
        @warn "delta_prime_for_mode: delta_prime_matrix is $(size(result.delta_prime_matrix, 1))x... but " *
              "there are $(length(surfaces)) singular_surfaces -- GPEC excluded some surfaces from the " *
              "Δ' BVP (see compute_delta_prime_matrix! debug log), so the row/column mapping used here " *
              "is unreliable. Inspect result.singular_surfaces / the GPEC log directly."
        return missing
    end
    i = findfirst(s -> m in s.m && n in s.n, surfaces)
    i === nothing && return missing
    return real(result.delta_prime_matrix[i, i])
end

"""
    compute_glasser_terms(gpec_raw, profiles::Profiles, gpec_settings::GPECSettings)

Glasser/chi_parallel-matching stabilization inputs for every rational surface GPEC found in this
run, via GPEC's own `InnerLayer.SLAYER.build_slayer_inputs` (Leuthold et al., "ARC physics basis --
MHD", Sec 4.2.1, Eqs 4.1-4.5): `Delta_eff = Delta' - Delta_crit`, where `Delta_crit` is the
Lutjens/Fitzpatrick chi_par-matching critical-Delta formula (`gpec_settings.dc_type`), auto-deriving
`D_R = E + F + H^2` from the same GGJ coefficients GPEC's local-stability pass already computes, and
reading the minor radius/shear at each surface from the actual equilibrium geometry.

`gpec_raw` is the raw NamedTuple returned by `run_GPEC` (i.e. `output.ffs` must still be alive --
this must be called before that object is discarded, unlike `delta_prime_for_mode` which only needs
the lightweight `read_output` summary). `profiles.omega` (E x B rotation, [rad/s]) must be set --
only `read_kinetic_profiles_from_file` files with a `wexb(rad/s)` column populate it.

Returns a `Vector` of `(m, n, delta_crit, D_R, rs, R0, sval_r, eta, tau_r)` NamedTuples, one per
singular surface, meant to be stored on a scan row (see `glasser_terms` in `evaluate_factor`) and
read out per-mode via `delta_crit_for_mode`/`delta_eff_for_mode`.
"""
function compute_glasser_terms(gpec_raw, profiles::Profiles, gpec_settings::GPECSettings)
    profiles.omega === nothing && error(
        "compute_glasser_terms: profiles.omega (E x B rotation) is not set -- read the kinetic " *
        "profiles from a .kin file with a wexb(rad/s) column (read_kinetic_profiles_from_file " *
        "picks it up automatically), or set profiles.omega by hand."
    )
    ffs = gpec_raw.ffs
    isempty(ffs.surfaces) && return NamedTuple[]

    # GPEC v2's KineticProfiles carries only n_e, T_e, T_i and the E x B rotation; build_slayer_inputs
    # derives the diamagnetic frequencies itself from the density/temperature splines.
    kinetic_profiles = GeneralizedPerturbedEquilibrium.Utilities.KineticProfiles(;
        psi=profiles.Psi, n_e=profiles.ne, T_e=profiles.Te, T_i=profiles.Ti, omega_E=profiles.omega)

    params = GeneralizedPerturbedEquilibrium.InnerLayer.SLAYER.build_slayer_inputs(
        ffs.equil, ffs.surfaces, kinetic_profiles;
        mu_i=gpec_settings.mu_i, zeff=gpec_settings.zeff,
        # One chi_perp drives every diffusivity. Delta_crit (dc_tmp) only uses chi_perp_e; chi_perp_i
        # is required by GPEC v2's default P_perp_model = :D_perp, which we don't read out.
        chi_perp_e=gpec_settings.chi_perp, chi_perp_i=gpec_settings.chi_perp, chi_tor=gpec_settings.chi_perp,
        dc_type=gpec_settings.dc_type)

    return [(m=p.m, n=p.n, delta_crit=p.dc_tmp, D_R=p.dr_val, rs=p.rs, R0=p.R0,
             sval_r=p.sval_r, eta=p.eta, tau_r=p.tau_r) for p in params]
end

"""
    delta_crit_for_mode(result, m::Int, n::Int)

Critical-Delta offset (chi_parallel matching, see `compute_glasser_terms`) for the `(m, n)` mode,
picked out of `result.glasser_terms`. Returns `missing` if `glasser_terms` is `nothing` (no
`profiles.omega` was available when this row was scanned) or has no entry for `(m, n)`.
"""
function delta_crit_for_mode(result, m::Int, n::Int)
    terms = result.glasser_terms
    (terms === nothing || isempty(terms)) && return missing
    i = findfirst(t -> t.m == m && t.n == n, terms)
    i === nothing && return missing
    return terms[i].delta_crit
end

"""
    delta_eff_for_mode(result, m::Int, n::Int)

`Delta_eff = Delta' - Delta_crit` for the `(m, n)` mode (Leuthold et al. Eq. 4.1) -- the
Glasser-stabilized effective tearing index. `Delta_eff > 0` is unstable; `Delta_eff < 0` is stable.
Returns `missing` if either `delta_prime_for_mode` or `delta_crit_for_mode` is `missing`.
"""
function delta_eff_for_mode(result, m::Int, n::Int)
    dp = delta_prime_for_mode(result, m, n)
    dc = delta_crit_for_mode(result, m, n)
    (ismissing(dp) || ismissing(dc)) && return missing
    return dp - dc
end
