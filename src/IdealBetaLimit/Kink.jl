"""
    load_default_settings_kink(toroidal_mode_number; wall=false, a=nothing, kwargs...)

`GPECSettings` set up the way the ideal (external kink) beta-limit workflow needs them:
ideal-MHD (`kinetic_factor=0.0`), free-boundary energies enabled via `run_GPEC`/`make_gpec_toml`.
Extra `kwargs` are forwarded to `GPECSettings` (e.g. `parallel_threads`).
"""
function load_default_settings_kink(toroidal_mode_number::Int;
                                     wall::Union{Bool,Dict{String,<:Any}}=false,
                                     a::Union{Real,Nothing}=nothing, kwargs...)
    return GPECSettings(; toroidal_mode_number, wall, a, kinetic_source="fixed", kinetic_factor=0.0, kwargs...)
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
