using Plots
using PythonCall

"""
    plot_profiles(results; quantities=(:Te, :Ti, :ne, :pressure)) -> Plots.Plot

Plots the requested profile quantities (each vs. `psi`) for every entry in
`results` (the vector returned by `coarse_scan`/`coarse_scan_until_unstable`),
one panel per quantity, one line per fraction. Relies on the per-iteration
profile snapshot fields `evaluate_factor` attaches to every result row
(`psi`, `Te`, `Ti`, `ne`, `pressure`).
"""
function plot_profiles(results::AbstractVector; quantities=(:Te, :Ti, :ne, :pressure))
    sorted = sort(collect(results); by=r -> r.fraction)
    panels = [begin
        p = plot(xlabel="psi", ylabel=String(q), legend=:topright, leftmargin=10Plots.mm)
        for r in sorted
            plot!(p, r.psi, getfield(r, q); label="f=$(round(r.fraction; digits=3))")
        end
        p
    end for q in quantities]
    return plot(panels...; layout=(length(quantities), 1), size=(700, 300 * length(quantities)))
end

"""
    plot_q_profile(results) -> Plots.Plot

Plots the safety factor q(psi) for every entry in `results`, one line per
fraction, with a horizontal reference line at q=1 (the m/n=1/1 rational
surface). Relies on the `psi_q`/`q_profile` fields `evaluate_factor` attaches
to every result row (from `mygs.get_q()` at that fraction's converged
equilibrium).
"""
function plot_q_profile(results::AbstractVector)
    sorted = sort(collect(results); by=r -> r.fraction)
    p = plot(xlabel="psi", ylabel="q", title="Safety factor profile", legend=:topright, leftmargin=8Plots.mm)
    for r in sorted
        plot!(p, r.psi_q, r.q_profile; label="f=$(round(r.fraction; digits=3))")
    end
    hline!(p, [1.0]; label="q=1", linestyle=:dash, color=:black)
    return p
end

"""
    plot_equilibrium(result; nlevels=25) -> Plots.Plot

Plots flux-surface contours (psi(R,Z)) for a single `coarse_scan`/
`coarse_scan_until_unstable` result row, read back from the gEQDSK saved at
`result.geqdsk_path` (written by `write_geqdsk_for_gpec` during
`evaluate_factor`). Uses the `eqdsk` Python package -- already a dependency
of the TORAX/TokaMaker toolchain, and already used elsewhere in this package
for COCOS handling -- to parse the file, rather than reimplementing a gEQDSK
reader in Julia.
"""
function plot_equilibrium(result; nlevels::Int=25)
    eqdsk_mod = pyimport("eqdsk")
    inst = eqdsk_mod.EQDSKInterface.from_file(result.geqdsk_path; to_cocos=nothing, no_cocos=true)

    nx = pyconvert(Int, inst.nx)
    nz = pyconvert(Int, inst.nz)
    xgrid1 = pyconvert(Float64, inst.xgrid1)
    xdim = pyconvert(Float64, inst.xdim)
    zmid = pyconvert(Float64, inst.zmid)
    zdim = pyconvert(Float64, inst.zdim)

    R = range(xgrid1, xgrid1 + xdim; length=nx)
    Z = range(zmid - zdim / 2, zmid + zdim / 2; length=nz)
    psi = pyconvert(Matrix{Float64}, inst.psi)  # (nx, nz), R-major
    xbdry = pyconvert(Vector{Float64}, inst.xbdry)
    zbdry = pyconvert(Vector{Float64}, inst.zbdry)

    p = contour(R, Z, psi'; levels=nlevels, xlabel="R [m]", ylabel="Z [m]",
                title="Equilibrium (fraction=$(round(result.fraction; digits=3)))",
                aspect_ratio=:equal, colorbar=false)
    plot!(p, xbdry, zbdry; label="LCFS", color=:red, linewidth=2)
    return p
end

"""
    plot_beta_scan_summary(results; beta_limit=nothing) -> Plots.Plot

Summary plot of a `coarse_scan`/`coarse_scan_until_unstable` run: `dW_total`
vs. `fraction` (x-axis, as scanned), with a horizontal reference line at
`dW_total=0` (the ideal-MHD marginal-stability threshold). Each scanned point
is annotated with its `beta_N`, since that -- not `fraction` -- is the
physically meaningful quantity.

If `beta_limit` (the NamedTuple returned by `estimate_beta_limit`) is given,
the estimated crossing is also marked: a vertical line at the fraction
corresponding to `beta_limit.beta_n_crossing` (found via the same
bracket-or-extrapolate procedure `estimate_beta_limit` uses, applied to
`fraction` instead of `beta_n` -- see `_interp_zero_crossing`), solid if
`beta_limit.bracketed` (a real, calculated bound) or dashed if not (an
extrapolated guess), labeled with the corresponding beta_N value.
"""
function plot_beta_scan_summary(results::AbstractVector; beta_limit=nothing)
    sorted = sort(collect(results); by=r -> r.fraction)
    fraction = [r.fraction for r in sorted]
    dW = [r.dW_total for r in sorted]
    beta_n = [r.beta_n for r in sorted]

    p = plot(fraction, dW; seriestype=:scatter, markersize=6, label="scanned points",
             xlabel="fraction", ylabel="dW_total", title="Ideal-MHD beta limit scan",
             legend=:topright, leftmargin=8Plots.mm)
    plot!(p, fraction, dW; label=nothing, color=:steelblue, linewidth=1)
    for (f, w, b) in zip(fraction, dW, beta_n)
        annotate!(p, f, w, text("  beta_N=$(round(b; digits=2))", 8, :left))
    end
    hline!(p, [0.0]; label="dW_total = 0", color=:black, linestyle=:dot)

    if beta_limit !== nothing
        fraction_res = _interp_zero_crossing(fraction, dW)
        style = beta_limit.bracketed ? :solid : :dash
        label = (beta_limit.bracketed ? "crossing (bracketed): " : "crossing (extrapolated): ") *
                "beta_N=$(round(beta_limit.beta_n_crossing; digits=3))"
        vline!(p, [fraction_res.crossing]; label=label, color=:red, linestyle=style, linewidth=2)
    end

    return p
end
