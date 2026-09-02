using Plots
using PythonCall

function plot_profiles(results::AbstractVector; quantities=(:Te, :Ti, :pressure))
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

function plot_q_profile(results::AbstractVector)
    sorted = sort(collect(results); by=r -> r.fraction)
    p = plot(xlabel="psi", ylabel="q", title="Safety factor profile", legend=:topright, leftmargin=8Plots.mm)
    for r in sorted
        plot!(p, r.psi_q, r.q_profile; label="f=$(round(r.fraction; digits=3))")
    end
    hline!(p, [1.0]; label="q=1", linestyle=:dash, color=:black)
    return p
end

function plot_equilibrium(result; nlevels::Int=25, python_site_packages::Union{Nothing,String}=nothing)
    result.geqdsk_path === nothing && error(
        "plot_equilibrium: this result has no saved gEQDSK to read (geqdsk_path === nothing) -- " *
        "it came from a scan run with working_directory=nothing (the default), which deletes each " *
        "fraction's files right after GPEC runs. Rerun the scan with an explicit `working_directory` " *
        "to keep the files this needs."
    )
    if python_site_packages !== nothing
        @pyexec (SITE_PACKAGES=python_site_packages,) => """
        import sys
        if SITE_PACKAGES not in sys.path:
            sys.path.insert(0, SITE_PACKAGES)
        """ => ()
    end
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

function plot_beta_scan_summary(results::AbstractVector; beta_limit=nothing)
    sorted = sort(collect(results); by=r -> r.fraction)
    fraction = [r.fraction for r in sorted]
    dW = [r.dW_total for r in sorted]
    beta_n = [r.beta_n for r in sorted]

    p = plot(fraction, dW; seriestype=:scatter, markersize=6, label="scanned points",
             xlabel="fraction", ylabel="dW_total", title="Ideal-MHD beta limit scan",
             legend=:bottomleft, leftmargin=8Plots.mm, rightmargin=20Plots.mm)
    plot!(p, fraction, dW; label=nothing, color=:steelblue, linewidth=1)
    for (f, w, b) in zip(fraction, dW, beta_n)
        annotate!(p, f, w, text("  β_N=$(round(b; digits=2))", 8, :left))
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
