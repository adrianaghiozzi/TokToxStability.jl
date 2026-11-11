using PythonCall 
using Logging

const boltzmann_constant = 1.6022e-19

Base.@kwdef mutable struct Profiles
    Ti::Union{Vector{Float64},Nothing} = nothing
    Te::Union{Vector{Float64},Nothing} = nothing
    ni::Union{Vector{Float64},Nothing} = nothing
    ne::Union{Vector{Float64},Nothing} = nothing
    Psi::Union{Vector{Float64},Nothing} = nothing
    omega::Union{Vector{Float64},Nothing} = nothing  # E x B rotation [rad/s], tearing/Glasser workflow only
end

Base.@kwdef mutable struct Equilibrium
    Psi::Union{Vector{Float64},Nothing} = nothing
    Jphi::Union{Vector{Float64},Nothing} = nothing
    Ip_target::Union{Float64,Nothing} = nothing
    rcentr::Union{Float64,Nothing} = nothing
    bcentr::Union{Float64,Nothing} = nothing
    ip::Union{Float64,Nothing} = nothing
    pax::Union{Float64,Nothing} = nothing
    rzout::Union{Matrix{Float64},Nothing} = nothing
    ffprim::Union{Vector{Float64},Nothing} = nothing
    pprime::Union{Vector{Float64},Nothing} = nothing
end

pressure_profile(profiles::Profiles) = boltzmann_constant .* (profiles.ne .* profiles.Te .+ profiles.ni .* profiles.Ti)

function read_kinetic_profiles_from_file(path::String)
    data, header = readdlm(path, header=true)
    cols = Dict(strip(String(h)) => i for (i, h) in enumerate(header))

    col(name) = Vector{Float64}(data[:, cols[name]])

    return Profiles(
        Ti = col("ti(eV)"),
        Te = col("te(eV)"),
        ni = col("ni(m^-3)"),
        ne = col("ne(m^-3)"),
        Psi = col("psi"),
        omega = haskey(cols, "wexb(rad/s)") ? col("wexb(rad/s)") : nothing,
    )
end

function read_equilibrium_from_file(path::String)
    read_eqdsk = pyimport("OpenFUSIONToolkit.TokaMaker.util").read_eqdsk
    eqdsk = read_eqdsk(path)

    ffprim = pyconvert(Vector{Float64}, eqdsk["ffprim"])
    pprime = pyconvert(Vector{Float64}, eqdsk["pprime"])

    return Equilibrium(
        Psi = collect(range(0.0, 1.0; length=length(ffprim))),
        Jphi = nothing,
        Ip_target = nothing,
        rcentr = pyconvert(Float64, eqdsk["rcentr"]),
        bcentr = pyconvert(Float64, eqdsk["bcentr"]),
        ip     = pyconvert(Float64, eqdsk["ip"]),
        pax    = pyconvert(Float64, eqdsk["pres"][0]),
        rzout  = pyconvert(Matrix{Float64}, eqdsk["rzout"]),
        ffprim = ffprim,
        pprime = pprime,
    )
end

function rescale_profile!(profiles::Profiles, frac::Real) # ! because this is editing in place
    profiles.Te .*= frac
    profiles.Ti .*= frac
    return profiles
end

function mtanh_pedestal_profile(rho::AbstractVector{<:Real}, ped_top::Real, foot::Real,
                                 core_slope::Real, knee::Real, hwid::Real, blend::Real)
    t_inner = tanh((knee - rho[1]) / hwid)
    t_foot = tanh((knee - 1.0) / hwid)
    A = (ped_top - foot) / (t_inner - t_foot)
    B = foot - A * t_foot
    ped = @. A * tanh((knee - rho) / hwid) + B
    core_line = @. ped_top + core_slope * (rho - rho[1])
    w = @. 1.0 / (1.0 + exp(-(rho - knee) / blend))
    return @. (1.0 - w) * core_line + w * ped
end

function numpy_like_gradient(f::AbstractVector{<:Real}) # replace this later with native Julia gradient 
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