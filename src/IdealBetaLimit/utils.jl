using PythonCall 
using Logging


const boltzmann_constant = 1.6022e-19

# just for testing, get rid of this later #
const _mygs          = Ref{Py}() 
const _psi_tkmkr     = Ref{Vector{Float64}}()
const _jphi_baseline = Ref{Vector{Float64}}()
const _ip_target     = Ref{Float64}()
###########################################

mutable struct Profiles 
    Ti::Union{Vector{Float64},Nothing}
    Te::Union{Vector{Float64},Nothing}
    ni::Union{Vector{Float64},Nothing}
    ne::Union{Vector{Float64},Nothing}
    Psi::Union{Vector{Float64},Nothing}
    pressure::Union{Vector{Float64},Nothing}
    pprime::Union{Vector{Float64},Nothing}
end

function read_profiles_from_file(path::String)
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
        col("psi"),
        pressure,
        nothing
    )
end

function rescale_profile!(profiles::Profiles, frac::Real) # ! because this is editing in place 
    profiles.Te .*= frac 
    profiles.Ti .*= frac 
    profiles.pressure = boltzmann_constant .* (profiles.ne .* profiles.Te .+ profiles.ni .* profiles.Ti)
    return profiles
end

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