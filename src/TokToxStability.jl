module TokToxStability
# shared utilities
include("utils.jl")
include("plots.jl")
include("GPEC.jl")
include("DriveTokaMaker.jl")
include("DriveTokTox.jl")

# ideal beta limit (external kink stability)
include("IdealBetaLimit/Kink.jl")

# tearing stability
include("TearingStability/Tearing.jl")
end
