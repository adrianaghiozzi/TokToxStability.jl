module TokToxStability
# general utility functions
include("utils.jl")
include("plots.jl")


# ideal beta limit 
include("IdealBetaLimit/DriveGPEC.jl")
include("IdealBetaLimit/DriveTokTox.jl")
include("IdealBetaLimit/DriveTokaMaker.jl")

# future channels here e.g. tearing stability
end
