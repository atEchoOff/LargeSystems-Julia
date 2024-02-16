include("LargeSystems/solvers.jl")
using Plots
using LinearAlgebra
include("Utils/PDEs.jl")

# We will model the PDE -u''(x)=f(x)

f(x) = pi^2 * cos(pi * x)
u(x) = cos(pi * x) # Our true solution

system = build_1D_poisson(30, 1, -1, f)

mutable struct ResidualMetadata <: Metadata
    # A type that handles saving norms of iterands
    x_norms::Vector{Float64}

    function ResidualMetadata()
        return new(Float64[])
    end
end

function save!(metadata::Metadata; kwargs...)
    push!(metadata.x_norms, norm(kwargs[:x]))
end

metadata = ResidualMetadata()
solver = ConjugateGradientMethodSolver(sparse=true)

x = solve(solver, system, metadata=metadata)
x = [1; x.values; -1] # Add the endpoints to our solution

# Fun fact about the conjugate gradient method: If you start at x0=0, then the norms of the iterands are strictly increasing!
x_norms = Vector{Float64}
for (i, norm) in enumerate(metadata.x_norms)
    println("||x_{$i}|| \t = \t $norm")
end

domain = range(0, 1, length(x))
plot()
plot!(domain, u, label="True u")
plot!(domain, x, label="Estimated u")
plot!(title="Estimation of u via conjugate gradient descent")