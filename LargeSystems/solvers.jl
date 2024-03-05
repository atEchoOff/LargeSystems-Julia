include("system.jl")
include("solution.jl")
using SparseArrays
using LinearAlgebra

# Stopping conditions to be passed in
@enum Tolerance begin
    RELATIVE_RESIDUAL = 0
    RESIDUAL = 1
    ZERO = 2 # zero tolerance, just go maximum iterations
end

abstract type Metadata end # Save data per iteration to a metadata type
# Metadata should have !save(metadata; **kwargs) to determine what to do with data at each frame
abstract type Solver end

struct SimpleSolver <: Solver
    # Solve system directly using built-in solver
    sparse::Bool

    function SimpleSolver(;sparse::Bool=false)
        return new(sparse)
    end
end

function solve(solver::SimpleSolver, system::System)
    if solver.sparse
        x = sparse(system.A) \ system.b
    else
        x = system.A \ system.b
    end

    return Solution(x, system.var_idxs)
end