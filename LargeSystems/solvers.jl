include("system.jl")
include("solution.jl")

struct SimpleSolver
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