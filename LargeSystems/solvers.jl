include("system.jl")
include("solution.jl")
using SparseArrays
using LinearAlgebra

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

struct GradientMethodSolver
    # Solve Ax = b using the gradient method
    sparse::Bool
    x0::Union{Vector, Nothing}
    abs_tol::Float64
    maxit::Integer

    function GradientMethodSolver(;sparse::Bool=false, x0::Union{Vector, Nothing}=nothing, abs_tol::Float64=1e-7, maxit::Integer=100)
        # Initialize a gradient method solver
        # If sparse, convert matrices to sparse before computation
        # x0 is the default starting vector, defaults to 0
        # abs_tol is the absolute tolerance of the resulting residual
        # maxit is the maximum number of iterations
        return new(sparse, x0, abs_tol, maxit)
    end
end

function solve(solver::GradientMethodSolver, system::System)
    # Solve using the gradient method with exact step size
    # Return solution x, list of residuals, and last iteration
    # Print warnings if necessary
    A = system.A
    b = system.b
    x = deepcopy(solver.x0)

    if solver.sparse
        A = sparse(A)
    end
    
    r = Dict{Integer, Vector{Float64}}() # We use a dictionary for better visual indexing

    if isnothing(x)
        # Shorthand for use the 0 vector
        x = zeros(Float64, size(A, 2))
    end

    # The algorithm is followed exactly from page 183 of the notes
    r[0] = b - A * x
    k = 0

    # Stop if and only if:
    #   We found residual r so that r.T * A * r <= 0, so A is not SPD
    #   k >= maxiter
    #   norm(r) <= tol
    while (r[k]'A*r[k] > 0) && (k < solver.maxit) && (norm(r[k]) > solver.abs_tol)
        a = norm(r[k])^2 / (r[k]'A*r[k])
        x += a * r[k]
        r[k + 1] = r[k] - a * A * r[k]
        k += 1
    end
    
    if r[k]'A*r[k] <= 0
        # A is not SPD. 
        @warn "Matrix A was not Symmetric Positive Definite"
        return x, get.(Ref(r), range(0, k), missing), k
    elseif k >= solver.maxit
        # We exceeded the number of iterations
        @warn "The maximum number of iterations was exceeded"
        return x, get.(Ref(r), range(0, k), missing), k
    end
    
    # We are good!
    return x, get.(Ref(r), range(0, k), missing), k
end

struct ConjugateGradientMethodSolver
    sparse::Bool
    x0::Union{Vector, Nothing}
    abs_tol::Float64
    maxit::Integer

    function ConjugateGradientMethodSolver(;sparse::Bool=False, x0::Union{Vector,Nothing}=nothing, abs_tol::Float64=1e-7, maxit::Integer=100)
        # Initialize a gradient method solver
        # If sparse, convert matrices to sparse before computation
        # x0 is the default starting vector, defaults to 0
        # abs_tol is the absolute tolerance of the resulting residual
        # maxit is the maximum number of iterations
        return new(sparse, x0, abs_tol, maxit)
    end
end


function solve(solver::ConjugateGradientMethodSolver, system::System)
    # Solve using the gradient method with exact step size
    # Return solution x, list of residuals, and last iteration
    # Print warnings if necessary
    A = system.A
    b = system.b
    x = deepcopy(solver.x0)
    infinity = typemax(1)

    if solver.sparse
        A = sparse(A)
    end
    
    r = Dict{Integer, Vector{Float64}}() # We use a dictionary for better visual indexing

    if isnothing(x)
        # Shorthand for use the 0 vector
        x = zeros(Float64, size(A, 2))
    end

    # The algorithm is followed exactly from page 196 of the notes
    p = r[0] = b - A * x
    
    for k in 0:infinity # loop through the natural numbers
        if norm(r[k]) <= solver.abs_tol
            # We are below the tolerance, nice!
            return x, r, k
        elseif p'A*p <= 0
            # A is not positive definite :(
            @warn "Matrix A was not Symmetric Positive Definite"
            return x, r, k
        elseif k >= solver.maxit
            # Too many iterations :(
            @warn "The maximum number of iterations was exceeded"
            return x, r, k
        end
        
        a = r[k]'p / (p'A*p)
        x += a * p
        r[k + 1] = r[k] - a * A * p
        b = norm(r[k + 1])^2 / norm(r[k])^2
        p = r[k + 1] + b * p
    end
end