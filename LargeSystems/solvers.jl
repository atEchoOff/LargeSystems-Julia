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

struct GradientMethodSolver <: Solver
    # Solve Ax = b using the gradient method
    sparse::Bool
    x0::Union{Vector, Nothing}
    tol_type::Tolerance
    tol::Float64
    maxit::Integer

    function GradientMethodSolver(;sparse::Bool=false, x0::Union{Vector, Nothing}=nothing, tol_type::Tolerance=RESIDUAL, tol::Float64=1e-7, maxit::Integer=100)
        # Initialize a gradient method solver
        # If sparse, convert matrices to sparse before computation
        # x0 is the default starting vector, defaults to 0
        # tol_type and tol determine the tolerance
        # maxit is the maximum number of iterations
        return new(sparse, x0, tol_type, tol, maxit)
    end
end

function solve(solver::GradientMethodSolver, system::System; metadata::Union{Nothing, Metadata}=nothing)
    # Solve using the gradient method with exact step size
    # Return solution x
    # Return metadata through passed in metadata object
    A = system.A
    b = system.b
    x = deepcopy(solver.x0)

    if solver.sparse
        A = sparse(A)
    end

    if isnothing(x)
        # Shorthand for use the 0 vector
        x = zeros(Float64, size(A, 2))
    end

    # The algorithm is followed exactly from page 183 of the notes
    r = b - A * x
    k = 0

    # Stop if and only if:
    #   We found residual r so that r.T * A * r <= 0, so A is not SPD
    #   k >= maxiter
    #   Matched tolerance
    while (r'A*r > 0) && (k < solver.maxit)
        if !isnothing(metadata)
            save!(metadata, x=x, r=r)
        end

        if solver.tol_type == RESIDUAL && (norm(r) <= solver.tol)
            break
        elseif solver.tol_type == RELATIVE_RESIDUAL && (norm(r) / norm(b) <= solver.tol)
            break
        end

        a = norm(r)^2 / (r'A*r)
        x += a * r
        r -= a * A * r
        k += 1
    end
    
    if !iszero(r) && r'A*r <= 0
        # A is not SPD. 
        @warn "Matrix A was not Symmetric Positive Definite"
        return Solution(x, system.var_idxs)
    elseif k >= solver.maxit
        # We exceeded the number of iterations
        @warn "The maximum number of iterations was exceeded"
        return Solution(x, system.var_idxs)
    end
    
    # We are good!
    return Solution(x, system.var_idxs)
end

struct ConjugateGradientMethodSolver <: Solver
    sparse::Bool
    x0::Union{Vector, Nothing}
    tol_type::Tolerance
    tol::Float64
    maxit::Integer

    function ConjugateGradientMethodSolver(;sparse::Bool=false, x0::Union{Vector,Nothing}=nothing, tol_type::Tolerance=RESIDUAL, tol::Float64=1e-7, maxit::Integer=100)
        # Initialize a conjugate gradient method solver
        # If sparse, convert matrices to sparse before computation
        # x0 is the default starting vector, defaults to 0
        # tol_type and tol determine the tolerance
        # maxit is the maximum number of iterations
        return new(sparse, x0, tol_type, tol, maxit)
    end
end


function solve(solver::ConjugateGradientMethodSolver, system::System; metadata::Union{Nothing, Metadata}=nothing)
    # Solve using conjugate gradient method
    # Return solution x
    # Return metadata through optional metadata object
    A = system.A
    b = system.b
    x = deepcopy(solver.x0)
    infinity = typemax(1)

    if solver.sparse
        A = sparse(A)
    end

    if isnothing(x)
        # Shorthand for use the 0 vector
        x = zeros(Float64, size(A, 2))
    end

    # The algorithm is followed exactly from page 196 of the notes
    p = r = b - A * x
    
    for k in 0:infinity # loop through the natural numbers
        if !isnothing(metadata)
            save!(metadata, x=x, r=r, p=p)
        end

        if solver.tol_type == RESIDUAL && norm(r) <= solver.tol
            # We are below the tolerance, nice!
            return Solution(x, system.var_idxs)
        elseif solver.tol_type == RELATIVE_RESIDUAL && norm(r) / norm(b) <= solver.tol
            # We are below the tolerance, nice!
            return Solution(x, system.var_idxs)
        elseif p'A*p <= 0
            # A is not positive definite :(
            @warn "Matrix A was not Symmetric Positive Definite"
            return Solution(x, system.var_idxs)
        elseif k >= solver.maxit
            # Too many iterations :(
            @warn "The maximum number of iterations was exceeded"
            return Solution(x, system.var_idxs)
        end
        
        a = r'p / (p'A*p)
        x += a * p

        new_r = r - a * A * p
        b = norm(new_r)^2 / norm(r)^2
        r = new_r

        p = r + b * p
    end
end

struct JacobiMethodSolver <: Solver
    # Solve a system using the Jacobi method
    sparse::Bool
    x0::Union{Vector, Nothing}
    tol_type::Tolerance
    tol::Float64
    maxit::Integer

    function JacobiMethodSolver(;sparse::Bool=false, x0::Union{Vector,Nothing}=nothing, tol_type::Tolerance=RESIDUAL, tol::Float64=1e-7, maxit::Integer=100)
        # Initialize a jacobi method solver
        # If sparse, convert matrices to sparse before computation
        # x0 is the default starting vector, defaults to 0
        # tol_type and tol determine the tolerance
        # maxit is the maximum number of iterations
        return new(sparse, x0, tol_type, tol, maxit)
    end
end

function solve(solver::JacobiMethodSolver, system::System; metadata::Union{Nothing, Metadata}=nothing)
    # Solve using jacobi method. 
    # Return solution x and add metadata to optional metadata argument
    A = system.A
    b = system.b
    x = solver.x0

    if solver.sparse
        A = sparse(A)
    end

    D = Diagonal(A)
    DmA = D - A # Precalculate this so we dont need it later

    if isnothing(x)
        # Shorthand for use the 0 vector
        x = zeros(Float64, size(A, 2))
    end

    # Begin iteration
    infinity = typemax(1)
    for k in 0:infinity
        r = b - A * x
        if !isnothing(metadata)
            save!(metadata, x=x, r=r)
        end

        if solver.tol_type == RESIDUAL && norm(r) <= solver.tol
            # We are done!
            return Solution(x, system.var_idxs)
        elseif solver.tol_type == RELATIVE_RESIDUAL && norm(r) / norm(b) <= solver.tol
            # We are done!
            return Solution(x, system.var_idxs)
        elseif k >= solver.maxit
            # We took too many iterations
            @warn "The maximum number of iterations was exceeded"
            return Solution(x, system.var_idxs)
        end

        x = D \ (DmA * x + b) # D is diagonal, so this is fast
    end
end

struct GaussSeidelSolver <: Solver
    # Solve a system using the Gauss-Seidel
    sparse::Bool
    x0::Union{Vector, Nothing}
    tol_type::Tolerance
    tol::Float64
    maxit::Integer

    function GaussSeidelSolver(;sparse::Bool=false, x0::Union{Vector,Nothing}=nothing, tol_type::Tolerance=RESIDUAL, tol::Float64=1e-7, maxit::Integer=100)
        # Initialize a Gauss-Seidel solver
        # If sparse, convert matrices to sparse before computation
        # x0 is the default starting vector, defaults to 0
        # tol_type and tol determine the tolerance
        # maxit is the maximum number of iterations
        return new(sparse, x0, tol_type, tol, maxit)
    end
end

function solve(solver::GaussSeidelSolver, system::System; metadata::Union{Nothing, Metadata}=nothing)
    # Solve a system using the Gauss-Seidel method.
    # Return solution x and pass in metadata to metadata argument
    A = system.A
    b = system.b
    x = solver.x0

    if solver.sparse
        A = sparse(A)
    end

    D = Diagonal(A)
    mU = D - UpperTriangular(A) # (negative) the strictly upper triangular portion of A
    DpL = LowerTriangular(A) # The lower triangular portion of A

    if solver.sparse
        # Julia likes to kick out sparse matrices
        mU = sparse(mU)
    end

    if isnothing(x)
        # Shorthand for use the 0 vector
        x = zeros(Float64, size(A, 2))
    end

    # Begin iteration
    infinity = typemax(1)
    for k in 0:infinity
        r = b - A * x
        if !isnothing(metadata)
            save!(metadata, x=x, r=r)
        end

        if solver.tol_type == RESIDUAL && norm(r) <= solver.tol
            # We are done!
            return Solution(x, system.var_idxs)
        elseif solver.tol_type == RELATIVE_RESIDUAL && norm(r) / norm(b) <= solver.tol
            # We are done!
            return Solution(x, system.var_idxs)
        elseif k >= solver.maxit
            # We took too many iterations
            @warn "The maximum number of iterations was exceeded"
            return Solution(x, system.var_idxs)
        end

        x = DpL \ (mU * x + b) # DmL is lower triangular, so this is fast
    end
end

struct SuccessiveOverrelaxationSolver <: Solver
    # Solve a system using Successive Overrelaxation
    sparse::Bool
    x0::Union{Vector, Nothing}
    tol_type::Tolerance
    tol::Float64
    maxit::Integer
    ω::Float64

    function SuccessiveOverrelaxationSolver(;sparse::Bool=false, x0::Union{Vector,Nothing}=nothing, tol_type::Tolerance=RESIDUAL, tol::Float64=1e-7, maxit::Integer=100, ω::Float64=2.0)
        # Initialize a SOR solver
        # If sparse, convert matrices to sparse before computation
        # x0 is the default starting vector, defaults to 0
        # tol_type and tol determine the tolerance
        # maxit is the maximum number of iterations
        # ω is the SOR hyperparameter
        return new(sparse, x0, tol_type, tol, maxit, ω)
    end
end

function solve(solver::SuccessiveOverrelaxationSolver, system::System; metadata::Union{Nothing, Metadata}=nothing)
    # Solve a system with SOR
    # Return solution x and pass in metadata to optional metadata argument
    A = system.A
    b = system.b
    x = solver.x0

    if solver.sparse
        A = sparse(A)
    end

    D = Diagonal(A)
    L = A - UpperTriangular(A)
    U = A - LowerTriangular(A)
    DpOL = 1 / solver.ω * (D + solver.ω * L) # Save these for quicker computation
    DmOU = 1 / solver.ω * ((1 - solver.ω) * D - solver.ω * U)

    if solver.sparse
        # Julia likes to kick out sparse matrices
        L = sparse(L)
        U = sparse(U)
        DpOL = sparse(DpOL)
        DmOU = sparse(DmOU)
    end

    if isnothing(x)
        # Shorthand for use the 0 vector
        x = zeros(Float64, size(A, 2))
    end

    # Begin iteration
    infinity = typemax(1)
    for k in 0:infinity
        r = b - A * x
        if !isnothing(metadata)
            save!(metadata, x=x, r=r)
        end

        if solver.tol_type == RESIDUAL && norm(r) <= solver.tol
            # We are done!
            return Solution(x, system.var_idxs)
        elseif solver.tol_type == RELATIVE_RESIDUAL && norm(r) / norm(b) <= solver.tol
            # We are done!
            return Solution(x, system.var_idxs)
        elseif k >= solver.maxit
            # We took too many iterations
            @warn "The maximum number of iterations was exceeded"
            return Solution(x, system.var_idxs)
        end

        x = DpOL \ (DmOU * x + b) # DpOL is lower triangular, so this is fast
    end
end