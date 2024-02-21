using Plots
using LaTeXStrings
using FiniteDiff
using LinearAlgebra

function kepler_RHS(u::AbstractArray)
    # RHS of kepler, modeled from sample matlab
    r = sqrt(u[1]^2 + u[2]^2)
    ret = []
    push!(ret, u[3])
    push!(ret, u[4])
    push!(ret, -u[1]/r^3 - .015u[1]/r^5)
    push!(ret, -u[2]/r^3 - .015u[2]/r^5)

    return ret
end

function generate_F(u::Vector{Float64}, Δt::Float64)
    # Generate a function F for Newtons method
    return function F(y::AbstractArray)
        # Function F for rootfinding
        # Defined as kepler_RHS((y+u)/2) - (y-u)/Δt
        return kepler_RHS((y+u)/2) - (y-u)/Δt
    end
end

function newton_step!(F::Function, y::Vector{Float64})
    # Make one step in newton

    # Use FiniteDiff to compute the jacobian
    jac = FiniteDiff.finite_difference_jacobian(F, y)
    Fy = F(y)

    # Step in newton
    y .-= jac \ Fy
end

function midpoint_step!(u::Vector{Float64}, Δt::Float64; maxit::Integer=50)
    # Given uⁿ, use Newton's method to determine uⁿ⁺¹
    # Iterate maxit times

    # First create the function we are trying to root
    F = generate_F(deepcopy(u), Δt)

    # Now, run a bunch of newton iterations
    y = deepcopy(u)
    for _ in range(1, maxit)
        newton_step!(F, y)
    end

    # Save result in u
    u .= y
end

function midpoint!(u::Vector{Float64}, Δt::Float64, T::Float64)
    # Apply midpoint method given initial condition u
    # given Δt timestep
    # given T final time
    # Return Solutions over time
    ret = Vector{Float64}[]

    for _ in 0:Δt:T
        push!(ret, deepcopy(u))
        midpoint_step!(u, Δt)
    end

    return permutedims(hcat(ret...))
end

function midpoint(Δt::Float64)
    # Apply midpoint given constants given from homework
    # Given Δt
    # Return solutions over time
    β = .6
    q1 = 1 - β
    q2 = 0.0
    p1 = 0.0
    p2 = sqrt((1 + β) / (1 - β))
    T = 500.0

    # Initial condition
    u = Float64[q1, q2, p1, p2]

    # Run midpoint
    return midpoint!(u, Δt, T)
end