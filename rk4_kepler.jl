using Plots
using LaTeXStrings

function kepler_RHS(u::Vector{Float64})
    # RHS of kepler, modeled from sample matlab
    r = sqrt(u[1]^2 + u[2]^2)
    ret = zeros(Float64, size(u))
    ret[1] = u[3]
    ret[2] = u[4]
    ret[3] = -u[1]/r^3 - .015u[1]/r^5
    ret[4] = -u[2]/r^3 - .015u[2]/r^5

    return ret
end

function RK4_step!(u::Vector{Float64}, Δt::Float64)
    # Apply a single RK4 step to a vector u
    k1 = kepler_RHS(u)
    k2 = kepler_RHS(u + Δt * k1/2)
    k3 = kepler_RHS(u + Δt * k2/2)
    k4 = kepler_RHS(u + Δt * k3)

    u .+= Δt / 6 * (k1 + 2k2 + 2k3 + k4)
end

function RK4!(u::Vector{Float64}, Δt::Float64, T::Float64)
    # Apply RK4 given initial condition u
    # given Δt timestep
    # given T final time
    # Return vector of solutions
    ret = Vector{Float64}[]

    for _ in 0:Δt:T
        push!(ret, deepcopy(u))
        RK4_step!(u, Δt)
    end

    # Return solutions as matrix
    return permutedims(hcat(ret...))
end

function RK4(Δt::Float64)
    # Apply RK4 given constants given from homework
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

    # Run RK4
    return RK4!(u, Δt, T)
end