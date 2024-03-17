include("LargeSystems/system.jl")
include("Utils/Layered.jl")
include("LargeSystems/solvers.jl")
using Polynomials

γ = 1
L = pi
T = 1
u0(x::Float64) = cos(2 * x)
u(x::Float64, t::Float64) = exp(-5t)cos(2 * x) # true solution

function build_system(M::Integer, N::Integer, θ::Float64)
    # Build the system over an M×N grid
    Δx = 2L / (M + 1)
    Δt = T / (N + 1)

    # Initialize the variables and the system
    U = ["U$j,$n" for j in range(1, M+1), n in range(1, N+1)]
    system = System(U)
    U = Layered{Tuple{Integer, Integer}, Union{Linear, Number}}([V(U[j,n]) for j in range(1, M+1), n in range(1, N+1)])

    # Initial conditions
    for j in range(0, M + 2)
        U[j, 0] = u0(-L + Δx * j)
    end

    # Initialize our boundary
    for n in range(1, N + 1)
        U[0, n] = U[M + 1, n]
        U[M + 2, n] = U[1, n]
    end

    # Now, we add our schema
    for n in range(0, N)
        for j in range(1, M+1)
            add_constraint!(system, U[j,n+1] == U[j,n] 
            + Δt/(2*Δx^2) * (U[j-1,n] - 2U[j,n] + U[j+1,n] + U[j-1,n+1] - 2U[j,n+1] + U[j+1,n+1]) 
            - Δt * γ * ((1 - θ)U[j,n] + θ*U[j,n+1]))
        end
    end

    return system
end

function evaluate_system(M::Integer, N::Integer, θ::Float64)
    # Build the system over an M×N grid, and then solve, and return error
    system = build_system(M, N, θ)
    Δx = 2L / (M + 1)
    Δt = T / (N + 1)

    solver = SimpleSolver(sparse=true)

    solution = solve(solver, system)
    U = [solution["U$j,$n"] for j in range(1, M + 1), n in range(1, N + 1)]
    true_U = [u(Δx * x - L, Δt * t) for x in range(1, M + 1), t in range(1, N + 1)]
    return sqrt(Δx * Δt) * norm(U - true_U, 2) # Frobenius norm for treating matrices as vectors
end

function estimate_numerical_convergence(Ns, errors)
    # Estimate the c and n for numerical convergence, based on a vector of Ns and errors
    hs = @. 1 / (1 + Ns)
    log_hs = log.(hs)
    log_errors = log.(errors)

    log_c, n = fit(log_hs, log_errors, 1)
    c = exp(log_c)
    return (;c, n)
end

function evaluate_Ns(θ::Float64, Ns...)
    # Evaluate the PDE for different Ns, get the L2 error, find the convergence rates and plot
    L2_errors = Float64[]
    for N in Ns
        push!(L2_errors, evaluate_system(N, N, θ))
    end

    println(L2_errors)

    _, n = estimate_numerical_convergence([Ns...], L2_errors)
    println("The estimated order of convergence was $n")
end

evaluate_Ns(0.5, 50, 60, 70, 80, 90, 100)