include("LargeSystems/solvers.jl")
using Plots; pyplot()
using Polynomials
include("Utils/PDEs.jl")
include("Utils/ShiftedList.jl")


f(x, y) = ((y - y^3)*(2pi)^2 + 6y) * cos(2pi * x) - 6y # out function f, solution to part a
Δf(x, y) = -(2pi)^2 * ((y - y^3) * (2pi)^2 + 12y) * cos(2pi * x) # the laplacian of f
u(x, y) = (y^3 - y) * (cos(2pi * x) - 1) # right hand side
boundary(x, y) = 0 # Our boundary value function

function evaluate_2D_poisson(N)
    # Evaluate the ODE, and then return the L2 error (modified for PDEs)
    h = 1 / (N + 1)
    system = build_2D_poisson(N, boundary, f) # the finite difference matrix and RHS
    solver = SimpleSolver(sparse=true)

    solution = solve(solver, system) # solve the system
    U = [solution["U$i,$j"] for i in range(1, N), j in range(1, N)] # obtain the solution
    true_U = [u(h * x, h * y) for x in range(1, N), y in range(1, N)]

    return h * norm(U - true_U, 2) # Frobenius norm for treating matrices as vectors
end

function evaluate_2D_poisson_h4(N)
    # Evaluate the ODE, and then return the L2 error (modified for PDEs)
    # The only difference between evaluate 2D_poisson and evaluate_2D_poisson_h4
    # is that this one passes in the laplacian of f to the system builder
    h = 1 / (N + 1)
    system = build_2D_poisson(N, boundary, f, Δf=Δf)
    solver = SimpleSolver(sparse=true)

    solution = solve(solver, system)
    U = [solution["U$i,$j"] for i in range(1, N), j in range(1, N)]
    true_U = [u(h * x, h * y) for x in range(1, N), y in range(1, N)]

    return h * norm(U - true_U, 2) # Frobenius norm for treating matrices as vectors
end

function estimateNumericalConvergence(Ns, errors)
    # Estimate the c and n for numerical convergence, based on a vector of hs and errors
    hs = @. 1 / (1 + Ns)
    log_hs = log.(hs)
    log_errors = log.(errors)

    log_c, n = fit(log_hs, log_errors, 1)
    c = exp(log_c)
    return (;c, n)
end

function evaluate_Ns(F::Function, Ns...)
    # Evaluate a function which takes in N, pass in a list of Ns, and estimate the numerical convergence rates
    L2_errors = Float64[]
    for N in Ns
        push!(L2_errors, F(N))
    end

    c, n = estimateNumericalConvergence([Ns...], L2_errors)
    println(c," ", n)
end

print("The statistics for 2D poisson without the Laplacian trick: ")
evaluate_Ns(evaluate_2D_poisson, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100)

print("The statistics for 2D poisson with the Laplacian trick: ")
evaluate_Ns(evaluate_2D_poisson_h4, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100)