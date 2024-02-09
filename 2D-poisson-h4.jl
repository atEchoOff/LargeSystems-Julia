include("LargeSystems/solvers.jl")
using Plots; pyplot()
using Polynomials
include("Utils/PDEs.jl")
include("Utils/ShiftedList.jl")

# We will model the PDE -u''(x)=f(x)+Δf(x), for better convergence

f(x, y) = ((y - y^3)*(2pi)^2 + 6y) * cos(2pi * x) - 6y
Δf(x, y) = -(2pi)^2 * ((y - y^3) * (2pi)^2 + 12y) * cos(2pi * x)

u(x, y) = (y^3 - y) * (cos(2pi * x) - 1)
boundary(x, y) = 0 # Our boundary value function

function evaluate_2D_poisson(N)
    # Evaluate the ODE, and then return the L2 error (modified for PDEs)
    h = 1 / (N + 1)
    system = build_2D_poisson(N, boundary, f, Δf=Δf)
    solver = SimpleSolver(sparse=true)

    solution = solve(solver, system)
    U = [solution["U$i,$j"] for i in range(1, N), j in range(1, N)]
    true_U = [u(h * x, h * y) for x in range(1, N), y in range(1, N)]

    return h * norm(U - true_U, 2) # Frobenius norm for treating matrices as vectors
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

function evaluate_Ns(Ns...)
    # Evaluate the PDE for different Ns, get the L2 error, find the convergence rates and plot
    L2_errors = Float64[]
    for N in Ns
        push!(L2_errors, evaluate_2D_poisson(N))
    end

    c, n = estimate_numerical_convergence([Ns...], L2_errors)
    println(c," ", n)
end

evaluate_Ns(10, 20, 30, 40, 50, 60, 70, 80, 90, 100)