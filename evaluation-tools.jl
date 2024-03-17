include("LargeSystems/solvers.jl")
using Polynomials
using Plots

function solve_system(build_system::Function, M::Integer, N::Integer, u0::Function, a::Float64, tf::Float64)
    # Build system and return solution in matrix form, including intial and boundary conditions
    system = build_system(M, N, u0, a, tf)
    solver = SimpleSolver(sparse=true)
    solution = solve(solver, system)

    inner_solution = [solution["U$j,$n"] for j in range(1, M + 1), n in range(1, N + 1)]

    # Add the x = 0 solution on the top
    inner_solution = vcat(inner_solution[end,:]', inner_solution)

    # Add the initial condition on the left
    inner_solution = hcat(u0.(range(0, L, M + 2)), inner_solution)

    return inner_solution
end

function plot_solution(solution::Matrix{Float64}, filename::String, u0::Function, a::Float64, tf::Float64)
    # Plot an animation of the solution over time
    M = size(solution, 1) - 2
    N = size(solution, 2) - 2
    Δx = L / (M + 1)
    Δt = tf / (N + 1)
    anim = @animate for n in 0:N+1
        plot(range(0, L, M + 2), solution[:, n+1], label="Estimated Solution", lw=2)
        plot!(range(0, L, M + 2), [u0(Δx * j - a * n * Δt) for j in range(0, M + 1)], label="True Solution", lw=2)
        plot!(xlabel="x", ylabel="u(x, t)")
        plot!(yrange=(0,1.0))
        plot!(title="Propagation of the Advection Equation Solution")
    end

    # Show animation
    gif(anim, filename, fps=30)
end

function evaluate_solution(solution::Matrix{Float64}, u0::Function, a::Float64, tf::Float64)
    # Solve the system and evaluate its error
    M = size(solution, 1) - 2
    N = size(solution, 2) - 2
    Δx = L / (M + 1)
    Δt = tf / (N + 1)

    true_U = [u0(x - a * t) for x in range(0, L, M + 2), t in range(0, tf, N + 2)]
    return sqrt(Δx * Δt) * norm(solution - true_U, 2) # Frobenius norm for treating matrices as vectors
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

function evaluate_Ns(build_system::Function, u0::Function, a::Float64, tf::Float64, Ns...)
    # Evaluate the PDE for different Ns, get the L2 error, find the convergence rates and plot
    # We choose M = 4N for each N to ensure that the error does not explode. Note that this is specific to the
    # velocities chosen in p3a and p3b, and this must be changed if used on a different PDE
    L2_errors = Float64[]
    for N in Ns
        solution = solve_system(build_system, 4N, N, u0, a, tf)
        push!(L2_errors, evaluate_solution(solution, u0, a, tf))
    end

    println(L2_errors)

    _, n = estimate_numerical_convergence([Ns...], L2_errors)
    println("The estimated order of convergence was $n")
end

function plugin(u0::Function, a::Float64, tf::Float64)
    # Plug these values into a specific function
    function evaluator(func::Function)
        # Run a function
        function run(args...)
            return func(args..., u0, a, tf)
        end
    end
end