include("Utils/PDEs.jl")
include("LargeSystems/solvers.jl")
using Plots
using LinearAlgebra

# Random function for right hand side
f(x, y) = rand(Float64)

# First, create our matrix
system = build_2D_poisson(32, zero_2D, f, stencil=5, spy_plot=true)
A = system.A
b = system.b
N = size(b, 1)

println("N = ", N)

# Jacobi method. Zero tolerance for error, and go for N // 2 iterations
jacobi = JacobiMethodSolver(tol_type=ZERO, maxit=N÷2, sparse=true)

# Gauss-Seidel solver. See above.
gauss_seidel = GaussSeidelSolver(tol_type=ZERO, maxit=N÷2, sparse=true)

# Successive Overrelaxation solver. See above, and set omega=1.7
sor = SuccessiveOverrelaxationSolver(tol_type=ZERO, maxit=N÷2, ω=1.7, sparse=true)

# Gradient method solver. See above.
gradient = GradientMethodSolver(tol_type=ZERO, maxit=N÷2, sparse=true)

# Conjugate gradient method solver. Go until the relative residual is less that 10⁻¹⁵. 
conjugate_gradient = ConjugateGradientMethodSolver(tol_type=RELATIVE_RESIDUAL, tol=1e-15, maxit=typemax(Int64), sparse=true)

# Before continuing, find the true solution so we can get the relative errors
true_x = solve(SimpleSolver(sparse=true), system).values
norm_x = norm(true_x) # precalculate these terms
norm_b = norm(b)

mutable struct ResidualMetadata <: Metadata
    # This will help us store our data at each iteration
    relative_errors::Vector{Float64}
    relative_residuals::Vector{Float64}
    diff_cost::Vector{Float64}

    function ResidualMetadata()
        return new(Float64[], Float64[], Float64[])
    end
end

function save!(metadata::ResidualMetadata; kwargs...)
    # Save relative error, relative residual, and f(x) - f(x*) at each frame
    x = kwargs[:x]
    r = kwargs[:r]

    push!(metadata.relative_errors, norm(x - true_x) / norm_x)
    push!(metadata.relative_residuals, norm(r) / norm_b)

    # The absolute value is just to make sure the cost is positive for non-gradient methods
    push!(metadata.diff_cost, abs(.5 * (x - true_x)'A*(x - true_x)))
end

function evaluate_solver(solver::Solver)
    # Evaluate a solver and return a list of relative residuals and relative errors (logged)
    # First, construct metadata object to store residuals and iterands
    metadata = ResidualMetadata()

    # Solve our system
    solve(solver, system, metadata=metadata)
    return log.(metadata.relative_residuals), log.(metadata.relative_errors), log.(metadata.diff_cost)
end

# Get the errors for each method
jacobi_log_resids, jacobi_log_errors, _ = evaluate_solver(jacobi)
gs_log_resids, gs_log_errors, _ = evaluate_solver(gauss_seidel)
sor_log_resids, sor_log_errors, _ = evaluate_solver(sor)

# For the descent methods, note that our matrix A is *negative definite*. So, we negate A and b to
# solve the same system, and make -A *positive definite*. 
system.A = -system.A
A = -A
system.b = -system.b
b = -b
grad_log_resids, grad_log_errors, grad_log_diff_costs = evaluate_solver(gradient)
cg_log_resids, cg_log_errors, cg_log_diff_costs = evaluate_solver(conjugate_gradient)

# Now, plot a semilog for the residuals for each method
plot() # Clear the plot
plot!(jacobi_log_resids, label="Jacobi")
plot!(gs_log_resids, label="Gauss-Seidel")
plot!(sor_log_resids, label="SOR")
plot!(grad_log_resids, label="Steepest Descent")
plot!(cg_log_resids, label="CG")
plot!(legend=:bottomright, foreground_color_legend = nothing, background_color_legend = nothing)
xlabel!("Iteration k")
ylabel!("log(relative residual)")
title!("The Relative Residuals over each Iteration")
savefig("relative_residuals.png")

# Plot semilog for relative errors
plot() # Clear the plot
plot!(jacobi_log_errors, label="Jacobi")
plot!(gs_log_errors, label="Gauss-Seidel")
plot!(sor_log_errors, label="SOR")
plot!(grad_log_errors, label="Steepest Descent")
plot!(cg_log_errors, label="CG")
plot!(legend=:bottomright, foreground_color_legend = nothing, background_color_legend = nothing)
xlabel!("Iteration k")
ylabel!("log(relative error)")
title!("The Relative Errors over each Iteration")
savefig("relative_errors.png")

# Now, we want to plot cost(x) - cost(x*) over each iteration for CG and Gradient Descent
plot() # Clear the plot
plot!(log.(eachindex(grad_log_diff_costs)), grad_log_diff_costs, label="Steepest Descent")
plot!(log.(eachindex(cg_log_diff_costs)), cg_log_diff_costs, label="CG")
plot!(legend=:bottomleft)
xlabel!("log(iteration k)")
ylabel!("log(cost)")
title!("The Effect of log(Iteration) on log(Cost)")
savefig("cost_difference.png")