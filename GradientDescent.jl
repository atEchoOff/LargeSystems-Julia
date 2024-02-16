include("LargeSystems/solvers.jl")
using Plots
include("Utils/PDEs.jl")

# We will model the PDE -u''(x)=f(x)

f(x) = pi^2 * cos(pi * x)
u(x) = cos(pi * x) # Our true solution

system = build_1D_poisson(30, 1, -1, f)
solver = GradientMethodSolver(sparse=true)

x = solve(solver, system)
x = [1; x.values; -1] # Add the endpoints to our solution

domain = range(0, 1, length(x))
plot()
plot!(domain, u, label="True u")
plot!(domain, x, label="Estimated u")
plot!(title="Estimation of u via gradient descent")