include("ProblemSolvers/TMOCSolver.jl")
using LinearAlgebra
include("Utils/ShiftedList.jl")
using Plots

A = [2  -1  0;
    -1   2 -1;
     0  -1  2]

B = [4 -1  2;
    -1  5  3;
     2  3  6]

Q = [7  1 -1;
     1  8  0;
    -1  0  9]

R = [10  2 -1;
      2 12  4;
     -1  4  8]

function Δlᶠ(y)
    return 0
end

function Δᵧl(y, u, t)
    return Q * y
end

function Δᵤl(y, u, t)
    return R * u
end

function f(y, u, t)
    return A * y + B * u
end

fᵧ = A

fᵤ = B

t0 = 0
tf = 1
y0 = ones(Float64, 3)
ny = 3
nu = 3

tmocSolver = TMOCSolver(t0, tf, y0, ny, nu, Δlᶠ, Δᵧl, Δᵤl, f, fᵧ, fᵤ)

K = 100
solution = solve(tmocSolver, .5, K)

yT = ShiftedList(1, [["y$j$i" for j in range(0, K)] for i in range(1, ny)])

yT_sols = []
for dim in range(1, ny)
    # Get the solution for dimension "dim" from y
    push!(yT_sols, solution[yT[dim]])
end

yT_sols = ShiftedList(1, yT_sols)

# Now we can plot it if we want
domain = range(0, 1, K + 1)

plot() # Clear the plot
for dim in range(1, ny)
    plot!(domain, yT_sols[dim], label="Solution in dimension $dim")
end

# This is redrawn because I cant figure out how to 
plot!(title="All Dimensions of the Solution y",
      xlabel="t",
      ylablel="y")