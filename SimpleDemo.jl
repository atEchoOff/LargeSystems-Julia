include("LargeSystems/linear.jl")
include("LargeSystems/solvers.jl")
include("LargeSystems/system.jl")

x, y, z = V("x", "y", "z")
system = System(["x", "y", "z"])

add_constraint!(system, 2x + 3y - z == 2)
add_constraint!(system, 2x - z + y == 1)
add_constraint!(system, x + y + z == 1)

solver = SimpleSolver()

println(solve(solver, system).values)