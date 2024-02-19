using Plots
using LaTeXStrings
using LinearAlgebra
using CSV
using DataFrames
include("midpoint_kepler.jl")
include("rk4_kepler.jl")

# Import reference solution from matlab run
# (From sample code)
# It takes a while so buckle your seatbelt
matlab_data = CSV.read("data.csv", DataFrame, header=false)
ū = Matrix(matlab_data)' # FIXME what should I do with reference solution?

# Now, estimate the solution with RK4
u_RK4_Δt1 = RK4(.1)
u_RK4_Δt01 = RK4(.01)

# Now implicit midpoint
u_midpoint_Δt1 = midpoint(.1)
u_midpoint_Δt01 = midpoint(.01)

function plot_solution(data, Δt, scheme_name)
    # Plot the solution
    plot(data[:,1], data[:,2], label="Estimated Solution")
    title!("Estimated Solution via $scheme_name with " * L"\Delta t = " * "$Δt")
    xaxis!(L"q_1")
    yaxis!(L"q_2")
    savefig("$scheme_name with Δt = $Δt.png")
end

# Plot all of our solutions
plot_solution(u_RK4_Δt1, .1, "RK4")
plot_solution(u_RK4_Δt01, .01, "RK4")
plot_solution(u_midpoint_Δt1, .1, "Midpoint")
plot_solution(u_midpoint_Δt01, .01, "Midpoint")

# Error at a specific index
function error_func_generator(u::Matrix{Float64})
    return function error(i::Integer)
        return norm(u[i,1:2] - ū[i,1:2]) # FIXME what to do with reference solution?
    end
end

function plot_errors(data, Δt, scheme_name)
    # Plot the error
    plot(collect(0:Δt:500), log.(error_func_generator(data).(range(1, size(data, 1)))), label="Solution Error")
    xaxis!("k")
    yaxis!("Error")
    title!("Error in Solution for $scheme_name with " * L"\Delta t = " * "$Δt")
    savefig("$scheme_name Solution error with Δt = $Δt.png")
end

plot_errors(u_RK4_Δt1, .1, "RK4")
plot_errors(u_RK4_Δt01, .01, "RK4")
plot_errors(u_midpoint_Δt1, .1, "Midpoint")
plot_errors(u_midpoint_Δt01, .01, "Midpoint")