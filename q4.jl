using Plots
using LaTeXStrings
using LinearAlgebra
include("midpoint_kepler.jl")
include("rk4_kepler.jl")

# Reference solution
ū = RK4(.0005)

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
    plot!(legend=false, xlim=(-1.5,1.5), ylim=(-1.5, 1.5))
    savefig("$scheme_name with Δt = $Δt.png")
end

# Plot all of our solutions
plot_solution(ū, .0005, "RK4")
plot_solution(u_RK4_Δt1, .1, "RK4")
plot_solution(u_RK4_Δt01, .01, "RK4")
plot_solution(u_midpoint_Δt1, .1, "Midpoint")
plot_solution(u_midpoint_Δt01, .01, "Midpoint")

function plot_errors(data, Δt, scheme_name)
    # Plot the error

    # Skip over gaps in our true solution since our Δt is smaller
    true_u = ū[1:Int(Δt/.0005):end,:]

    # Plot the curves
    plot() # clear the plot
    plot!(collect(0:Δt:500), log.(abs.(data[:,1] - true_u[:,1])), label="log(q1 error)")
    plot!(collect(0:Δt:500), log.(abs.(data[:,2] - true_u[:,2])), label="log(q2 error)")
    plot!(collect(0:Δt:500), log.(abs.(data[:,3] - true_u[:,3])), label="log(p1 error)")
    plot!(collect(0:Δt:500), log.(abs.(data[:,4] - true_u[:,4])), label="log(p2 error)")
    xaxis!("k")
    yaxis!("log(Error)")
    title!("Error in Solution for $scheme_name with " * L"\Delta t = " * "$Δt")
    savefig("$scheme_name Solution error with Δt = $Δt.png")
end

function plot_errors_smooth(data, Δt, scheme_name)
    # Plot the error but plot only accumulative maxima to make it smoother

    # Skip over gaps in our true solution since our Δt is smaller
    true_u = ū[1:Int(Δt/.0005):end,:]

    # Plot the curves
    plot() # clear the plot
    plot!(collect(0:Δt:500), accumulate(max, log.(abs.(data[:,1] - true_u[:,1]))), label="log(q1 error)")
    plot!(collect(0:Δt:500), accumulate(max, log.(abs.(data[:,2] - true_u[:,2]))), label="log(q2 error)")
    plot!(collect(0:Δt:500), accumulate(max, log.(abs.(data[:,3] - true_u[:,3]))), label="log(p1 error)")
    plot!(collect(0:Δt:500), accumulate(max, log.(abs.(data[:,4] - true_u[:,4]))), label="log(p2 error)")
    xaxis!("k")
    yaxis!("log(Error)")
    title!("Accum. Max. Error for $scheme_name with " * L"\Delta t = " * "$Δt")
    savefig("Accum. Max. $scheme_name Solution error with Δt = $Δt.png")
end

# Plot all of our errors
plot_errors(u_RK4_Δt1, .1, "RK4")
plot_errors(u_RK4_Δt01, .01, "RK4")
plot_errors(u_midpoint_Δt1, .1, "Midpoint")
plot_errors(u_midpoint_Δt01, .01, "Midpoint")

# Smooth the errors and plot again
plot_errors_smooth(u_RK4_Δt1, .1, "RK4")
plot_errors_smooth(u_RK4_Δt01, .01, "RK4")
plot_errors_smooth(u_midpoint_Δt1, .1, "Midpoint")
plot_errors_smooth(u_midpoint_Δt01, .01, "Midpoint")

function hamiltonian(vec::Union{Matrix{Float64}, SubArray})
    # Return the hamiltonian evaluated at vector
    q1, q2, p1, p2 = vec
    r = sqrt(q1^2 + q2^2)
    return (p1^2 + p2^2) / 2 - 1 / r - .01 / (2r^3)
end

# H(q(0),p(0))
β = .6
H00 = hamiltonian(Float64[1 - β 0 0 sqrt((1 + β) / (1 - β))])

@views function max_diff_hamiltonian(data::Matrix{Float64}, scheme_name::String, Δt::Float64)
    print("$scheme_name, Δt = $Δt: ")
    println(maximum(abs.([hamiltonian(row) - H00 for row in eachrow(data)])))
end


println("Maximal absolute difference in hamiltonian: ")
max_diff_hamiltonian(u_RK4_Δt1, "RK4", .1)
max_diff_hamiltonian(u_RK4_Δt01, "RK4", .01)
max_diff_hamiltonian(u_midpoint_Δt1, "Midpoint", .1)
max_diff_hamiltonian(u_midpoint_Δt01, "Midpoint", .01)