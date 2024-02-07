using LinearAlgebra
include("linear.jl")
include("solution.jl")

mutable struct System
    var_names::Vector{String}
    A::Matrix{Float64}
    b::Vector{Float64}
    var_idxs::Dict{String, Int}
    determined::Int

    function System(names::Vector{String})
        var_idxs = Dict(zip(names, 1:length(names)))
        A = zeros(Float64, length(names), length(names))
        b = zeros(Float64, length(names))
        return new(names, A, b, var_idxs, 1)
    end
end

function add_constraint!(system::System, equation::Equation)
    linear = equation.linear
    height = size(linear.left[1], 1)
    bottom = system.determined + height
    system.b[system.determined:bottom - 1] += equation.RHS

    for i in 1:length(linear.left)
        newnewA = linear.left[i]
        newnewx = linear.right[i]

        for j in 1:length(newnewx)
            var = newnewx[j]
            var_idx = system.var_idxs[var]

            system.A[system.determined:bottom - 1, var_idx] += newnewA[:, j]
        end
    end

    system.determined = bottom
end