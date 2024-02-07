struct Solution
    values::Vector
    var_idxs::Dict{String, Int}

    function Solution(values::Vector, var_idxs::Dict{String, Int})
        return new(values, var_idxs)
    end
end

import Base: getindex

function getindex(sol::Solution, key::String)
    return sol.values[sol.var_idxs[key]]
end

function getindex(sol::Solution, key::Vector{String})
    return sol.values[[sol.var_idxs[k] for k in key]]
end