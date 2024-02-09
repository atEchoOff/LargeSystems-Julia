mutable struct Layered{U}
    # A container which has a main section and an outer section
    # At construction, initialize the main part of the list
    # User can then set more values outside of the list's bounds
    # This is useful for creation of variables for system and setting boundary values outside

    list::Union{Vector, ShiftedList, Matrix}
    outer::Dict{U, Number}

    function Layered{U}(list::Union{Vector, ShiftedList, Matrix}) where {U}
        return new{U}(list, Dict{U, Number}())
    end
end

Base.:(getindex)(list::Layered, idxs...) = begin
    try
        return getindex(list.list, idxs...)
    catch BoundsError
        return list.outer[idxs]
    end
end

Base.:(setindex!)(list::Layered, v::Number, idxs...) = begin
    list.outer[idxs] = v
end