using LinearAlgebra

struct Identity
    val::Number
    n::Int64

    function Identity(val::Number, dim::Int64)
        return new(val, dim)
    end
end

Base.:(size)(ident::Identity, _::Int64) = begin
    return ident.n
end

Base.:(getindex)(ident::Identity, _::Colon, column::Int64) = begin
    # Get a specific column from identity matrix
    ret = zeros(Float64, (ident.n, 1))
    ret[column, 1] = ident.val
    return ret
end

Base.:(*)(ident::Identity, val::Number) = begin
    return Identity(ident.val * val, ident.n)
end

Base.:(*)(val::Number, ident::Identity) = begin
    return Identity(val * ident.val, ident.n)
end

Base.:(*)(ident::Identity, matr::Union{Identity, AbstractMatrix}) = begin
    return ident.val * matr
end

Base.:(*)(matr::Union{Identity, AbstractMatrix}, ident::Identity) = begin
    return matr * ident.val
end

Base.:(/)(ident::Identity, val::Number) = begin
    return ident * (1 / val)
end

Base.:-(ident::Identity) = -1 * ident

function V(vars...)
    if length(vars) == 1
        return Linear(Identity(1, length(vars[1])), vars[1])
    end

    ret = []
    for var in vars
        if typeof(var) == String
            # Variable must be packed in a list
            var = [var]
        end
        push!(ret, Linear(Identity(1, length(var)), var))
    end
    return ret
end

mutable struct Linear
    left::Vector{Union{Identity, AbstractMatrix}}
    right::Vector{Vector{String}}
    constant::Vector{Float64}

    function Linear(left::Vector{Union{Identity, AbstractMatrix}}, right::Vector{Vector{String}}, constant::Vector{Float64})
        return new(left, right, constant)
    end

    function Linear(left::Union{Identity, AbstractMatrix}, right::Vector{String})
        constant = zeros(Float64, size(left, 1))
        return new([left], [right], constant)
    end
end

Base.:(deepcopy)(linear::Linear) = begin
    ret = Linear(deepcopy(linear.left), deepcopy(linear.right), deepcopy(linear.constant))
    return ret
end

Base.:(==)(linear::Linear, other::Linear) = begin
    return Equation(linear - other, other.constant - linear.constant)
end

Base.:(==)(linear::Linear, other::Vector) = begin
    return Equation(linear, other - linear.constant)
end

Base.:(==)(other::Vector, linear::Linear) = begin
    return linear == other # Reduce to Linear == Vector
end

Base.:(==)(linear::Linear, other::Number) = begin
    # Shorthand, turn number into tall vector
    height = size(linear.left[1], 1)
    other = ones(height) * other
    return linear == other # Reduce to Linear == Vector
end

Base.:(==)(other::Number, linear::Linear) = begin
    return linear == other # Reduce to Linear == Number
end

Base.:(*)(linear::Linear, val::Union{Number, AbstractMatrix, Identity}) = begin
    ret = deepcopy(linear)
    for i in 1:length(linear.left)
        ret.left[i] = linear.left[i] * val
    end

    ret.constant = linear.constant * val

    return ret
end

Base.:(*)(val::Union{Number, AbstractMatrix, Identity}, linear::Linear) = begin
    ret = deepcopy(linear)
    for i in 1:length(linear.left)
        ret.left[i] = val * linear.left[i]
    end

    ret.constant = val * linear.constant

    return ret
end

Base.:(/)(linear::Linear, other::Number) = begin
    return linear * (1 / other)
end

Base.:(+)(linear::Linear, other::Linear) = begin
    ret = deepcopy(linear)
    append!(ret.left, other.left)
    append!(ret.right, other.right)
    ret.constant = linear.constant + other.constant

    return ret
end

Base.:(+)(linear::Linear, other::Vector) = begin
    ret = deepcopy(linear)
    ret.constant += other
    return ret
end

Base.:(+)(other::Vector, linear::Linear) = begin
    return linear + other # Reduce to Linear + Vector
end

Base.:(+)(linear::Linear, other::Number) = begin
    return linear + ones(size(linear.left[1], 1)) * other # Reduce to Linear + Vector
end

Base.:(+)(other::Number, linear::Linear) = begin
    return linear + other # Reduce to Linear + Number
end

Base.:-(linear::Linear) = -1 * linear

Base.:-(linear::Linear, val::Union{Linear, AbstractMatrix, Identity, Number}) = linear + (-val)

Base.:-(val::Union{AbstractMatrix, Identity, Number}, linear::Linear) = val + (-linear)

struct Equation
    linear::Linear
    RHS::Vector{Float64}

    function Equation(linear::Linear, RHS::Vector{Float64})
        return new(linear, RHS)
    end
end