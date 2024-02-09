struct ShiftedList
    # A simple list which indexes by a custom shift
    start::Int64
    list::Vector
    hard::Bool # Do not allow negative indices if hard

    function ShiftedList(start::Int64, list::Vector; hard::Bool=false)
        return new(start, list, hard)
    end
end

Base.:(getindex)(list::ShiftedList, idx::Int64) = begin
    if idx - list.start < 1
        throw(DomainError(idx, "This shifted list is hard, it cannot be accessed out of bounds"))
    end
    return list.list[idx - list.start + 1]
end

Base.:(length)(list::ShiftedList) = begin
    return length(list.list)
end

Base.:(iterate)(list::ShiftedList) = begin
    return list[list.start], list.start + 1
end

Base.:(iterate)(list::ShiftedList, state) = begin
    if state >= length(list.list) + list.start
        return nothing
    else
        return list[state], state + 1
    end
end