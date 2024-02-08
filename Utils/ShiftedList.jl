struct ShiftedList
    # A simple list which indexes by a custom shift
    start::Int64
    list::Vector

    function ShiftedList(start::Int64, list::Vector)
        return new(start, list)
    end
end

Base.:(getindex)(list::ShiftedList, idx::Int64) = begin
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