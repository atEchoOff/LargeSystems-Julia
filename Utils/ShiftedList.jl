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