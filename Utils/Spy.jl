using Plots

function spy_better(obj, dim1, dim2)
    # Spy helper, because julia Plots/Spy is awful
    # Pass in object and its dimensions
    to_plot = Vector{Integer}[]
    for i in dim1
        for j in dim2
            if !isa(obj[i,j], Number) || obj[i,j] != 0
                # This is not a number or the value is nonzero, so mark it on the plot
                push!(to_plot, [i, j])
            end
        end
    end

    # Convert to matrix
    to_plot = permutedims(hcat(to_plot...))

    # Plot!
    scatter(to_plot[:,1], to_plot[:, 2], xlims=(dim1.start, dim1.stop), ylims=(dim2.start,dim2.stop), legend=false)
end