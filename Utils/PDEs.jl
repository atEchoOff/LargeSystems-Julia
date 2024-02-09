include("../LargeSystems/linear.jl")
include("../LargeSystems/system.jl")
include("Layered.jl")

function build_1D_poisson(N, a, b, f)
    # Build PDE -u''(x) = f(x)
    # So that u(0) = a
    # and u(1) = b

    h = 1 / (N + 1)
    U = ["U$i" for i in range(1, N)]
    system = System(U)
    U = V(U...) # Turn U into a list of variables for indexing

    # Add our constraints
    add_constraint!(system, (2 * U[1] - U[2]) / h^2 == f(h * 1) + a / h^2)
    for i in range(2, N - 1)
        add_constraint!(system, (-U[i - 1] + 2 * U[i] - U[i + 1]) / h^2 == f(h * i))
    end

    add_constraint!(system, (-U[N - 1] + 2 * U[N]) / h^2 == f(h * N) + b / h^2)

    return system
end

function build_2D_poisson(N, boundary, f; Δf=zero, stencil=9)
    # Build the PDE Δu = f(x,y)
    # So that over ∂([0,1]²), u(x,y)=boundary(x,y)
    # Optionally pass the laplacian of f to improve convergence to O(h⁴), works only when stencil=9
    # Choose either the 5 point or 9 point stencil

    h = 1 / (N + 1)
    U = ["U$i,$j" for i in range(1, N), j in range(1, N)]
    system = System(U)
    U = Layered{Tuple{Integer, Integer}}([V(U[i,j]) for i in range(1, N), j in range(1, N)]) # Make U a 2D array for indexing variables

    # Set boundary values
    for i in range(0, N + 1)
        U[i,0] = boundary(i * h, 0)
        U[i,N + 1] = boundary(i * h, 1)
    end

    for j in range(0, N + 1)
        U[0,j] = boundary(0, j * h)
        U[N + 1,j] = boundary(1, j * h)
    end

    # Add constraints
    if stencil == 9
        for j in range(1, N)
            for i in range(1, N)
                add_constraint!(system, 1 / (6h^2) * (4U[i-1,j] + 4U[i+1,j] + 4U[i,j-1] 
                                        + 4U[i,j+1] + U[i-1,j-1] + U[i+1,j-1] + U[i+1,j+1] + U[i-1,j+1] - 20U[i,j]) == f(h*i, h*j) + h^2 / 12 * Δf(h*i, h*j))
            end
        end
    elseif stencil == 5
        for j in range(1, N)
            for i in range(1, N)
                add_constraint!(system, 1 / h^2 * (U[i-1,j] + U[i+1,j] + U[i,j-1] + U[i,j+1] - 4U[i,j]) == f(h*i, h*j))
            end
        end
    end

    return system
end