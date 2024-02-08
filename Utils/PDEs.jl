include("../LargeSystems/linear.jl")
include("../LargeSystems/system.jl")

function build_poisson(N, a, b, f)
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