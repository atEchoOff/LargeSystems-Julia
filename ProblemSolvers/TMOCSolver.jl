include("../LargeSystems/system.jl")
include("../LargeSystems/solvers.jl")
include("../LargeSystems/linear.jl")
include("../Utils/ShiftedList.jl")


mutable struct TMOCSolver
    # A class designed for solving an optimal control problem of the form given in assignment 2
    # Using the θ method, with equal stepsizes h
    # Due to the amount of parameters, this class will use the builder construct
    t0::Number
    tf::Number
    y0::Vector
    ny::Int64
    nu::Int64

    Δlᶠ::Function
    Δᵧl::Function
    Δᵤl::Function
    f::Function
    fᵧ::Matrix
    fᵤ::Matrix

    function TMOCSolver(t0::Number, tf::Number, y0::Vector, ny::Int64, nu::Int64, Δlᶠ::Function, Δᵧl::Function, Δᵤl::Function, f::Function, fᵧ::Matrix, fᵤ::Matrix)
        return new(t0, tf, y0, ny, nu, Δlᶠ, Δᵧl, Δᵤl, f, fᵧ, fᵤ)
    end
end

function solve(tmocSolver::TMOCSolver, θ, K)
    # Solve the problem given θ
    # A Solution object is returned
    
    # Collect the variables into one long list
    # Initialize the variables
    _y = ShiftedList(0, [ShiftedList(1, ["y$i$j" for j in range(1, tmocSolver.ny)]) for i in range(0, K)])
    _λ = ShiftedList(1, [ShiftedList(1, ["λ$i$j" for j in range(1, tmocSolver.ny)]) for i in range(1, K)])
    _u = ShiftedList(0, [ShiftedList(1, ["u$i$j" for j in range(1, tmocSolver.nu)]) for i in range(0, K)])

    # Shorthand
    Δlᶠ = tmocSolver.Δlᶠ
    Δᵧl = tmocSolver.Δᵧl
    Δᵤl = tmocSolver.Δᵤl
    f = tmocSolver.f
    fᵧ = tmocSolver.fᵧ
    fᵤ = tmocSolver.fᵤ
    y = ShiftedList(0, [V(_y[i].list) for i in range(0, K)])
    λ = ShiftedList(1, [V(_λ[i].list) for i in range(1, K)])
    u = ShiftedList(0, [V(_u[i].list) for i in range(0, K)])
    h = (tmocSolver.tf - tmocSolver.t0) / K
    y0 = tmocSolver.y0

    # Initialize our domain
    t = ShiftedList(0, [tmocSolver.t0 + h * i for i in range(0, K)])
    
    # Start building our system!
    system = System(_y, _λ, _u)

    # Constraint from (8)
    add_constraint!(system, λ[K] - h * (1 - θ) * fᵧ' * λ[K] ==
                                -Δlᶠ(y[K]) - h * (1 - θ) * Δᵧl(y[K], u[K], t[K]))

    for k in range(0, K - 2)
        # Add constraint from (9)
        add_constraint!(system, λ[k + 1] - h * (1 - θ) * fᵧ' * λ[k + 1] ==
                                    -h * Δᵧl(y[k + 1], u[k + 1], t[k + 1]) + λ[k + 2]
                                    + h * θ * fᵧ' * λ[k + 2])
    end
        
    
    # Add the constraint from (10)
    add_constraint!(system, 0 == h * Δᵤl(y[0], u[0], t[0])
                                - h * fᵤ' * λ[1])
    
    # Add the constraint from (11)
    add_constraint!(system, 0 == h * Δᵤl(y[K], u[K], t[K])
                                - h * fᵤ' * λ[K])
    
    for k in range(1, K - 1)
        # Add the constraint from (12)
        add_constraint!(system, 0 == h * Δᵤl(y[k], u[k], t[k])
                                    - h * θ * fᵤ' * λ[k + 1]
                                    - h * (1 - θ) * fᵤ' * λ[k])
    end
        
    # Add θ method base case
    add_constraint!(system, y[0] == y0)

    # Add all θ method steps
    for k in range(0, K - 1)
        add_constraint!(system, y[k + 1] == y[k]
                                    + h * θ * f(y[k], u[k], t[k]) 
                                    + h * (1 - θ) * f(y[k + 1], u[k + 1], t[k + 1]))
    end
    
    return solve(SimpleSolver(sparse=true), system) # The resulting system is relative sparse
end