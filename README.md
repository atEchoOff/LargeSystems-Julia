Ever have a massive system of equations where the variables are complicated and dynamic? Ever try to make a massive system for an optimal control problem and get stuck trying to figure out what indices go where in your massive 100x100 matrix? Worry no longer. This Julia library (with a twin Python library) is designed for exactly those purposes: the creation of linear systems of equations using a simple and easy notation. See below, a quick demo:

    include("LargeSystems/linear.jl")
    include("LargeSystems/solvers.jl")
    include("LargeSystems/system.jl")
    
    x, y, z = V("x", "y", "z")
    system = System("x", "y", "z")
    
    add_constraint!(system, 2x + 3y - z == 2)
    add_constraint!(system, 2x - z + y == 1)
    add_constraint!(system, x + y + z == 1)
    
    solver = SimpleSolver()
    
    println(solve(solver, system).values)

See how clean that is?? Three constraints, easily plugged in to a 3x3 matrix, without having to worry about indexing. However, the system above is very simple. Want to get more complicated? What about the Laplace 9-point stencil matrix? Each row follows the constraint

![9 Point Laplacian Stencil](https://www.dropbox.com/scl/fi/8md9qd1on8nq084kuycxf/Laplacian.png?rlkey=8srlx3ogjag88qd2wu6hake4r&raw=1)

Where $i$ and $j$ vary from 1 to $N$. Also, there's boundary conditions that need to be taken into account! Usually, this matrix is massive, taking account for the equations on the corners, each side of a square, and all the internal points, and keeping track of it can be a nightmare! Even in MATLAB, where there are functions to put together these matrices, the resulting code is borderline unreadable. However, below, you can see that the equations are plugged directly in:

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

See how beautifully readable and concise that is? The boundary values are set, and then every single row is taken care of in a single equation! Okay okay I hear you, your system is more complicated than the simple 9-point stencil above, a notoriously annoying matrix to build. No, you need the real deal. Consider the theta method optimal control problem, whose equations are goverened by:

![Theta Method Optimal Control System](https://www.dropbox.com/scl/fi/kvelfn6ssnnprhgowwam9/TMOC.png?rlkey=wtfkheal1fx8u0w5tmhjj29iv&raw=1)

Woof! Needless to say, creating the matrix for the system above should require hundreds of lines of code. This library? Plug the constraints directly in:

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

Okay needless to say, this still looks very complicated, but think of how much more simple it is with this library! I rest my case. 

You dont like julia? You are more of a python lover? Check out my equivalent python library.
