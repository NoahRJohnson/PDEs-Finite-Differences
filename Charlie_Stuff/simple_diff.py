#!/Users/edelsonc/anaconda/bin/python
"""
Implementation of finite difference methods to numerically solve the diffusion
equation. A Forward Time Centered Space (FTCS), Centered Time Centered Space
(CTCS), and a Crank-Nicolson scheme are solved for with Dirichlet, Neumann, or
Mixed boundary conditions.
"""
import numpy
from numpy.linalg import solve
from scipy.sparse import diags
from scipy.sparse.linalg import lsqr
from matplotlib import pyplot

dx = 1  # separation between points in x dimension
x_start, x_end = 0.0, 5.0  # x-bounds
x = numpy.arange(x_start, x_end + dx, dx)  # 1D x dimension

dt =  dx**2/2 # t separation steps for stability
t_start, t_end = 0.0, 3.0  # t-bounds
t = numpy.arange(t_start, t_end + dt, dt)  # 1D t dimension


def phi_0(x):
    """Creates initial condition of heat equation

    Arguments
    ---------
    x -- x-array to represent x-dimension
    """   
    u = 25 - x**2

    return u


def diff_forward(u, t, dt, dx, bd = 'Dirichlet'):
    """Implements a simple forward time centered space scheme for solving the
    diffusion equation for the Dirchlet Boundary conditions: u(0, t) = 0 and 
    u(l,t) = 0.

    Arguments
    ---------
    u -- initial condition of solution (array)
    t -- t-dimension linear array
    dt -- step distance between points in t
    dx -- step distance between points in x
    bd -- boundary data. Can be Dirichlet, Neumann, or Mixed. Dirichlet by 
          default
    """

    for t in t:
        un = u.copy()
        for i in range(1, len(u) - 1):
            u[i] = dt/dx**2 * (un[i+1] - 2*un[i] + un[i-1]) + un[i]
        # enforce appropriate boundary conditions
        if bd == 'Dirichlet':
            u[0] = 0
            u[-1] = 0
        elif bd == 'Neumann':
            u[0] = dt/dx**2 * 2 * (un[1] - un[0]) + un[0]
            u[-1] = dt/dx**2 * 2 * (un[-2] - un[-1]) + un[-1]
        elif bd == 'Mixed':
            u[0] = 0
            u[-1] = dt/dx**2 * 2 * (un[-2] - un[-1]) + un[-1]
        else:
            raise ValueError("Invalid Boundary Condition")
        
    return u


def diff_center(u, t, dt, dx, bd='Dirichlet'):
    """Implements a simple centered time and centered space scheme for solving
    the diffusion equation for the Dirchlet Boundary conditions: u(0, t) = 0 
    and u(l,t) = 0.

    Arguments
    ---------
    u -- initial condition of solution (array)
    t -- t-dimension linear array
    dt -- step distance between points in t
    dx -- step distance between points in x
    bd -- boundary data. Can be Dirichlet, Neumann, or Mixed. Dirichlet by 
          default
    """

    for i,t in enumerate(t):
    
        if i == 0:
            un = u.copy()
            for i in range(1, len(u) - 1):
                u[i] = dt/dx**2 * (un[i+1] - 2*un[i] + un[i-1]) + un[i]

            # enforce appropriate boundary conditions
            if bd == 'Dirichlet':
                u[0] = 0
                u[-1] = 0
            elif bd == 'Neumann':
                u[0] = dt/dx**2 * 2 * (un[1] - un[0]) + un[0]
                u[-1] = dt/dx**2 * 2 * (un[-2] - un[-1]) + un[-1]
            elif bd == 'Mixed':
                u[0] = 0
                u[-1] = dt/dx**2 * 2 * (un[-2] - un[-1]) + un[-1]
            else:
                raise ValueError("Invalid Boundary Condition")
        else:
            un2 = u.copy()
            for i in range(1, len(u) - 1):
                u[i] = 2 * dt/dx**2 * (un2[i+1] - 2*un2[i] + un2[i-1]) + un[i]

            # enforce appropriate boundary conditions
            if bd == 'Dirichlet':
                u[0] = 0
                u[-1] = 0
            elif bd == 'Neumann':
                u[0] = dt/dx**2 * 2 * (un2[1] - un2[0]) + un[0]
                u[-1] = dt/dx**2 * 2 * (un2[-2] - un2[-1]) + un[-1]
            elif bd == 'Mixed':
                u[0] = 0
                u[-1] = dt/dx**2 * 2 * (un2[-2] - un2[-1]) + un[-1]
            else:
                raise ValueError("Invalid Boundary Condition")
            un = un2

    return u


def diff_crank_nicolson(u, t, dt, dx, theta=0.5, bd='Dirichlet'):
    """Implements a Crank-Nicolson scheme for solving the diffusion equation 
    for the Dirchlet Boundary conditions: u(0, t) = 0 and u(l,t) = 0.

    Arguments
    ---------
    u -- initial condition of solution, an array.
    t -- t-dimension linear array
    dt -- step distance between points in t
    dx -- step distance between points in x
    theta -- theta value used in shceme. Between 0 and 1. Theta 0.5 default
    bd -- boundary data. Can be Dirichlet, Neumann, or Mixed. Dirichlet by 
          default
    """
    if 0 > theta or  theta > 1:
        raise ValueError("theta must be between 0 and 1")
    else:
        pass

    s = dt/dx**2
    n = len(u)
    u = numpy.array([u]).T
    alpha = 1 - theta

    if bd == 'Dirichlet':
        # create matrices for linear system
        A = diags([-theta*s, (1+2*theta*s), -theta*s], [2, 1, 0],
                    shape=(n-2, n)).toarray()
        A[0,0] = 0
        A[-1,-1] = 0
        B = diags([alpha*s, (1-2*alpha*s), alpha*s], [2, 1, 0],
                    shape=(n-2, n)).toarray()
        # step through time
        for i, t in enumerate(t):
            if i == 0:            
                B[0,0] = s*alpha
                B[-1,-1] = s*alpha
            else:
                B[0,0] = 0
                B[-1,-1] = 0
            un = u.copy()  # create a copy to work wth
            u_rhs = numpy.dot(B, un)
            u = lsqr(A,u_rhs)[0]  # least squared used since n x n-2 matrix

    elif bd == 'Neumann':
        # create Neumann matrices for linear system on lhs
        A = diags([-theta*s, (1+2*theta*s), -theta*s], [-1, 0, 1],
                    shape=(n, n)).toarray()
        A[0,1] *= 2
        A[-1,-2] *= 2
        # create rhs matrix
        B = diags([alpha*s, (1-2*alpha*s), alpha*s], [-1, 0, 1],
                    shape=(n, n)).toarray()
        B[0,1] *= 2
        B[-1,-2] *= 2
        # step through time
        for i, t in enumerate(t):
            un = u.copy()
            u_rhs = numpy.dot(B, un)
            u = solve(A, u_rhs)

    elif bd == 'Mixed':
        # create mixed lhs matrix
        A = diags([-theta*s, (1+2*theta*s), -theta*s], [2, 1, 0],
                    shape=(n-1, n)).toarray()
        A[0,0] = 0
        A[-1,-2] *= 2
        # create rhs matrix
        B = diags([alpha*s, (1-2*alpha*s), alpha*s], [2, 1, 0],
                    shape=(n-1, n)).toarray()
        B[-1,-2] *= 2
        
        
        for i, t in enumerate(t):
            if i == 0:
                B[0,0] = s*alpha
            else:
                B[0,0] = 0
            un = u.copy()
            u_rhs = numpy.dot(B, un)
            u = lsqr(A, u_rhs)[0]
    else:
        raise ValueError("Invalid Boundary Condition")

    return u

# set initial conditions
u_0 = phi_0(x)
u1 = u_0.copy()
u2 = u_0.copy()
u3 = u_0.copy()


# apply forward solver
u_forward = diff_forward(u1, t, dt, dx, bd='Dirichlet')
u_center = diff_center(u2, t, dt, dx, bd='Dirichlet')
u_cn = diff_crank_nicolson(u3, t, dt, dx, theta=0.5, bd='Dirichlet')

# plot to initial conditions and forward solution
size = 10
pyplot.figure(figsize=(size, size))
pyplot.plot(x, u_0, label="$\\phi(x)$")
pyplot.plot(x, u_forward, label="$u(x,3)$ forward")
#pyplot.plot(x, u_center, label="$u(x,3)$ center")
pyplot.plot(x, u_cn, label="$u(x,3)$ Crank-Nicolson")
pyplot.xlim(xmin=x_start, xmax=x_end)
pyplot.ylim(ymin=0, ymax=30)
pyplot.legend()
pyplot.savefig("Dirichlet_diff.png")
