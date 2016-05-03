#!/Users/edelsonc/anaconda/bin/python
"""Implementation of finite difference method to numerically solve the diffusion
equation with Dirichlet Boundary Conditions. Contains a FTCS and CTCS 
implentation."""
import numpy
from numpy.linalg import inv
from matplotlib import pyplot

dx = 1.0  # separation between points in x dimension
x_start, x_end = 0.0, 5.0  # x-bounds
x = numpy.arange(x_start, x_end + dx, dx)  # 1D x dimension

dt =  0.5 # t separation steps for stability
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


def diff_forward(u, t, dt, dx):
    """Implements a simple forward time centered space scheme for solving the
    diffusion equation for the Dirchlet Boundary conditions: u(0, t) = 0 and 
    u(l,t) = 0.

    Arguments
    ---------
    u -- initial condition of solution (array)
    t -- t-dimension linear array
    dt -- step distance between points in t
    dx -- step distance between points in x
    """

    for t in t:
        un = u.copy()
        for i in range(1, len(u) - 1):
            u[i] = dt/dx**2 * (un[i+1] - 2*un[i] + un[i-1]) + un[i]
        u[0] = 0
        u[-1] = 0

    return u


def diff_center(u, t, dt, dx):
    """Implements a simple centered time and centered space scheme for solving
    the diffusion equation for the Dirchlet Boundary conditions: u(0, t) = 0 
    and u(l,t) = 0.

    Arguments
    ---------
    u -- initial condition of solution (array)
    t -- t-dimension linear array
    dt -- step distance between points in t
    dx -- step distance between points in x
    """

    for i,t in enumerate(t):
    
        if i == 0:
            un = u.copy()
            for i in range(1, len(u) - 1):
                u[i] = dt/dx**2 * (un[i+1] - 2*un[i] + un[i-1]) + un[i]
            u[0] = 0
            u[-1] = 0
        else:
            un2 = u.copy()
            for i in range(1, len(u) - 1):
                u[i] = 2 * dt/dx**2 * (un2[i+1] - 2*un2[i] + un2[i-1]) + un[i]
            un = un2

    return u


def tridiag(a, b, c, k1=-1, k2=0, k3=1):
    """Function used to create a tridiagonal matrix for use in the Crank-
    Nicolson scheme.

    Arguments
    ---------
    a, b, c -- the lower, center, and upper diagonal elements (arrays)
    k1, k2, k3 -- indicies of the diagonal. Centered by default
    """

    return numpy.diag(a, k1) + numpy.diag(b, k2) + numpy.diag(c, k3)


def diff_crank_nicolson(u, t, dt, dx):
    """Implements a Crank-Nicolson scheme for solving the diffusion equation 
    for the Dirchlet Boundary conditions: u(0, t) = 0 and u(l,t) = 0.

    Arguments
    ---------
    u -- initial condition of solution, an array.
    t -- t-dimension linear array
    dt -- step distance between points in t
    dx -- step distance between points in x
    """

    s = dt/dx**2
    n = len(u)

    # create matrices for linear system with coefficients
    a0 = numpy.ones(n) * (2 + 2*s)
    a1 = numpy.ones(n-1) * -s
    A = inv(tridiag(a1, a0, a1))

    b0 = numpy.ones(n) * (2 - 2*s)
    b1 = numpy.ones(n-1) * s
    B = tridiag(b1, b0, b1)

    # tranpose u into column vector
    u = numpy.array([u]).T   
   
    for i, t in enumerate(t):
        un = u.copy()
        
        if i == 0:
            u_rhs = numpy.dot(B, un)
            u = numpy.dot(A, u_rhs)
            u[0,0] = 0
            u[-1,0] = 0

        else:
            u_rhs = numpy.dot(B, un)
            u = numpy.dot(A, u_rhs)

    return u.T

# set initial conditions
u_0 = phi_0(x)
u1 = u_0.copy()
u2 = u_0.copy()
u3 = u_0.copy()


# apply forward solver
u_forward = diff_forward(u1, t, dt, dx)
u_center = diff_center(u2, t, dt, dx)
u_cn = diff_crank_nicolson(u3, t, dt, dx)[0]
print(u_cn)

# plot to initial conditions and forward solution
size = 10
pyplot.figure(figsize=(size, size))
pyplot.plot(x, u_0, label="$\\phi(x)$")
pyplot.plot(x, u_forward, label="$u(x,3)$ forward")
pyplot.plot(x, u_center, label="$u(x,3)$ center")
pyplot.plot(x, u_cn, label="$u(x,3)$ Crank-Nicolson")
pyplot.xlim(xmin=x_start, xmax=x_end)
pyplot.ylim(ymin=0, ymax=30)
pyplot.legend()
pyplot.savefig("test_diff.png")
