#!/Users/edelsonc/anaconda/bin/python
"""
This script was used to create animations without altering the functions for the
original solver.
"""
import numpy
from numpy.linalg import solve
from scipy.sparse import diags
from scipy.sparse.linalg import lsqr
from matplotlib import pyplot, animation

dx = 1/4  # separation between points in x dimension
x_start, x_end = 0.0, 5.0  # x-bounds
x = numpy.arange(x_start, x_end + dx, dx)  # 1D x dimension

dt =  1/4 # t separation steps for stability
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
    u_soln = [u.copy()]
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
        u_i = u.copy()
        u_soln.append(u_i)
    
    return u_soln


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
    u_soln = [u.copy()]
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
        u_i = u.copy()    
        u_soln.append(u_i)
            
    return u_soln


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

    u_soln = [u.copy()]
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
            u_i = u.copy()    
            u_soln.append(u_i)
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
            u_i = u.copy()    
            u_soln.append(u_i)
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
            u_i = u.copy()    
            u_soln.append(u_i)
    else:
        raise ValueError("Invalid Boundary Condition")

    return u_soln

# set initial conditions
u_0 = phi_0(x)
u1 = u_0.copy()
u2 = u_0.copy()
u3 = u_0.copy()


# apply forward solver
u_forward = diff_forward(u1, t, dt, dx, bd='Mixed')
u_center = diff_center(u2, t, dt, dx, bd='Mixed')
u_cn = diff_crank_nicolson(u3, t, dt, dx, theta=0.5, bd='Dirichlet')

# create figure and line object to be animated using pyplot
size = 10
fig = pyplot.figure(figsize=(size, size))
pyplot.xlabel('position')
pyplot.ylabel('displacement')
ax = pyplot.axes(xlim=(x_start, x_end), ylim=(0, 30), xlabel='position (x)',
    ylabel='displacement ($u(x,t)$)')
line, = ax.plot([], [])

# Defining the init_func and the animated function for FuncAnimate in animation
def init():
    """initializes the animation and sets the initial line data to zero"""
    line.set_data([], [])
    return line,

def animate(i, u, x):
    """Function iterates throught the solution of the heat equation in order to
    animate then using FuncAnimation.

    Arguments
    ---------
    i -- interable
    u -- list of time step solutions to heat equation
    """
    u_i = u[i]
    line.set_data(x, u_i)
    return line,

anim = animation.FuncAnimation(fig, animate, init_func=init,
                                fargs=(u_cn, x), frames=len(t))
anim.save("cn_Dirichlet_ustable.mp4")