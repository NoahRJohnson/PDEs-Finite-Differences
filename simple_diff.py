#!/Users/edelsonc/anaconda/bin/python
"""Script runs a simple forward finite difference model of the diffusion 
equation. I believe that the end-points are supposed to be excluded..."""
import numpy
from matplotlib import pyplot

dx = 1.0  # separation between points in x dimension
x_start, x_end = 0.0, 5.0  # x-bounds
x = numpy.arange(x_start, x_end + dx, dx)  # 1D x dimension
print(x)

dt =  1.0/2 # t separation steps for stability
t_start, t_end = 0.0, 300.0  # t-bounds
t = numpy.arange(t_start, t_end + dt, dt)  # 1D t dimension

def phi_0(x, t):
    """Creates initial condition of heat equation

    Arguments
    ---------
    x -- x-array to represent x-dimension
    """   
    u = numpy.zeros((x.shape[0],t.shape[0]), dtype='float')

    for i, x in enumerate(x):   
        u[i,0] = 25 - x**2
    u[0,0] = 0
    u[5,0] = 0
    return u

u = phi_0(x,t)

# plot to confirm initial conditions
size = 10
pyplot.figure(figsize=(size, size))
pyplot.plot(x, u[:,0])
pyplot.xlim(xmin=x_start, xmax=x_end)
pyplot.ylim(ymin=0)
pyplot.title('initial condition')
pyplot.show()
#pyplot.savefig("test_0.png")

def simple_diff(u, t, dt, dx):
    """Implements a simple scheme for solving the diffusion equation for
    the Dirchlet Boundary conditions: u(0, t) = 0 and u(l,t) = 0.
    For Neumann Boundary conditions, uncomment line 59 & 60.
    For mixed Neumann - Dirichlet conditions, uncomment either 59 or 60.

    Arguments
    ---------
    u -- initial condition of solution (array)
    t -- t-dimension linear array
    dt -- step distance between points in t
    dx -- step distance between points in x
    """
    s = dt/(dx**2)
    print u.shape[0]
    for timeStep in t:
        if (timeStep == t_end):
            break
        for i in range(1, u.shape[0] - 1):
            u[i,int(timeStep/dt)+1] = s * (u[i+1,int(timeStep/dt)] - 2*u[i,int(timeStep/dt)] + u[i-1,int(timeStep/dt)]) + u[i,int(timeStep/dt)]
#         u[-1] = u[-3]
#         u[0] = u[2]

    return u

u = simple_diff(u, t, dt, dx)

# plot to confirm
size = 10
pyplot.figure(figsize=(size, size))
pyplot.plot(x, u[:,100])
pyplot.xlim(xmin=x_start, xmax=x_end)
pyplot.ylim(ymin=0)
pyplot.title('end condition')
pyplot.show()

#pyplot.savefig("test_f.png")
