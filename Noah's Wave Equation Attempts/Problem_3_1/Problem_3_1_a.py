'''
author: Noah

Dirichlet wave equation!
u_tt = u_xx
'''
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation

dt = 0.5
dx = 1.0
c = 1
s = c**2 * dt**2 / dx**2

x_start, x_end = 0.0, 5.0  # x-bounds
t_start, t_end = 0.0, 15.0  # t-bounds

X = np.arange(x_start, x_end + dx, dx)  # 1D x dimension
T = np.arange(t_start, t_end + dt, dt)  # 1D t dimension

J, N = X.shape[0], T.shape[0] # number of discrete x-axis and t-axis elements
#print J, N


def phi_j(j): #initial condition: u(x,0)
    return 25 - (j*dx)**2

def psi_j(j): #initial velocity: du/dt(x,0)
    return 0


def initial_u(X, T):
    """
    Creates initial condition of wave equation
    """
    u = np.zeros((N,J), dtype='float')

    # the first row
    for j in range(J):   
        u[0,j] = phi_j(j)

    # the second row
    for j in range(1,J-1):  # exclude edge points due to (j-1) and (j+1) terms, which enforces dirichlet condition
        u[1,j] = (s/float(2)) * (phi_j(j+1) + phi_j(j-1)) + (1-s)*phi_j(j) + psi_j(j)*dt
    
    return u

u = initial_u(X,T)

# plot first two rows to confirm initial conditions
fig = plt.figure()

ax1 = fig.add_subplot(211)
ax1.set_title("Initial condition")
ax1.plot(u[0])
ax2 = fig.add_subplot(212)
ax2.set_title("n = 1")
ax2.plot(u[1])

plt.show()
#pyplot.savefig("test_0.png")

def picard_engage(u, s, J, N):
    """Implements a simple scheme for solving the diffusion equation for
    the Dirichlet Boundary conditions: u(t, 0) = 0 and u(t, l) = 0.

    Arguments
    ---------
    u -- wave solution, with first two rows already filled in.
    s -- ratio of step distances between points in t, and between points in x
    J -- total number of discrete x-axis points
    N -- total number of discrete t-axis points
    """
    for n in range(1,N-1):
        for j in range(1, J - 1):
            u[n+1,j] = s*(u[n,j+1] + u[n,j-1]) + 2*(1-s)*u[n,j] - u[n-1,j]

        #enforce boundary conditions
        u[n+1,0] = 0
        u[n+1,J-1] = 0
    
    return u

u = picard_engage(u, s, J, N)

# Create an animation of the solution as time elapses
fig, ax = plt.subplots()
ax.set_ylim([min(u.flatten()),max(u.flatten())])
#ax.set_autoscale_on(False)
ax.set_xlabel('X')
ax.set_ylabel('u')
wave, = ax.plot(X, u[0,:])

def animate(i):
    ax.set_title("n = %d" % i)
    wave.set_ydata(u[i,:])  # update the data
    return wave,

# Init only required for blitting to give a clean slate.
def init():
    wave.set_ydata(u[0,:])
    return wave,

ani = animation.FuncAnimation(fig, animate, N, init_func=init,
                              interval=15000/N, blit=False, repeat=False)
plt.show()
ani.save('3_1_a.mp4')

# Show (3,3) point in space-time
fig = plt.figure()

ax1 = fig.add_subplot(111)
ax1.set_title("Time = 3; u(3,3) = %f" % u[T.tolist().index(3),X.tolist().index(3)])
ax1.set_xlabel('X')
ax1.set_ylabel('u')
ax1.plot(u[T.tolist().index(3),:])

plt.show()
plt.savefig('3_1_a.jpg')