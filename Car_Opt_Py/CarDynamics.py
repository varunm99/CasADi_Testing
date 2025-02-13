import numpy as np

from casadi import vertcat, sin, cos


def f(state, u):
    x = state[0]
    y = state[1]
    theta = state[2]
    vx = state[3]
    vy = state[4]
    omega = state[5]
    F = u[0]
    tau = u[1]


    xdot = vx
    ydot = vy
    thetadot = omega
    vxdot = F*cos(theta)
    vydot = F*sin(theta)
    omegadot = tau

    return vertcat(xdot, ydot, thetadot, vxdot, vydot, omegadot)

def obstacle(state):
    return state[0:2]