import numpy as np
from pykalman import UnscentedKalmanFilter


def transition_function(state):
    # state: [qx, qy, qz, qw, tx, ty, tz, qx', qy', qz', qw', dx, dy, dz]
    q = state[:4]
    t = state[4:7]
    q_ = state[7:11]
    dt = state[11:]
    omega = np.acos(np.dot(q_, q))
    return np.array([
        *(q*np.sin(2*omega) + q_*np.sin(-omega)) / np.sin(omega),
        *t + dt, *q, *dt])


def observation_function(state):
    return state


ukf = UnscentedKalmanFilter(
    transition_functions=transition_function,
    observation_functions=observation_function
)
