import numpy as np
from pykalman import UnscentedKalmanFilter


STATE_DIMS = 14
OBS_DIMS = 7

OBS_COV_LOW = np.eye(OBS_DIMS) * 0.01
OBS_COV_HIGH = np.eye(OBS_DIMS) * 0.5


def transition_function(state, noise):
    # state: [qw, qx, qy, qz,
    #         tx, ty, tz,
    #         qw', qx', qy', qz',
    #         dx, dy, dz]
    q = state[:4]
    t = state[4:7]
    q_ = state[7:11]
    dt = state[11:]
    omega = np.arccos(np.dot(q_, q) / (np.linalg.norm(q_) * np.linalg.norm(q)))
    if omega != 0:
        return np.array([
            *(q*np.sin(2*omega) + q_*np.sin(-omega)) / np.sin(omega),
            *t + dt, *q, *dt]) + noise
    return np.array([*q, *t + dt, *q, *dt]) + noise


def observation_function(state, noise):
    return state[:7] + noise


initial_state = np.array([1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])  # Identity quaternion & zero translation
initial_covariance = np.eye(STATE_DIMS) * 1.0  # Small initial uncertainty

transition_covariance = np.eye(STATE_DIMS) * 5.0  # Process noise
observation_covariance = OBS_COV_HIGH  # Measurement noise (quaternion & translation only)


ukf = UnscentedKalmanFilter(
    transition_functions=transition_function,
    observation_functions=observation_function,
    transition_covariance=transition_covariance,
    observation_covariance=observation_covariance,
    initial_state_mean=initial_state,
    initial_state_covariance=initial_covariance
)
