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


class ParticleFilter:
    """ Particle Filter for 3D pose tracking (rx, ry, rz, tx, ty, tz). """
    def __init__(self, num_particles=500, process_noise=None, measurement_noise=None):
        self.num_particles = num_particles

        # State: [rx, ry, rz, tx, ty, tz]
        self.particles = np.zeros((num_particles, 6))

        # Initialize particles randomly (small noise around origin)
        self.particles[:, :3] = np.random.uniform(-0.1, 0.1, (num_particles, 3))  # Rotation noise
        self.particles[:, 3:] = np.random.uniform(-0.1, 0.1, (num_particles, 3))  # Translation noise

        # Weights initialized uniformly
        self.weights = np.ones(num_particles) / num_particles

        # Default noise levels
        self.process_noise = process_noise if process_noise is not None else np.array(
            [0.05, 0.05, 0.05, 0.1, 0.1, 0.1])
        self.measurement_noise = measurement_noise if measurement_noise is not None else np.array(
            [0.01, 0.01, 0.01, 0.05, 0.05, 0.05])

    def transition(self):
        """ Motion model: Applies process noise to simulate movement. """
        noise = np.random.normal(0, self.process_noise, self.particles.shape)
        self.particles += noise  # Apply noise to all particles

    def observe(self, measurement, continuous=True):
        """ Measurement update: Updates particle weights based on observation likelihood. """
        if measurement is None:
            return

        if not continuous:
            self.sample(measurement)

        # Compute error between particles and the measured pose
        diff = self.particles - measurement

        # Normalize angles to [-π, π]
        diff[:, :3] = (diff[:, :3] + np.pi) % (2 * np.pi) - np.pi

        # Gaussian likelihood function
        likelihood = np.exp(-0.5 * np.sum((diff / self.measurement_noise) ** 2, axis=1))

        # Update weights
        self.weights = likelihood
        self.weights += 1e-300  # Avoid zero weights
        self.weights /= np.sum(self.weights)  # Normalize

    def sample(self, measurement):
        for i in range(len(measurement)):
            self.particles[:, i] = np.random.normal(measurement[i], self.measurement_noise[i], self.num_particles)

    def resample(self):
        """ Resamples particles based on importance weights (low variance resampling). """
        indices = np.random.choice(self.num_particles, self.num_particles, p=self.weights)
        self.particles = self.particles[indices]
        self.weights = np.ones(self.num_particles) / self.num_particles  # Reset weights

    def estimate(self):
        """ Estimates the best pose (weighted mean). """
        return np.average(self.particles, axis=0, weights=self.weights)
