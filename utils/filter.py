import numpy as np
from pykalman import UnscentedKalmanFilter
from scipy.spatial.transform import Rotation as R


STATE_DIMS = 12
OBS_DIMS = 6

OBS_COV_LOW = np.eye(OBS_DIMS) * 0.05
OBS_COV_HIGH = np.eye(OBS_DIMS) * 0.5


def transition_function(state, noise):
    # state: [rx, ry, rz,
    #         tx, ty, tz,
    #         rx', ry', rz',
    #         dx, dy, dz]
    r = state[:3]
    t = state[3:6]
    r_ = state[6:9]
    dt = state[9:]
    return np.array([*(r + r_), *(t + dt), *r_, *dt]) + noise


def observation_function(state, noise):
    return state[:OBS_DIMS] + noise


initial_state = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # Identity quaternion & zero translation
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
    """ Particle Filter with Adaptive Diffusion Control for 3D pose tracking (rx, ry, rz, tx, ty, tz). """
    def __init__(self, num_particles=2500, process_noise=None, measurement_noise=None):
        self.num_particles = num_particles

        # State: [rx, ry, rz, tx, ty, tz]
        self.particles = np.zeros((num_particles, 6))

        # Initialize particles randomly (small noise around origin)
        self.particles[:, :3] = np.random.uniform(-0.1, 0.1, (num_particles, 3))  # Rotation noise
        self.particles[:, 3:] = np.random.uniform(-0.1, 0.1, (num_particles, 3))  # Translation noise

        # Weights initialized uniformly
        self.weights = np.ones(num_particles) / num_particles

        # Default noise levels
        self.base_process_noise = process_noise if process_noise is not None else np.array(
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1])  # Base process noise
        self.measurement_noise = measurement_noise if measurement_noise is not None else np.array(
            [0.01, 0.01, 0.01, 0.05, 0.05, 0.05])

        # Adaptive Diffusion Control Parameters
        self.alpha = 0.5  # Controls how aggressively noise adapts
        self.min_noise = 0.01  # Lower bound for process noise
        self.max_noise = 1.0  # Upper bound for process noise

    def compute_adaptive_noise(self):
        """ Adjusts process noise based on particle spread (Adaptive Diffusion Control). """
        spread = np.std(self.particles, axis=0)  # Compute spread of particles

        # Scale process noise adaptively (between min_noise and max_noise)
        adaptive_noise = self.base_process_noise * (1 + self.alpha * (spread / np.mean(spread)))
        self.process_noise = np.clip(adaptive_noise, self.min_noise, self.max_noise)

    def transition(self):
        """ Motion model: Applies adaptive process noise to simulate movement. """
        self.compute_adaptive_noise()  # Update process noise adaptively
        noise = np.random.normal(0, self.process_noise, self.particles.shape)
        self.particles += noise  # Apply noise to all particles

    def observe(self, measurement, continuous=True):
        """ Measurement update: Updates particle weights based on observation likelihood. """
        if measurement is None:
            return

        if not continuous or np.max(np.linalg.norm(self.particles[:, 3:]-measurement[3:], axis=1)) > 2.0:
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
        n_eff = 1.0 / np.sum(self.weights ** 2)  # Effective particle count
        if n_eff < self.num_particles * 0.5:  # Only resample if necessary
            indices = np.random.choice(self.num_particles, self.num_particles, p=self.weights)
            self.particles = self.particles[indices]
            self.weights = np.ones(self.num_particles) / self.num_particles  # Reset weights

    def estimate(self):
        """ Estimates the best pose (weighted mean). """
        return np.average(self.particles, axis=0, weights=self.weights)


class ParticleFilterQuat:
    """ Particle Filter for 3D pose tracking using Quaternions and Translation. """

    def __init__(self, num_particles=500, process_noise_rot=0.05, process_noise_trans=0.2,
                 measurement_noise_rot=0.1, measurement_noise_trans=0.3):
        self.num_particles = num_particles

        # State: [qx, qy, qz, qw, x, y, z]
        self.particles = np.zeros((num_particles, 7))

        # Initialize quaternions (small random rotation)
        random_rotations = R.random(num_particles)
        self.particles[:, :4] = random_rotations.as_quat()  # (qx, qy, qz, qw)

        # Initialize translations with small noise
        self.particles[:, 4:] = np.random.uniform(-0.1, 0.1, (num_particles, 3))

        # Weights initialized uniformly
        self.weights = np.ones(num_particles) / num_particles

        # Process noise (quaternion + translation)
        self.process_noise_rot = process_noise_rot
        self.process_noise_trans = process_noise_trans

        # Explicit Measurement Noise
        self.measurement_noise_rot = measurement_noise_rot  # Rotation noise (radians)
        self.measurement_noise_trans = measurement_noise_trans  # Translation noise (meters)

    def transition(self):
        """ Motion model: Apply random perturbation to rotation and translation. """
        random_rot = R.from_euler('xyz', np.random.normal(0, self.process_noise_rot, (self.num_particles, 3)))
        new_quats = (R.from_quat(self.particles[:, :4]) * random_rot).as_quat()
        self.particles[:, :4] = new_quats  # Update quaternion states

        # Apply translation noise
        self.particles[:, 4:] += np.random.normal(0, self.process_noise_trans, (self.num_particles, 3))

    def observe(self, measurement, continuous=True):
        """ Measurement update: Updates particle weights based on observation likelihood. """
        if measurement is None:
            return

        if not continuous:
            self.sample(measurement)

        # Rotation Measurement Update
        measured_rotation = R.from_quat(measurement[:4])
        particle_rotations = R.from_quat(self.particles[:, :4])
        angular_diffs = particle_rotations.inv() * measured_rotation
        angle_errors = angular_diffs.magnitude()  # Compute angular difference

        # Compute Gaussian likelihood for rotation
        rot_likelihood = np.exp(-0.5 * (angle_errors / self.measurement_noise_rot) ** 2)

        # Translation Measurement Update
        translation_diffs = np.linalg.norm(self.particles[:, 4:] - measurement[4:], axis=1)

        # Compute Gaussian likelihood for translation
        trans_likelihood = np.exp(-0.5 * (translation_diffs / self.measurement_noise_trans) ** 2)

        # Combine Likelihoods
        self.weights = rot_likelihood * trans_likelihood
        self.weights += 1e-300  # Avoid zero weights
        self.weights /= np.sum(self.weights)  # Normalise

    def sample(self, measurement):
        for i in range(4):
            self.particles[:, i] = np.random.normal(measurement[i], self.measurement_noise_rot, self.num_particles)
        for i in range(3):
            self.particles[:, 4+i] = np.random.normal(measurement[4+i], self.measurement_noise_trans, self.num_particles)

    def resample(self):
        """ Low-variance resampling to maintain particle diversity. """
        n_eff = 1.0 / np.sum(self.weights ** 2)  # Effective sample size
        if n_eff < self.num_particles / 2:  # Only resample if necessary
            indices = np.random.choice(self.num_particles, self.num_particles, p=self.weights)
            self.particles = self.particles[indices]
            self.weights = np.ones(self.num_particles) / self.num_particles  # Reset weights

    def estimate(self):
        """ Estimate the best pose (weighted mean). """
        avg_quat = R.from_quat(self.particles[:, :4]).mean().as_quat()  # Average quaternion
        avg_trans = np.average(self.particles[:, 4:], axis=0, weights=self.weights)  # Weighted translation
        return np.hstack((avg_quat, avg_trans))
