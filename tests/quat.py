import numpy as np
from scipy.spatial.transform import Rotation

from utils.general import DEG2RAD, RAD2DEG


q_ = Rotation.from_euler('xyz', np.array((0, 0, 0))*DEG2RAD).as_quat()
q = Rotation.from_euler('xyz', np.array((0, 0, 5))*DEG2RAD).as_quat()

omega = np.arccos(np.dot(q_, q) / (np.linalg.norm(q_) * np.linalg.norm(q)))
q_new = (q*np.sin(2*omega) + q_*np.sin(-omega)) / np.sin(omega)
print(Rotation.from_quat(q_new).as_euler('xyz')*RAD2DEG)
