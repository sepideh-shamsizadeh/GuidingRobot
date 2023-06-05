import yaml
import numpy as np
from scipy.linalg import block_diag
from filterpy.kalman import UnscentedKalmanFilter
from filterpy.kalman import JulierSigmaPoints
from filterpy.common import Q_discrete_white_noise


def load_data_from_yaml(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data


def unscented_kalman_filter(z, R, Q):
    def fx(x, dt):
        # State transition function
        # Here, assume a constant velocity motion model
        x[0] += dt * x[2]
        x[1] += dt * x[3]
        return x

    def hx(x):
        # Measurement function
        return np.array([x[0], x[1]])

    # Initialize UKF
    dt = 1.0  # Time step
    points = JulierSigmaPoints(n=4, kappa=2)  # Sigma points generator
    ukf = UnscentedKalmanFilter(dim_x=4, dim_z=2, dt=dt, fx=fx, hx=hx, points=points)

    # Initial state and covariance
    ukf.x = np.array([z[0], z[1], 0., 0.])  # [x, y, vx, vy]
    ukf.P = np.eye(4) * 0.1

    # Process noise covariance
    ukf.Q = Q_discrete_white_noise(dim=2, dt=dt, var=Q)

    # Measurement noise covariance
    ukf.R = np.diag([R, R])

    # Perform filtering
    filtered_states = []
    for measurement in z[2:]:
        ukf.predict()
        ukf.update(measurement)
        filtered_states.append(ukf.x)

    return np.array(filtered_states)


def nearest_neighbor_association(measurements, tracks, gate_threshold):
    associations = []
    for i, measurement in enumerate(measurements):
        min_distance = float('inf')
        best_track_id = -1
        for j, track in enumerate(tracks):
            distance = np.linalg.norm(measurement - track)
            if distance < min_distance and distance < gate_threshold:
                min_distance = distance
                best_track_id = j
        associations.append(best_track_id)
    return associations


# Load data from YAML
data = load_data_from_yaml('data.yaml')

# Extract measurements
measurements = []
for frame in data.values():
    for obj in frame:
        for id_, position in obj.items():
            measurements.append([position['x'], position['y']])

# Parameters for the Unscented Kalman Filter
R = 0.1  # Measurement noise covariance
Q = 0.01  # Process noise covariance

# Perform tracking using Unscented Kalman Filter
filtered_states = unscented_kalman_filter(measurements, R, Q)

# Perform data association using Nearest Neighbor
gate_threshold = 1.0  # Association gate threshold
associations = nearest_neighbor_association(measurements, filtered_states, gate_threshold)

# Print the results
for i, association in enumerate(associations):
    measurement = measurements[i]
    filtered_state = filtered_states[association]
    print(f"Measurement {i} - True: {measurement}, Estimated: {filtered_state}")
