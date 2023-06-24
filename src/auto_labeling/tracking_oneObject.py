from filterpy.kalman import UnscentedKalmanFilter
from filterpy.kalman import MerweScaledSigmaPoints
import numpy as np
from scipy.spatial.distance import cdist

# Define the process and measurement functions
def process_model(state, dt):
    # Implement the process model
    # Update the state using the system dynamics
    # Return the updated state
    return state  # Placeholder, modify according to the system dynamics

def measurement_model(state):
    # Implement the measurement model
    # Generate the measurement based on the current state
    # Return the measurement
    return state[:2]  # Return [x, y] coordinates

# Create the Unscented Kalman Filter
dim_state = 4  # Dimension of the state vector
dim_measurement = 2  # Dimension of the measurement vector

points = MerweScaledSigmaPoints(n=dim_state, alpha=0.1, beta=2., kappa=-1)

ukf = UnscentedKalmanFilter(dim_x=dim_state, dim_z=dim_measurement, dt=0.1,
                            hx=measurement_model, fx=process_model, points=points)

# Initialize the state and covariance matrix
initial_state = np.array([1.5, 2, 0, 0])  # Initial state vector [x, y, vx, vy]
initial_covariance = np.eye(dim_state)  # Initial covariance matrix

ukf.x = initial_state
ukf.P = initial_covariance

# Perform the filtering
frames = [
    [[1, 2], [2, 3], [3, 4]],  # Frame 1 with multiple positions
    [[4, 5], [5, 6], [6, 7]]   # Frame 2 with multiple positions
]

for frame in frames:
    positions = np.array(frame)
    # Calculate distances between predicted state and frame positions
    distances = cdist([ukf.x[:2]], positions)

    # Find the index of the nearest neighbor
    nearest_index = np.argmin(distances)

    # Update the state using the nearest neighbor measurement
    nearest_measurement = np.array(positions[nearest_index]).reshape(dim_measurement)

    print(nearest_measurement)
    ukf.predict(dt=0.1)  # Pass the time step dt
    ukf.update(nearest_measurement)

    estimated_state = ukf.x  # Estimated state after each update
    estimated_covariance = ukf.P
    print("Track position:", estimated_state[:2])
