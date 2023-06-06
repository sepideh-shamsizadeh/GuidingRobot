import numpy as np
import yaml
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
from scipy.spatial.distance import mahalanobis


# Define the state transition function
def state_transition_fn(x, dt):
    # Implement the state transition function
    # x: current state vector [x, y, vx, vy]
    # dt: time step
    # Return the predicted state vector
    # Example: simple constant velocity model
    return np.array([x[0] + x[2] * dt, x[1] + x[3] * dt, x[2], x[3]])


# Define the measurement function
def measurement_fn(x):
    # Implement the measurement function
    # x: current state vector [x, y, vx, vy]
    # Return the measurement vector
    # Example: position measurement
    return x[:2]


# Set the parameters
num_states = 4  # Number of states (x, y, vx, vy)
num_measurements = 2  # Number of measurements (position)
process_noise_variance = 0.01  # Process noise variance
measurement_noise_variance = 0.1  # Measurement noise variance
object_ids = [1, 2, 3]  # List of object IDs
dt = 0.1  # Time step
loss_association_threshold = 3  # Number of consecutive frames without association to consider loss of measurement association

# Define the process and measurement noise covariance matrices
Q = np.eye(num_states) * process_noise_variance
R = np.eye(num_measurements) * measurement_noise_variance

# Set initial state and covariance matrix
initial_state = np.array([0, 0, 0, 0])  # Initial state vector [x, y, vx, vy]
initial_covariance = np.eye(num_states) * 1.0  # Initial covariance matrix

# Create filters for each object
filters = []
for object_id in object_ids:
    # Create an UnscentedKalmanFilter instance
    filter_i = UnscentedKalmanFilter(dim_x=num_states, dim_z=num_measurements, dt=dt,
                                     fx=state_transition_fn, hx=measurement_fn,
                                     points=MerweScaledSigmaPoints(num_states, alpha=0.1, beta=2., kappa=-1.0))

    # Set initial state and covariance matrix
    filter_i.x = initial_state
    filter_i.P = initial_covariance

    # Set process and measurement noise covariance matrices
    filter_i.Q = Q
    filter_i.R = R

    # Set object ID
    filter_i.object_id = object_id

    # Initialize loss of measurement association counter
    filter_i.loss_association_counter = 0

    filters.append(filter_i)

# Measurements for each frame
frames = [
    [[2.1, 1.9], [4.2, 3.8]],  # Frame 1: Measurements for objects 1 and 2
    [[6.3, 5.7], [8.4, 7.6], [10.5, 9.5]],  # Frame 2: Measurements for objects 1, 2, and 3
    [[7.3, 5.8], [9.4, 7.9]],  # Add measurements for more frames here
]

# Tracks dictionary to store the results
tracks = {}

# Main loop
for frame_idx, measurements in enumerate(frames):
    frame_tracks = {}

    # Predict the next state for each object
    for filter_i in filters:
        filter_i.predict()

    # Associate measurements using nearest neighbor algorithm
    for measurement in measurements:
        for filter_i in filters:
            # Calculate the Mahalanobis distance between the measurement and predicted measurement
            predicted_measurement = filter_i.hx(filter_i.x)
            distance = mahalanobis(measurement, predicted_measurement, filter_i.S)

            # Choose the filter with the minimum distance
            if distance < 3.0:  # Adjust the threshold as needed
                # Associate the measurement with the filter
                filter_i.update(np.array(measurement))
                filter_i.loss_association_counter = 0  # Reset loss of association counter
                break  # Move to the next measurement
        else:
            # Increment loss of association counter if no association found
            for filter_i in filters:
                filter_i.loss_association_counter += 1

    # Handle loss of ID and new ID assignments
    for filter_i in filters:
        if filter_i.loss_association_counter >= loss_association_threshold:
            # Handle loss of ID
            filter_i.handle_loss_of_id()

        if filter_i.object_id is None:
            # Handle new ID assignment
            filter_i.handle_new_id_assignment()

    # Store the tracks for the current frame
    for filter_i in filters:
        object_id = filter_i.object_id
        state = filter_i.x.tolist()

        frame_tracks[object_id] = state

    # Add the frame tracks to the overall tracks dictionary
    tracks[frame_idx] = frame_tracks

# Save tracks to a YAML file
with open('tracks.yaml', 'w') as f:
    yaml.dump(tracks, f)

print("Tracks saved to 'tracks.yaml'")
