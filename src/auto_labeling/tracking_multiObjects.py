import yaml
import numpy as np
from scipy.spatial.distance import mahalanobis
from filterpy.kalman import UnscentedKalmanFilter
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import MerweScaledSigmaPoints

import math


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


def handle_loss_of_id(filter_i):
    # Handle the case when an object loses its ID association
    filter_i.object_id = None
    reset_filter_state(filter_i)


def get_new_object_id():
    # Generate a new object ID
    global current_object_id
    new_object_id = current_object_id
    current_object_id += 1
    return new_object_id

def handle_new_id_assignment(filter_i):
    # Handle the case when a new ID needs to be assigned to an object
    filter_i.object_id = get_new_object_id()
    reset_filter_state(filter_i)


def reset_filter_state(filter_i):
    # Reset the filter state and covariance matrix
    filter_i.x = np.zeros(filter_i.dim_x)
    filter_i.P = np.eye(filter_i.dim_x) * 1.0


# Set the parameters
num_states = 4  # Number of states (x, y, vx, vy)
num_measurements = 2  # Number of measurements (position)
process_noise_variance = 0.01  # Process noise variance
measurement_noise_variance = 0.1  # Measurement noise variance
dt = 0.1  # Time step
loss_association_threshold = 3  # Number of consecutive frames without association to consider loss of measurement association

# Define the process and measurement noise covariance matrices
Q = np.eye(num_states) * process_noise_variance
R = np.eye(num_measurements) * measurement_noise_variance

# Set initial state and covariance matrix
initial_covariance = np.eye(num_states) * 1.0  # Initial covariance matrix

# Create filters for each object
filters = []

with open('/home/sepid/workspace/Thesis/GuidingRobot/data2/output0.yaml', 'r') as file:
    data = yaml.safe_load(file)


tracks = {}
for frame, frame_data in data.items():
    # Get measurements for the current frame
    measurements = []
    for person_data in frame_data:
        person_id = list(person_data.keys())[0]
        position = np.array([person_data[person_id]['x'], person_data[person_id]['y']])
        measurements.append(position)

    frame_tracks = {}
    if len(filters) == 0:
        for object_id, person in enumerate(measurements):
            filter_i = UnscentedKalmanFilter(dim_x=num_states, dim_z=num_measurements, dt=dt,
                                             fx=state_transition_fn, hx=measurement_fn,
                                             points=MerweScaledSigmaPoints(num_states, alpha=0.1, beta=2.,
                                                                           kappa=-1.0))

            # Set initial state and covariance matrix
            filter_i.x = [person[0], person[1], 0, 0]
            filter_i.P = initial_covariance
            filter_i.dim_x = num_states

            # Set process and measurement noise covariance matrices
            filter_i.Q = Q
            filter_i.R = R

            # Set object ID
            filter_i.object_id = object_id
            current_object_id = object_id

            # Initialize loss of measurement association counter
            filter_i.loss_association_counter = 0

            filters.append(filter_i)
    else:
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
                    filter_i.update(measurement)
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
                handle_loss_of_id(filter_i)

            if filter_i.object_id is None:
                # Handle new ID assignment
                handle_new_id_assignment(filter_i)

            # Store the tracks for the current frame
            object_id = filter_i.object_id
            state = filter_i.x.tolist()
            frame_tracks[object_id] = state

        # Add the frame tracks to the overall tracks dictionary
        tracks[frame] = frame_tracks

# Save tracks to a YAML file
with open('tracks.yaml', 'w') as f:
    yaml.dump(tracks, f)

print("Tracks saved to 'tracks.yaml'", tracks)