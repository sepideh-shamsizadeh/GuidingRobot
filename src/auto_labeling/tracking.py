import yaml
import numpy as np
from filterpy.kalman import UnscentedKalmanFilter
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import MerweScaledSigmaPoints

# Load YAML file
with open('/home/sepid/workspace/Thesis/GuidingRobot/data2/output0.yaml', 'r') as file:
    data = yaml.safe_load(file)

# Parameters for the UKF
dt = 1.0  # Time step
std_acc = 0.5  # Standard deviation of acceleration
std_meas = 1.0  # Standard deviation of measurement noise

# State transition function
def fx(x, dt):
    # Constant velocity motion model
    F = np.array([[1, dt, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, dt],
                  [0, 0, 0, 1]])
    return F @ x

# Measurement function
# Measurement function
def hx(x):
    # Measurement is the position (x, y) and velocity (vx, vy)
    return x[[0, 2, 1, 3]]


# Initialize the UKF
dim_state = 4  # Dimension of the state vector
dim_meas = 4  # Dimension of the measurement vector
# Create sigma points for Unscented Kalman Filter
points = MerweScaledSigmaPoints(n=dim_state, alpha=0.1, beta=2.0, kappa=1.0)

# Initialize the list of tracks
tracks = []

# Iterate over frames
for frame, frame_data in data.items():
    # Get measurements for the current frame
    measurements = []
    for person_data in frame_data:
        person_id = list(person_data.keys())[0]
        position = np.array([person_data[person_id]['x'], person_data[person_id]['y'],0,0])
        measurements.append(position)

    # Data association
    if len(tracks) > 0 and len(measurements) > 0:
        num_tracks = len(tracks)
        num_measurements = len(measurements)
        cost = np.zeros((num_tracks, num_measurements))

        for i in range(num_tracks):
            for j in range(num_measurements):
                track = tracks[i]['filter'].x
                cost[i, j] = np.linalg.norm(track - measurements[j])

        assignment = linear_sum_assignment(cost)
        assigned_tracks = assignment[0]
        assigned_measurements = assignment[1]
        unassigned_tracks = np.setdiff1d(range(num_tracks), assigned_tracks)
        unassigned_measurements = np.setdiff1d(range(num_measurements), assigned_measurements)

        # Update tracks with assigned measurements
        for i, j in zip(assigned_tracks, assigned_measurements):
            track = tracks[i]
            track['filter'].x = measurements[j]
            track['filter'].update(measurements[j])
            track['age'] += 1
            track['total_visible_frames'] += 1

        # Create new tracks for unassigned measurements
        for j in unassigned_measurements:
            track = {
                'filter': UnscentedKalmanFilter(dim_x=dim_state, dim_z=dim_meas, dt=dt, fx=fx, hx=hx, points=points),
                'age': 1,
                'total_visible_frames': 1
            }
            track['filter'].x = measurements[j]
            track['filter'].P = np.eye(dim_state) * 10.0
            # print('Track')
            # print('Position:', track['filter'].x)
            # print('Age:', track['age'])
            # print('Frames:', frame)
            # print()
            tracks.append(track)

        # Delete tracks with no assigned measurements for too long
        tracks = [track for track in tracks if track['age'] < 10]

    else:
        # Create new tracks for all measurements
        for measurement in measurements:
            track = {
                'filter': UnscentedKalmanFilter(dim_x=dim_state, dim_z=dim_meas, dt=dt, fx=fx, hx=hx, points=points),
                'age': 1,
                'total_visible_frames': 1
            }
            track['filter'].x = measurement
            track['filter'].P = np.eye(dim_state) * 10.0
            tracks.append(track)

    # Rest of the code...

    # Print the final tracks
    for i, track in enumerate(tracks):
        print('Track', i)
        print('Position:', track['filter'].x)
        print('Age:', track['age'])
        print('Total Visible Frames:', track['total_visible_frames'])
        print()
