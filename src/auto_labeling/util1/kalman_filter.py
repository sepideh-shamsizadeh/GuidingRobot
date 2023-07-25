import numpy as np
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise


def create_kalman_filter():
    """Create a Kalman filter with 4 state variables (x,y,vx,vy) and constant velocity model."""
    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.F = np.array([[1, 0, 1, 0],
                     [0, 1, 0, 1],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])

    kf.H = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0]])

    kf.R = np.diag([10, 10])

    q_var = 0.01
    kf.Q = Q_discrete_white_noise(dim=4, dt=1.0, var=q_var)
    kf.P *= 1000

    return kf


def jpda_tracker(positions):
    """Performs tracking of persons using JPDA algorithm."""
    # Initialize empty list to store tracks
    tracks = []

    # Initialize empty dictionary to store previous frame data
    prev_frame_data = {}

    # Loop through frames
    for frame_num in sorted(positions.keys()):
        # Initialize cost matrix and lists to store measurements and track predictions
        cost_matrix = np.zeros((len(tracks), len(positions[frame_num])))
        measurements = []
        track_predictions = []

        # Loop through persons in current frame
        for person_id, pos in positions[frame_num].items():
            # Append measurement to list
            if pos is not None:
                measurements.append(np.array(pos).reshape((2, 1)))

            # Create Kalman filter prediction for each track and append to list
            for track in tracks:
                kf = track['kf']
                track_prediction = kf.predict()
                track_predictions.append(track_prediction)

                # Calculate Mahalanobis distance between measurement and track prediction
                if pos is not None:
                    if track_prediction is None:
                        continue
                    innovation = np.array(pos).reshape((2, 1)) - kf.H @ track_prediction.reshape((-1, 1))
                    S = kf.H @ kf.P @ kf.H.T + kf.R
                    md = np.sqrt(innovation.T @ np.linalg.inv(S) @ innovation)[0, 0]

                    # Update cost matrix with Mahalanobis distance
                    cost_matrix[track['id'], person_id] = md

        # Use linear sum assignment to match tracks to measurements
        row_inds, col_inds = linear_sum_assignment(cost_matrix)

        # Loop through matched pairs and update Kalman filter state
        for i in range(len(row_inds)):
            track = tracks[row_inds[i]]
            person_id = col_inds[i]+1
            if 'Person '+str(person_id) in positions[frame_num]:
                pos = positions[frame_num]['Person '+str(person_id)]
                kf = track['kf']
            else:
                pos = None

            if pos is not None:
                # Update Kalman filter with measurement
                z = np.array(pos).reshape((2, 1))
                kf.update(z)

                # Store current position and time step for this track
                track['positions'].append(pos)
                track['times'].append(frame_num)

                # Reset missed frames counter
                track['missed_frames'] = 0

            else:
                # Measurement is lost, update Kalman filter with prediction only
                kf.update(kf.H @ kf.x)

                # Increment missed frames counter
                track['missed_frames'] += 1

        # Loop through unmatched tracks and increment missed frames counter
        unmatched_tracks = set(range(len(tracks))) - set(row_inds)
        for i in unmatched_tracks:
            track = tracks[i]
            track['kf'].predict()
            track['missed_frames'] += 1

        # Loop through unmatched measurements and create new tracks
        unmatched_measurements = set(positions[frame_num].keys()) - set(col_inds)
        for person_id in unmatched_measurements:
            pos = positions[frame_num][person_id]
            if pos is not None:
                # Create new Kalman filter for this measurement
                kf = create_kalman_filter()
                z = np.array(pos).reshape((2, 1))
                kf.update(z)

                # Store track ID, Kalman filter, current position, and time step for this track
                track = {'id': len(tracks), 'kf': kf, 'positions': [pos], 'times': [frame_num], 'missed_frames': 0}
                tracks.append(track)

        # Remove tracks with too many consecutive missed frames
        tracks = [track for track in tracks if track['missed_frames'] < 5]

        # Update previous frame data
        prev_frame_data = positions[frame_num]

    return tracks


positions = {}
with open('test.txt', 'r') as f:
    frame_num = None
    for line in f:
        line = line.strip()
        if line.startswith('Frame'):
            frame_num = int(line.split()[1])
            positions[frame_num] = {}  # Create an empty dictionary for this frame
        else:
            person_id, pos = line.split(': ')
            if pos == 'lost':
                positions[frame_num][person_id] = None
            else:
                x, y = map(int, pos.strip('()').split(', '))
                positions[frame_num][person_id] = (x, y)
tracks = jpda_tracker(positions)

for frame_num in sorted(positions.keys()):
    print(f"Frame {frame_num}:")
    for person_id, pos in positions[frame_num].items():
        if pos is not None:
            x, y = pos
            print(f"{person_id}: ({x}, {y})")
    print()

for track in tracks:
    print(f"Track {track['id']}:")
    for i in range(len(track['positions'])):
        frame_num = track['times'][i]
        x, y = track['positions'][i]
        print(f"Frame {frame_num}: ({x}, {y})")
    print()