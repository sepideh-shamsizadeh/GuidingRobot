import numpy as np
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints


# Define motion model
def fx(x, dt):
    F = np.array([[1, 0, dt, 0],
                  [0, 1, 0, dt],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])
    return F @ x


# Define measurement function
def hx(x):
    H = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0]])
    return H @ x


# Define JPDA function
def jpda_filter(zs):
    # Initialize variables
    kf_list = []
    prob_list = []
    gate_threshold = 3.0
    score_threshold = 0.5

    # Loop through measurements
    for z in zs:
        # Initialize probabilities
        prob_detection = 0.9
        prob_false_alarm = 0.1

        # Loop through tracked objects
        for i, kf in enumerate(kf_list):

            # Predict step
            kf.predict()

            # Compute gating distance
            S = kf.S + kf.R
            z_diff = z - hx(kf.x)
            mahalanobis = z_diff.T @ np.linalg.inv(S) @ z_diff

            # If measurement is within gating distance, update step
            if mahalanobis < gate_threshold ** 2:
                kf.update(z)
                prob_detection *= kf.likelihood(z, log=True)
                break

            # If measurement is outside gating distance, set detection probability to 0
            else:
                prob_detection *= 1 - kf.likelihood(z, log=True)


        # Combine detection and false alarm probabilities
        prob = prob_detection * prob_false_alarm

        # Normalize probabilities
        for j, _ in enumerate(kf_list):
            prob_j = prob_detection / kf_list[j].likelihood(z, log=True) if j == i else (1 - prob_detection) / (
                        len(kf_list) - 1)
            prob *= prob_j

        prob_list.append(prob)

    # Return filtered measurements
    x_list = [kf.x[:2] for kf in kf_list]
    return x_list, prob_list


# Example usage
if __name__ == '__main__':
    # Generate simulated measurements
    xs = [
        np.array([-3, -3]),
        np.array([5, 3]),
        np.array([-2, 6]),
        np.array([7, 9]),
        np.array([2, 1]),
        np.array([0, 4])
    ]
    zs = [x + np.random.normal(size=2) for x in xs]

    # Filter measurements using JPDA
    x_list, prob_list = jpda_filter(zs)

    # Print results
    print('Filtered objects:')
    for i, x in enumerate(x_list):
        print(f'Object {i + 1}: ({x[0]:.2f}, {x[1]:.2f})')
    print('Probabilities:')
    for i, prob in enumerate(prob_list):
        print(f'Measurement {i + 1}: {prob:.4f}')
