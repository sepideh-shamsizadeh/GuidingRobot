import numpy as np
from scipy.stats import multivariate_normal


def unscented_kalman_filter(state_estimate, covariance_matrix, measurement, motion_model, measurement_model,
                            sigma_points, weights):
    n = state_estimate.shape[0]

    # Predict
    sigma_points_pred = np.zeros((2 * n + 1, n))
    for i in range(2 * n + 1):
        sigma_points_pred[i, :] = motion_model(sigma_points[i, :])
    state_estimate_pred, covariance_matrix_pred = estimate_mean_covariance(sigma_points_pred, weights)

    # Update
    residual = measurement - measurement_model(state_estimate_pred)
    sigma_points_residual = np.zeros((2 * n + 1, n))
    for i in range(2 * n + 1):
        sigma_points_residual[i, :] = measurement_model(sigma_points_pred[i, :]) - measurement_model(
            state_estimate_pred)
    state_estimate_residual, covariance_matrix_residual = estimate_mean_covariance(sigma_points_residual, weights)
    cross_covariance = np.zeros((n, n))
    for i in range(2 * n + 1):
        cross_covariance += weights[i] * np.outer(sigma_points_pred[i, :] - state_estimate_pred,
                                                  sigma_points_residual[i, :] - state_estimate_residual)

    regularization_constant = 1e-6  # a small positive constant, e.g. 1e-6

    try:
        covariance_matrix_inverse = np.linalg.inv(covariance_matrix_residual)
    except np.linalg.LinAlgError:
        covariance_matrix_inverse = regularized_covariance_inverse(covariance_matrix_residual, regularization_constant)

    kalman_gain = np.dot(cross_covariance, covariance_matrix_inverse)
    state_estimate = state_estimate_pred + np.dot(kalman_gain, residual)
    covariance_matrix = covariance_matrix_pred - np.dot(kalman_gain, np.dot(covariance_matrix_residual, kalman_gain.T))

    return state_estimate, covariance_matrix


def regularized_covariance_inverse(covariance_matrix, regularization_constant):
    covariance_matrix = covariance_matrix + regularization_constant * np.eye(covariance_matrix.shape[0])
    return np.linalg.inv(covariance_matrix)


def estimate_mean_covariance(sigma_points, weights):
    state_estimate = np.zeros(sigma_points.shape[1])
    covariance_matrix = np.zeros((sigma_points.shape[1], sigma_points.shape[1]))
    for i in range(2 * n + 1):
        state_estimate += weights[i] * sigma_points[i, :]
        covariance_matrix += weights[i] * np.outer(sigma_points[i, :] - state_estimate,
                                                   sigma_points[i, :] - state_estimate)
    return state_estimate, covariance_matrix


# Define the motion model
def motion_model(state):
    return state + np.array([1, 1])


# Define the measurement model
def measurement_model(state):
    return np.array([state[0], state[1]])


# Initial state estimate
state_estimate = np.array([0, 0])
covariance_matrix = np.array([[1, 0], [0, 1]])

# Define sigma points and weights
alpha = 0.1
kappa = 0
beta = 2
n = state_estimate.shape[0]
lambda_ = alpha ** 2 * (n + kappa) - n
weights = np.zeros(2 * n + 1)
weights[0] = lambda_ / (n + lambda_)
weights[1:] = 0.5 / (n + lambda_)
sigma_points = np.zeros((2 * n + 1, n))
sigma_points[0, :] = state_estimate
for i in range(1, n + 1):
    sigma_points[i, :] = state_estimate + np.sqrt(n + lambda_) * np.sqrt(covariance_matrix[i - 1, i - 1])
    sigma_points[i + n, :] = state_estimate - np.sqrt(n + lambda_) * np.sqrt(covariance_matrix[i - 1, i - 1])

# Define the measurements
measurements = np.array([[1, 1], [2, 2], [3, 3]])

# Perform the unscented Kalman filter
for i in range(measurements.shape[0]):
    state_estimate, covariance_matrix = unscented_kalman_filter(state_estimate, covariance_matrix, measurements[i, :],
                                                                motion_model, measurement_model, sigma_points, weights)

print("Final state estimate: ", state_estimate)
