from filterpy.kalman import UnscentedKalmanFilter
from filterpy.stats import logpdf
from scipy.linalg import block_diag
from scipy.optimize import linear_sum_assignment
import numpy as np

class JPDAFilter:
    def __init__(self, init_x, init_P, motion_model, observation_model,
                 clutter_rate, prob_detection, prob_false_alarm):
        self.x = init_x
        self.P = init_P
        self.motion_model = motion_model
        self.observation_model = observation_model
        self.clutter_rate = clutter_rate
        self.prob_detection = prob_detection
        self.prob_false_alarm = prob_false_alarm

    def update(self, z):
        # Predict step
        fx, P, Q = self.motion_model(self.x, self.P)
        P += Q
        # Update step
        H, R = self.observation_model(z)
        S = np.dot(np.dot(H, P), H.T) + R
        Sinv = np.linalg.inv(S)
        K = np.dot(np.dot(P, H.T), Sinv)
        y = z - np.dot(H, fx)
        self.x = fx + np.dot(K, y)
        self.P = P - np.dot(np.dot(K, H), P)
        # Compute association probabilities
        D, N = H.shape[0], fx.shape[0] // D
        z_pred = np.array([self.observation_model(fx[N*i:N*(i+1), :])[0] for i in range(D)])
        probs = np.zeros((D, z.shape[1]+1))
        probs[:, -1] = self.clutter_rate
        for i in range(z.shape[1]):
            dz = z[:, i].reshape((-1, 1)) - z_pred
            probs[:, i] = self.prob_detection * np.exp(logpdf(dz, Sinv) - 0.5*D*np.log(2*np.pi) + 0.5*np.log(np.linalg.det(S)))
        probs = np.vstack((probs, 1-self.prob_false_alarm*np.ones((1, z.shape[1]+1))))
        # JPDA
        C = np.hstack((probs, self.prob_detection*self.clutter_rate*np.ones((D+1, 1))))
        C = C / np.sum(C, axis=1, keepdims=True)
        row_ind, col_ind = linear_sum_assignment(-C)
        associations = []
        for i, j in zip(row_ind, col_ind):
            if i < D and j < z.shape[1]:
                associations.append(j)
            else:
                associations.append(None)
        return associations
