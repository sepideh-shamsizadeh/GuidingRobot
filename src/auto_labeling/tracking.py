import numpy as np
import yaml
from filterpy.kalman import UnscentedKalmanFilter
from filterpy.kalman import JulierSigmaPoints
from filterpy.common import Q_discrete_white_noise

def parse_yaml_file(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data

def mahalanobis_distance(x, y, covariance):
    diff = x - y
    inv_cov = np.linalg.inv(covariance)
    distance = np.sqrt(np.dot(np.dot(diff, inv_cov), diff.T))
    return distance

def create_measurement(measurements):
    return measurements[:, 0]

def create_ukf():
    ukf = UnscentedKalmanFilter(dim_x=2, dim_z=2, dt=1.0, points=JulierSigmaPoints(n=2, kappa=1), hx=lambda x: x, fx=lambda x, dt: x)
    ukf.x = np.array([0., 0.])  # initial state
    ukf.P *= 0.1  # initial state covariance
    ukf.R = np.diag([0.1, 0.1])  # measurement noise covariance
    ukf.Q = Q_discrete_white_noise(dim=2, dt=1.0, var=0.1)  # process noise covariance
    return ukf

def track_positions(data, out):
    for frame, objects in data.items():
        ukf = create_ukf()
        data_s = []
        for i,obj_data in enumerate(objects):
            obj_id = 'id'+str(i)
            data = obj_data[obj_id]
            x = data['x']
            y = data['y']
            measurements = np.array([(x, y)])
            ukf.predict()
            ukf.update(create_measurement(measurements))
            position = {
                'x': float(ukf.x[0]), 'y': float(ukf.x[1])
            }
            pp = {obj_id: position}
            data_s.append(pp)

        yaml_data = {frame: data_s}
        with open(out, 'a') as file:
            yaml.dump(yaml_data, file)

# Example usage
input_file = '/home/sepid/workspace/Thesis/GuidingRobot/data/input.yaml'
output_file = 'output.yaml'

# Step 1: Parse the YAML file
data = parse_yaml_file(input_file)

# Step 2 and 3: Track positions using Unscented Kalman Filter and nearest neighbor algorithm
track_positions(data, output_file)
