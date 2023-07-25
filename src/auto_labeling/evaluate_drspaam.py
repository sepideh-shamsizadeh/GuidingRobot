import yaml
import math
import numpy as np
from scipy.spatial.distance import cdist, mahalanobis
import matplotlib.pyplot as plt


# Distance function to calculate Euclidean distance between two positions

def global_nearest_neighbor(reference_points, query_points):
    distances = cdist(reference_points, query_points, lambda u, v: mahalanobis(u, v, covariance_matrix))
    nearest_indices = np.argmin(distances)
    if distances[nearest_indices][0] < 0.7:
        # print(reference_points)
        # print(query_points)
        # print(distances)
        # print(nearest_indices)
        # print(distances[nearest_indices])
        return True, nearest_indices
    else:
        return False, -1

def calculate_distance(pos1, pos2):
    x1, y1 = pos1[0], pos1[1]
    x2, y2 = pos2[0], pos2[1]
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

path1 = '/home/sepid/workspace/Thesis/GuidingRobot/data2/groundtruth.yaml'
path2 = '/home/sepid/workspace/Thesis/GuidingRobot/src/auto_labeling/2D_lidar_person_detection/dr_spaam/dr.yaml'
# Read the YAML files
# path2 = '/home/sepid/workspace/Thesis/GuidingRobot/data2/output0.yaml'
with open(path2) as file1, open(path1) as file2:
    data1 = yaml.safe_load(file1)
    data2 = yaml.safe_load(file2)

# Threshold for matching positions (adjust as needed)
distance_threshold = 0.5

# Variables to store true positives, false positives, and false negatives
true_positives = 0
false_positives = 0
false_negatives = 0
id = 0
gtt = 0
# Iterate over the frames in data2 (ground truth)
for frame_label, frame2 in data2.items():
    frame_num = int(frame_label.split()[1])
    print(f"Frame {frame_num}:")
    frame1 = data1.get(frame_label, [])  # Retrieve frame1 if it exists, otherwise an empty list

    # Iterate over the IDs in ground truth (frame2)
    gt = []
    gt_id = []
    for ground_truth in frame2:
        person_id = list(ground_truth.keys())[0]
        gt_id.append(person_id)
        ground_truth_pos = np.array([ground_truth[person_id]['x'], ground_truth[person_id]['y']])
        gt.append(ground_truth_pos)
        gtt +=1
    # print(gt)
        # Iterate over the IDs in data1 to find a match based on distance
    pp = []
    pp_id = []
    # print(len(frame1))
    for data_pos in frame1:
        person_id = list(data_pos.keys())[0]
        pp_id.append(person_id)
        position = np.array([data_pos[person_id]['x'], data_pos[person_id]['y']])
        pp.append(position)
    # print(pp)

    for i, p in enumerate(pp):
        matched=False
        idd = False
        covariance_matrix = np.array([[1, 0], [0, 1]])
        # If a match is found based on distance threshold
        gtr = np.array(gt)
        matched, ind = global_nearest_neighbor(gtr, [p])
        if ind > -1:
            if ind < len(pp_id) and ind < len(gt_id):
                if gt_id[ind] != pp_id[i]:
                    print(gt_id[ind])
                    print(pp_id[i])
                    id += 1

        # Assign true positive or false negative based on match
        if matched:
            true_positives += 1
        else:
            false_positives += 1

    # Iterate over the remaining IDs in data1 for false positives
    for g in gt:
        unmatched = True
        covariance_matrix = np.array([[1, 0], [0, 1]])
        # If a match is found based on distance threshold
        position_d = np.array(pp)
        ans, _ = global_nearest_neighbor(position_d, [g])
        if ans:
            unmatched = False

        # Assign false positive if there is no match in ground truth
        if unmatched:
            print(frame_label,'ff')
            false_negatives += 1

# Print the results
print(f"True Positives: {true_positives}")
print(f"False Positives: {false_positives}")
print(f"False Negatives: {false_negatives}")
print('Gt', gtt)
print('ID', id)



tr = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
true_positives = [281, 275, 273, 270, 248, 231, 222, 200, 173]

plt.plot(tr, true_positives, marker='o')
plt.xlabel('tr')
plt.ylabel('True Positives')
plt.title('True Positives vs. tr')
plt.grid(True)
plt.show()

# Threshold values
thresholds = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

# False positives
false_positives = [40171, 764, 608, 513, 440, 399, 354, 313, 266]

# Plotting
plt.plot(thresholds, false_positives, marker='o')
plt.xlabel('Threshold (tr)')
plt.ylabel('False Positives')
plt.title('False Positives based on Threshold')
plt.grid(True)
plt.show()





tr_values = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
false_negatives = [227, 543, 572, 593, 611, 627, 638, 659, 684]

plt.plot(tr_values, false_negatives, marker='o')
plt.xlabel('tr')
plt.ylabel('False Negatives')
plt.title('False Negatives based on tr')
plt.grid(True)
plt.show()
