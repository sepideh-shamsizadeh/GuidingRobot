import yaml

def load_yaml_file(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data

def compute_id_switches(tracking_output, ground_truth):
    id_switches = 0

    for frame_idx, frame_truth in enumerate(ground_truth):
        frame_output = tracking_output[frame_idx]

        for truth_id, truth_data in frame_truth.items():
            if truth_id in frame_output:
                output_id = frame_output[truth_id]['id']
                if output_id != truth_id:
                    id_switches += 1

    return id_switches

# Load tracking output and ground truth from YAML files
tracking_output = load_yaml_file('tracking_output.yaml')
ground_truth = load_yaml_file('ground_truth.yaml')

num_id_switches = compute_id_switches(tracking_output, ground_truth)
print("Number of ID switches:", num_id_switches)
