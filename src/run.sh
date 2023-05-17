#!/bin/bash

# Set the path to the folder of images
IMAGE_DIR="auto_labeling/calib/scenes/left"

# Loop through all the files in the folder
for IMAGE_FILE in "$IMAGE_DIR"/*; do
    # Run the Python script on each file
    python3 auto_labeling/utils/board_pixels.py "$IMAGE_FILE"
done
