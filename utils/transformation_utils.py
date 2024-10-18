import numpy as np
import cv2
from .bbox_utils import measure_distance

world_points = np.array([
    [0, 0],         # Bottom-left corner of the court
    [8.23, 0],      # Bottom-right corner of the court
    [8.23, 23.77],  # Top-right corner of the court
    [0, 23.77],     # Top-left corner of the court
], dtype='float32')

def get_image_width_height(image):

    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_h, original_w = img_rgb.shape[:2]
    return original_h, original_w

def get_closest_keypoints_to_edges(keypoints, image_width, image_height):
    # Separate keypoints into (x, y) pairs
    keypoints = np.array(keypoints).reshape(-1, 2)

    # Initialize variables to store the closest keypoints and their distances
    closest_to_top_left = None
    closest_to_top_right = None
    closest_to_bottom_left = None
    closest_to_bottom_right = None

    min_dist_to_top_left = float('inf')
    min_dist_to_top_right = float('inf')
    min_dist_to_bottom_left = float('inf')
    min_dist_to_bottom_right = float('inf')
    
    # Loop over each keypoint
    for (x, y) in keypoints:
        top_left = (0, 0)
        top_right = (image_width, 0)
        bottom_left = (0, image_height)
        bottom_right = (image_width, image_height)

        # Calculate distances to each corner
        dist_to_top_left = measure_distance((x, y), top_left)
        dist_to_top_right = measure_distance((x, y), top_right)
        dist_to_bottom_left = measure_distance((x, y), bottom_left)
        dist_to_bottom_right = measure_distance((x, y), bottom_right)

        # Check if this point is closer to the top left corner
        if dist_to_top_left < min_dist_to_top_left:
            min_dist_to_top_left = dist_to_top_left
            closest_to_top_left = (x, y)

        # Check if this point is closer to the top right corner
        if dist_to_top_right < min_dist_to_top_right:
            min_dist_to_top_right = dist_to_top_right
            closest_to_top_right = (x, y)

        # Check if this point is closer to the bottom left corner
        if dist_to_bottom_left < min_dist_to_bottom_left:
            min_dist_to_bottom_left = dist_to_bottom_left
            closest_to_bottom_left = (x, y)

        # Check if this point is closer to the bottom right corner
        if dist_to_bottom_right < min_dist_to_bottom_right:
            min_dist_to_bottom_right = dist_to_bottom_right
            closest_to_bottom_right = (x, y)

    # Return the closest keypoints and their distances to each corner
    return {
        'top_left': (closest_to_top_left, min_dist_to_top_left),
        'top_right': (closest_to_top_right, min_dist_to_top_right),
        'bottom_left': (closest_to_bottom_left, min_dist_to_bottom_left),
        'bottom_right': (closest_to_bottom_right, min_dist_to_bottom_right)
    }

