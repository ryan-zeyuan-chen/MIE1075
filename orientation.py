import numpy as np


def tracked_center_displacement(center_x, center_depth, focal_length, sensor_width):
    """
    Return value = angle
    """
    tracked_center_to_depth_x = (center_depth / focal_length) * (sensor_width / 2)
    tracked_center_offset_x = center_x - tracked_center_to_depth_x

    angle = np.arctan(tracked_center_offset_x / center_depth) * (180 / np.pi)
    return angle