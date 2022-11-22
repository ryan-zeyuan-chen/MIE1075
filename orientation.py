import numpy as np


def tracked_center_displacement(center_x, center_depth, img_shape, camera_intrinsic):
    """
    Return value = angle
    """
    tracked_center_to_depth_x = (center_depth / camera_intrinsic[0][0]) * (img_shape[0] / 2)
    tracked_center_offset_x = center_x - tracked_center_to_depth_x

    angle = np.arctan(tracked_center_offset_x / center_depth)
    return angle