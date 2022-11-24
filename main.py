import cv2

import tracking
from model import AppearanceModel
from model.siamfc import SiamFC
from depth import *
import argparse
import yaml
import model_config
import numpy as np
import torch

from orientation import tracked_center_displacement


def load_video(path):
    frames = []
    if not (path[path.index('.') + 1:] == 'mp4' or (path[path.index('.') + 1:] == 'avi')):
        print("WRONG VIDEO FORMAT, PLEASE RETRY")
        return -1
    video = cv2.VideoCapture(path)
    while True:
        ret, frame = video.read()
        if not ret:
            break
        frames.append(frame)
    return frames


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str)
    args = parser.parse_args()

    with open('model_config/imagenet_resnet18_s3.yaml') as f:
        common_args = yaml.load(f)
    for k, v in common_args['common'].items():
        setattr(args, k, v)
    for k, v in common_args['sot'].items():
        setattr(args, k, v)
    model = SiamFC(base=AppearanceModel(args).to(args.device))
    model.eval()
    net = model.cuda()

    tracker = tracking.ObjectTracking(model)

    video = load_video(args.input)
    if video != -1:
        for frame in video:
            # track object & extract tracking center
            track_center_x, track_center_y = tracker.target_segmentation(frame, args)
            # feed tracking center to depth computer network
            depth_map = predict_depth(frame)
            depth = depth_map[track_center_x, track_center_y]
            # use depth to calculate steering angle
            angle = tracked_center_displacement(track_center_x, depth, frame.shape, intrinstic)
            # waste identification
            waste_img =
