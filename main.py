import cv2

import tracking
from depth.predict import predict_depth
from model import AppearanceModel
from model.siamfc import SiamFC
from depth import *
from waste import *
import argparse
import yaml
import model_config
import numpy as np
import torch
import matplotlib.pyplot as plt

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

    intrinstic = np.array([[7, 0, 5.4], [0, 7, 9.6], [0, 0, 1]])
    if video != -1:
        f_count = 0
        for frame in video:
            # track object & extract tracking center
            track_center_x, track_center_y = tracker.target_segmentation(frame.copy(), args)
            # feed tracking center to depth computer network
            depth_map = predict_depth(frame)
            try:
                depth = depth_map[track_center_x, track_center_y]
                depth_map = cv2.normalize(depth_map, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
                cv2.imshow("depth", depth_map)
                cv2.waitKey(40)

                # use depth to calculate steering angle
                angle = tracked_center_displacement(track_center_x / 100, depth, (9.6, 5.4), intrinstic)
                print(angle)
            except IndexError as e:
                print(track_center_x)
                print(track_center_y)
                print("!!")
                
            # waste identification
            if f_count > 100:
                run(source= frame, weight ='/MIE1075/waste/yolov5/runs/train/exp3/weights/best.pt', save_txt = True）
                f_count = 0
            else:
                f_count += 1
