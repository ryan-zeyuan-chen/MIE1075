import cv2
import numpy as np
import torch


def crop_frame(frame, bounding_box, output_size):
    ma = np.array([[(output_size - 1) / (bounding_box[2] - bounding_box[0]), 0,
                    -(output_size - 1) / (bounding_box[2] - bounding_box[0]) * bounding_box[0]],
                   [0, (output_size - 1) / (bounding_box[3] - bounding_box[1]),
                    -(output_size - 1) / (bounding_box[3] - bounding_box[1]) * bounding_box[1]]]).astype(np.float)
    return np.transpose(
        cv2.warpAffine(frame, ma, (output_size, output_size), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)),
        (2, 0, 1))


class ObjectTracking(object):

    def __init__(self, model):
        self.model = model
        self.initial_frame = True
        self.search_scale = 3
        self.scale_factor = 1.0275 ** (np.arange(self.search_scale) - self.search_scale // 2)
        self.search_regularization = 0.985
        self.tracked_center = []

        self.curr_patch = None
        self.pos_legacy = 0
        self.size_legacy = 0
        self.window_size = 0
        self.size_max = 0
        self.size_min = 0

    def target_segmentation(self, initial_frame, args):
        track_center_x = -1
        track_center_y = -1

        img = initial_frame.astype(np.float32) / 255.0
        frame = img[:, :, ::-1].copy()
        frame = torch.from_numpy(np.transpose(frame, (2, 0, 1))).float()
        for t, m, s in zip(frame, args.im_mean, args.im_std):
            t.sub_(m)
            t.div_(s)
        frame = frame.permute(1, 2, 0).numpy()
        if self.initial_frame:
            self.initial_frame = False
            try:
                roi_highlighted = cv2.selectROI("CAMERA", initial_frame, False, False)
                pos, size = np.array([roi_highlighted[0] + roi_highlighted[2] / 2 - 1, roi_highlighted[1] + \
                                      roi_highlighted[3] / 2 - 1]), np.array([roi_highlighted[2], roi_highlighted[3]])
                self.size_max = np.minimum(frame.shape[:2], 5 * size)
                self.size_min = np.maximum(0.2 * size, 4)
            except Exception:
                exit()
            self.window_size = size * 5
            bounding_box = np.array([pos[0] - self.window_size[0] / 2, pos[1] - self.window_size[1] / 2, pos[0] + self.window_size[0] / 2,
                                     pos[1] + self.window_size[1] / 2])
            cropped_frame = crop_frame(frame, bounding_box, 520)
            self.model.update(torch.Tensor(np.expand_dims(cropped_frame, axis=0)).cuda(), lr=1)
            #self.tracked_center.append(np.array([pos[0] - size[0] / 2 + 1, pos[1] - size[1] / 2 + 1, size[0], size[1]]))
            self.curr_patch = np.zeros((self.search_scale, cropped_frame.shape[0],
                                   cropped_frame.shape[1], cropped_frame.shape[2]), np.float32)
            self.pos_legacy = pos
            self.size_legacy = size
        else:
            for i in range(self.search_scale):
                self.window_size = (self.scale_factor[i] * 5) * self.size_legacy
                bounding_box = np.array(
                    [self.pos_legacy[0] - self.window_size[0] / 2, self.pos_legacy[1] - self.window_size[1] / 2, self.pos_legacy[0] + self.window_size[0] / 2,
                     self.pos_legacy[1] + self.window_size[1] / 2])
                self.curr_patch[i, :] = crop_frame(frame, bounding_box, 520)
            out = self.model(torch.Tensor(self.curr_patch).cuda())
            search, index = torch.max(out.view(self.search_scale, -1), 1)
            search_best = np.argmax(self.search_regularization * search.data.cpu().numpy())
            r, c = np.unravel_index(index[search_best].cpu(), [out.shape[-2], out.shape[-1]])
            self.window_size = self.size_legacy * (self.scale_factor[search_best] * 5)
            self.pos_legacy = self.pos_legacy + np.array([c - out.shape[-1] * 0.5, r - out.shape[-2] * 0.5]) * self.window_size / [out.shape[-2], out.shape[-1]]
            self.size_legacy = np.minimum(np.maximum(self.window_size / 5, self.size_min), self.size_max)

            tracked = np.array([self.pos_legacy[0] - self.size_legacy[0] / 2 + 1, self.pos_legacy[1] - self.size_legacy[1] / 2 + 1, self.size_legacy[0], self.size_legacy[1]])
            #self.tracked_center.append(tracked)

            # rendering
            bounding_box = list(map(int, tracked))
            cv2.rectangle(initial_frame, (bounding_box[0], bounding_box[1]),
                          (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]), (255, 255, 255), 3)
            cv2.circle(initial_frame, (bounding_box[0] + bounding_box[2] // 2, bounding_box[1] + bounding_box[3] // 2), 5, (0, 0, 255), -1)
            cv2.imshow("CAMERA", initial_frame)
            cv2.waitKey(40)

            track_center_x = bounding_box[0] + bounding_box[2] // 2
            track_center_y =  bounding_box[1] + bounding_box[3] // 2

        return track_center_x, track_center_y


