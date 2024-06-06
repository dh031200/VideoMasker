# SPDX-FileCopyrightText: 2024-present dh031200 <imbird0312@gmail.com>
#
# SPDX-License-Identifier: MIT
from collections import defaultdict
from pathlib import Path

import numpy as np
import cv2


class Cropper:
    def __init__(self, classes=None, capture_thresh=30, save_dir="cropped_images"):
        self.classes = classes
        self.names = {0: "face"}
        self.capture_thresh = capture_thresh
        self.captured = defaultdict(bool)
        self.save_dir = save_dir
        self.mkdir()

    def __call__(self, tracker, image=None):
        tracks = tracker.active_tracks
        cropped_images = []
        if image is not None:
            cropped_images = self.crop(tracks=tracks, image=image)

        return cropped_images

    def mkdir(self):
        path = Path(self.save_dir)
        path.mkdir(parents=True, exist_ok=True)

    def crop(self, tracks, image):
        cropped_images = []
        for track in tracks:
            tid = track.track_id
            cls = track.cls
            cls_filter = self.classes is None or (track.cls in self.classes)

            if all(
                [
                    not self.captured[tid],
                    track.tracklet_len > self.capture_thresh,
                    cls_filter,
                ]
            ):
                left, top, right, bottom = map(int, track.tlbr)
                cropped_image = image[top:bottom, left:right]
                width, height = right - left, bottom - top

                margin = [np.abs(height - width) // 2, np.abs(height - width) // 2]

                # 부족한 길이가 절반으로 안 떨어질 경우 +1
                if np.abs(height - width) % 2 != 0:
                    margin[0] += 1

                # 가로, 세로 가운데 부족한 쪽에 margin 추가
                if height < width:
                    margin_list = [margin, [0, 0]]
                else:
                    margin_list = [[0, 0], margin]

                # color 이미지일 경우 color 채널 margin 추가
                if len(cropped_image.shape) == 3:
                    margin_list.append([0, 0])

                # 이미지에 margin 추가
                output = np.pad(cropped_image, margin_list, mode="constant")

                cv2.imwrite(f"{self.save_dir}/{tid}_{self.names[cls]}.png", output)
                cropped_images.append(f"{self.save_dir}/{tid}_{self.names[cls]}.png")
                self.captured[tid] = True
        return cropped_images


__all__ = ("Cropper",)
