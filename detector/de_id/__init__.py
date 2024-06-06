# SPDX-FileCopyrightText: 2024-present dh031200 <imbird0312@gmail.com>
#
# SPDX-License-Identifier: MIT
import cv2
import numpy as np


class DeId:
    def __init__(self, method="blur", exclude_face_ids=None):
        self.exclude_face_ids = [] if exclude_face_ids is None else exclude_face_ids
        if method == "blur":
            self.method = self.blur
        elif method == "pixelate":
            self.method = self.mosaic
        else:
            self.method = self.blur

    def __call__(self, detections, image, strength_factor=1):
        return self.de_id(detections, image, strength_factor=strength_factor)

    def de_id(self, detections, image, strength_factor=1):
        for det in detections:
            if int(det[6]) in self.exclude_face_ids:
                continue
            left, top, right, bottom = map(int, det[:4])
            canvas = np.zeros_like(image[top:bottom, left:right])
            h, w, _ = canvas.shape
            strength = int((((h ** 2) + (w ** 2)) ** 0.5) / 10) * strength_factor
            cv2.ellipse(canvas, ((w // 2, h // 2), (w, h), 0), (255, 255, 255), -1)
            a = cv2.bitwise_and(image[top:bottom, left:right], cv2.bitwise_not(canvas))
            b = cv2.bitwise_and(self.method(image[top:bottom, left:right], strength=strength), canvas)
            image[top:bottom, left:right] = a + b

    @staticmethod
    def mosaic(image, strength):
        strength = strength // 2
        h, w, _ = image.shape
        image = cv2.resize(image, dsize=(w // strength, h // strength))
        image = cv2.resize(image, dsize=(w, h), interpolation=cv2.INTER_AREA)
        return image

    @staticmethod
    def blur(image, strength):
        return cv2.blur(image, (strength, strength), image)

    __all__ = ("DeId",)
