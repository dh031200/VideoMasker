# SPDX-FileCopyrightText: 2024-present dh031200 <imbird0312@gmail.com>
#
# SPDX-License-Identifier: MIT
import cv2
import torch
import numpy as np
import supervision as sv
from ultralytics import YOLO
from imgbeddings import imgbeddings
from PIL import Image

from .tracker import Tracker
from .cropper import Cropper
from .utils import to_supervision_detections, make_labels


class Detector:
    def __init__(self, fps, width, height):
        if torch.backends.mps.is_available() and False:
            model_path = "model/yolov8s-oiv7-face.mlpackage"
            device = "mps"
        elif torch.cuda.is_available():
            model_path = "model/yolov8s-oiv7-face.pt"
            device = "cuda"
        else:
            model_path = "model/yolov8s-oiv7-face.pt"
            device = "mps"
        self.model = YOLO(model_path, task="detect")
        self.device = device
        self.tracker = Tracker(method="bytetrack")
        # self.tracker = Tracker(method="SFSORT", fps=fps, width=width, height=height) # TODO
        self.cropper = Cropper(capture_thresh=fps // 2, save_dir="cropped_images")
        self.box_annotator = sv.BoundingBoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(text_scale=0.5, text_padding=2)
        self.embedder = imgbeddings()

    def __call__(self, frame, mode="analyze"):
        detections = self.detect(frame)
        embeddings = np.zeros(shape=(len(detections), 768), dtype=np.float32)
        for idx, detection in enumerate(detections):
            left, top, right, bottom = map(int, detection[:4])
            cropped_img = frame[top:bottom, left:right]
            img_cropped = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
            img_embedding = self.embedder.to_embeddings(Image.fromarray(img_cropped))[0]
            embeddings[idx] = img_embedding
        detections = self.tracker(detections, embeddings)
        if mode == "analyze":
            cropped_image_list = self.cropper(tracker=self.tracker, image=frame)
            return detections, cropped_image_list
        else:
            return detections

    def detect(self, image):
        det = self.model(
            image, stream=True, verbose=False, device=self.device, conf=0.15
        )
        return next(det).boxes.data.cpu().numpy()

    def visualize(self, original_image, detections):
        detections = to_supervision_detections(detections)
        annotated_frame = self.box_annotator.annotate(
            scene=original_image,
            detections=detections,
        )
        annotated_frame = self.label_annotator.annotate(
            scene=annotated_frame, detections=detections, labels=make_labels(detections)
        )
        return annotated_frame


__all__ = ("Detector",)
