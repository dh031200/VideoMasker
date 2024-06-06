# SPDX-FileCopyrightText: 2024-present dh031200 <imbird0312@gmail.com>
#
# SPDX-License-Identifier: MIT
from collections import defaultdict

import cv2
import torch
import numpy as np
import supervision as sv
from ultralytics import YOLO
from imgbeddings import imgbeddings
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

from .tracker import Tracker
from .cropper import Cropper
from .utils import to_supervision_detections, make_labels


class EmbedDetector:
    def __init__(self):
        model_path = "yolov8s-oiv7-face.pt"
        device = "mps"
        self.model = YOLO(model_path, task="detect")
        self.device = device

        self.box_annotator = sv.BoundingBoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(text_scale=0.5, text_padding=2)
        self.embedder = imgbeddings()

        self.face_embeddings_dict = defaultdict(int)
        self.face_embeddings_group = defaultdict(list)
        self.face_embedding_list = []
        self.face_image_list = []
        self.box_annotator = sv.BoundingBoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(text_scale=0.5, text_padding=2)
        self.face_idx = 0
        self.global_idx = 0

    def __call__(self, frame):
        detections = self.detect(frame)
        detections = self.track(frame, detections)
        return detections

    def detect(self, frame):
        det = self.model(frame, stream=True, verbose=False, device=self.device)
        return next(det).boxes.data.cpu().numpy()

    def get_embedding(self, img):
        img_cropped = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_embedding = self.embedder.to_embeddings(Image.fromarray(img_cropped))
        return img_embedding[0]

    def track(self, frame, detections):
        dets = []
        for detection in detections:
            l, t, r, b = map(int, detection[:4])
            cropped_img = frame[t:b, l:r]
            embedding = self.get_embedding(cropped_img)
            max_similarity = 0
            matched_face_idx = self.face_idx
            if len(self.face_embedding_list):
                similarity = cosine_similarity(self.face_embedding_list, [embedding])
                max_similarity_idx = similarity.argmax()
                max_similarity = similarity[max_similarity_idx][0]
                if max_similarity > 0.9:
                    matched_face_idx = self.face_embeddings_dict[max_similarity_idx]
                    self.face_embeddings_group[matched_face_idx].append(self.global_idx)
                else:
                    self.face_idx += 1

            else:
                self.face_idx += 1
            self.face_embeddings_dict[self.global_idx] = matched_face_idx
            self.face_embeddings_group[self.face_idx].append(self.global_idx)
            self.face_embedding_list.append(embedding)
            self.global_idx += 1
            dets.append([l, t, r, b, max_similarity, 0, matched_face_idx])
        return np.array(dets)

    def visualize(self, original_image, detections):
        if len(detections):
            detections = to_supervision_detections(detections)

            annotated_frame = self.box_annotator.annotate(
                scene=original_image,
                detections=detections,
            )
            annotated_frame = self.label_annotator.annotate(
                scene=annotated_frame,
                detections=detections,
                labels=make_labels(detections),
            )
        else:
            annotated_frame = original_image
        return annotated_frame


# class Detector:
#     def __init__(self):
#         self.detector_backend = "yolov8"
#         # self.detector_backend = "centerface"
#         # self.detector_backend = "yunet"
#         self.model_name = "Facenet512"
#         # self.model_name = "VGG-Face"
#         self.normalization = "Facenet"
#         # self.normalization = "VGGFace2"
#         self.face_embeddings_dict = defaultdict(int)
#         self.face_embeddings_group = defaultdict(list)
#         self.face_embedding_list = []
#         self.face_image_list = []
#         self.box_annotator = sv.BoundingBoxAnnotator(thickness=2)
#         self.label_annotator = sv.LabelAnnotator(text_scale=0.5, text_padding=2)
#         self.face_idx = 0
#         self.global_idx = 0
#
#     def __call__(self, frame):
#         detections = self.detect(frame)
#         detections = self.track(detections)
#         return detections
#
#     def detect(self, frame):
#         return DeepFace.represent(
#             img_path=frame,
#             enforce_detection=False,
#             detector_backend=self.detector_backend,
#             model_name=self.model_name,
#             normalization=self.normalization,
#         )
#
#     def track(self, detections):
#         dets = []
#         for detection in detections:
#             if detection['face_confidence'] < 0.7:
#                 continue
#             embedding = detection['embedding']
#             max_similarity = 0
#             matched_face_idx = self.face_idx
#             if len(self.face_embedding_list):
#                 similarity = cosine_similarity(self.face_embedding_list, [embedding])
#                 max_similarity_idx = similarity.argmax()
#                 max_similarity = similarity[max_similarity_idx][0]
#                 print("max_similarity")
#                 print(max_similarity)
#                 if max_similarity > 0.8:
#                     matched_face_idx = self.face_embeddings_dict[max_similarity_idx]
#                     self.face_embeddings_group[matched_face_idx].append(self.global_idx)
#                 else:
#                     self.face_idx += 1
#                     print("similarity")
#                     print(similarity)
#                     print("detection")
#                     print(detection)
#
#             else:
#                 self.face_idx += 1
#             self.face_embeddings_dict[self.global_idx] = matched_face_idx
#             self.face_embeddings_group[self.face_idx].append(self.global_idx)
#             self.face_embedding_list.append(embedding)
#             self.global_idx += 1
#             area = detection['facial_area']
#             x, y, w, h = area['x'], area['y'], area['w'], area['h']
#             dets.append([x, y, x + w, y + h, max_similarity, 0, matched_face_idx])
#         return np.array(dets)
#
#     def visualize(self, original_image, detections):
#         if len(detections):
#             detections = to_supervision_detections(detections)
#
#             annotated_frame = self.box_annotator.annotate(
#                 scene=original_image,
#                 detections=detections,
#             )
#             annotated_frame = self.label_annotator.annotate(
#                 scene=annotated_frame, detections=detections, labels=make_labels(detections)
#             )
#         else:
#             annotated_frame = original_image
#         return annotated_frame


class Detector:
    def __init__(self, fps, width, height):
        if torch.backends.mps.is_available() and False:
            model_path = "model/yolov8s-oiv7-face.mlpackage"
            device = "mps"
        elif torch.cuda.is_available() and False:
            model_path = "model/yolov8s-oiv7-face.pt"
            device = "cuda"
        else:
            model_path = "model/yolov8s-oiv7-face.pt"
            device = "mps"
        self.model = YOLO(model_path, task="detect")
        self.device = device
        self.tracker = Tracker(method="bytetrack")
        # self.tracker = Tracker(method="SFSORT", fps=fps, width=width, height=height)
        self.cropper = Cropper(capture_thresh=fps // 2, save_dir="cropped_images")
        self.box_annotator = sv.BoundingBoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(text_scale=0.5, text_padding=2)
        self.embedder = imgbeddings()

    def __call__(self, frame, mode="analyze"):
        detections = self.detect(frame)
        embeddings = np.zeros(shape=(len(detections), 768), dtype=np.float32)
        for idx, detection in enumerate(detections):
            l, t, r, b = map(int, detection[:4])
            cropped_img = frame[t:b, l:r]
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
