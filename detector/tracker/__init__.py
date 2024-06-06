# SPDX-FileCopyrightText: 2024-present dh031200 <imbird0312@gmail.com>
#
# SPDX-License-Identifier: MIT

from .bytetrack import BYTETracker
from .sfsort import SFSORT, gen_track_args


class Tracker:
    def __init__(self, method=None, track_thresh=0.15, width=None, height=None, fps=None):
        if method == "bytetrack":
            tracker = BYTETracker(track_thresh=track_thresh)
        elif method == 'SFSORT':
            if all([width, height, fps]):
                tracker = SFSORT(gen_track_args(width=width, height=height, fps=fps))
            else:
                msg = "SFSORT needs width, height, fps"
                raise ValueError(msg)
        else:
            tracker = BYTETracker(track_thresh=track_thresh)
        self.tracker = tracker

    def __call__(self, detections, embeddings):
        return self.tracker.update(detections, embeddings)

    @property
    def active_tracks(self):
        return self.tracker.get_active_tracks()

    @property
    def frame_id(self):
        return self.tracker.frame_id

    def get_origin(self):
        return [i.init_coords for i in self.tracker.tracked_stracks]


__all__ = ("Tracker",)
