# Code From https://github.com/ifzhang/ByteTrack

from collections import OrderedDict
from typing import ClassVar

import numpy as np


class TrackState:
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class BaseTrack:
    _count = 0

    track_id = 0
    is_activated = False
    state = TrackState.New

    history: ClassVar[OrderedDict] = OrderedDict()
    features: ClassVar[list] = []
    curr_feature = None
    score = 0
    start_frame = 0
    frame_id = 0
    time_since_update = 0

    # multi-camera
    location = (np.inf, np.inf)

    @staticmethod
    def reset():
        BaseTrack._count = 0

        BaseTrack.track_id = 0
        BaseTrack.is_activated = False
        BaseTrack.state = TrackState.New

        BaseTrack.history = OrderedDict()
        BaseTrack.features = []
        BaseTrack.curr_feature = None
        BaseTrack.score = 0
        BaseTrack.start_frame = 0
        BaseTrack.frame_id = 0
        BaseTrack.time_since_update = 0

        # multi-camera
        BaseTrack.location = (np.inf, np.inf)

    @property
    def end_frame(self):
        return self.frame_id

    @staticmethod
    def next_id():
        BaseTrack._count += 1
        return BaseTrack._count

    def activate(self, *args):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def update(self, *args, **kwargs):
        raise NotImplementedError

    def mark_lost(self):
        self.state = TrackState.Lost

    def mark_removed(self):
        self.state = TrackState.Removed
