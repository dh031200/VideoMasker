# Base code From https://github.com/ifzhang/ByteTrack
# Modify by dh031200 <imbird0312@gmail.com>
import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .basetrack import BaseTrack, TrackState
from .kalman_filter import KalmanFilter
from .matching import fuse_score, iou_distance, linear_assignment


class STrack(BaseTrack):
    shared_kalman = KalmanFilter()

    def __init__(self, tlwh, score, cls, embedding):
        super().__init__()
        self._tlwh = np.asarray(tlwh, dtype=float)
        self.init_coords = self.get_init_coords(self._tlwh)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.cls = cls
        self.tracklet_len = 0
        self.embedding = embedding

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, *, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()

        self._tlwh = new_track._tlwh
        self.embedding = new_track.embedding
        self.score = new_track.score

    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self._tlwh = new_track._tlwh
        self.embedding = new_track.embedding
        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
        width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @staticmethod
    def get_init_coords(tlwh):
        _tlwh = tlwh.copy()
        _tlwh[2:] += _tlwh[:2]
        return _tlwh

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self._tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return f"OT_{self.track_id}_({self.start_frame}-{self.end_frame})"


class BYTETracker:
    MIN_SCORE = 0.1
    MAX_PDIST = 0.15

    def __init__(self, track_thresh=0.5, match_thresh=0.9, track_buffer=60, frame_rate=30):
        BaseTrack.reset()
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.track_thresh = track_thresh
        self.det_thresh = track_thresh + 0.1
        self.match_thresh = match_thresh
        self.buffer_size = int(frame_rate / 30.0 * track_buffer)
        self.max_time_lost = self.buffer_size * 4
        self.kalman_filter = KalmanFilter()

    def get_tracks(self):
        tracks = np.zeros(shape=(len(self.tracked_stracks), 7))
        for idx, track in enumerate(self.tracked_stracks):
            tracks[idx][:4] = track.tlbr
            tracks[idx][4] = track.score
            tracks[idx][5] = track.cls
            tracks[idx][6] = track.track_id

        return tracks

    def update(self, output_results, embeddings):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        bboxes = output_results[:, :4]
        scores = output_results[:, 4]
        classes = output_results[:, 5]

        remain_inds = scores > self.track_thresh
        inds_low = scores > self.MIN_SCORE
        inds_high = scores < self.track_thresh

        inds_second = np.logical_and(inds_low, inds_high)

        dets = bboxes[remain_inds]
        clsses = classes[remain_inds]
        scores_keep = scores[remain_inds]
        embeddings_keep = embeddings[remain_inds]
        dets_second = bboxes[inds_second]
        clsses_second = classes[inds_second]
        scores_second = scores[inds_second]
        embeddings_second = embeddings[inds_second]

        if len(dets) > 0:
            """Detections"""
            detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s, c, e) for (tlbr, s, c, e) in
                          zip(dets, scores_keep, clsses, embeddings_keep)]
        else:
            detections = []

        """ Add newly detected tracklets to tracked_stracks"""
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        """ Step 2: First association, with high score detection boxes"""
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        STrack.multi_predict(strack_pool)
        dists = iou_distance(strack_pool, detections)
        dists = fuse_score(dists, detections)
        matches, u_track, u_detection = linear_assignment(dists, thresh=self.match_thresh)

        unmatched_tracks = []
        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            z = cosine_similarity([track.embedding], [det.embedding]).squeeze()
            if z > 0.93:
                if track.state == TrackState.Tracked:
                    track.update(detections[idet], self.frame_id)
                    activated_starcks.append(track)
                else:
                    track.re_activate(det, self.frame_id, new_id=False)
                    refind_stracks.append(track)
            else:
                unmatched_tracks.append(det)
                u_track = np.append(u_track, itracked)
                u_detection = np.append(u_detection, idet)


        """ Step 3: Second association, with low score detection boxes"""
        # association the untrack to the low score detections
        if len(dets_second) > 0:
            """Detections"""
            detections_second = [
                STrack(STrack.tlbr_to_tlwh(tlbr), s, c, e)
                for (tlbr, s, c, e) in zip(dets_second, scores_second, clsses_second, embeddings_second)
            ]
        else:
            detections_second = []

        detections_second += unmatched_tracks
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]

        dists = iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = linear_assignment(dists, thresh=0.5)
        unmatched_tracks = []
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            z = cosine_similarity([track.embedding], [det.embedding]).squeeze()
            if z > 0.89:
                if track.state == TrackState.Tracked:
                    track.update(det, self.frame_id)
                    activated_starcks.append(track)
                else:
                    track.re_activate(det, self.frame_id, new_id=False)
                    refind_stracks.append(track)
            else:
                unmatched_tracks.append(det)
                u_track = np.append(u_track, itracked)
                u_detection = np.append(u_detection, idet)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        """Deal with unconfirmed tracks, usually tracks with only one beginning frame"""
        detections = [detections[i] for i in u_detection]

        detections += unmatched_tracks

        dists = iou_distance(unconfirmed, detections)
        dists = fuse_score(dists, detections)
        matches, u_unconfirmed, u_detection = linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])

        refined_lost_track = []
        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            if len(lost_stracks) :
                for idx, a_track in enumerate(lost_stracks):
                    z = cosine_similarity([a_track.embedding], [track.embedding]).squeeze()
                    if z > 0.9:
                        a_track.re_activate(track, self.frame_id, new_id=False)
                        refind_stracks.append(a_track)
                        refined_lost_track.append(idx)
                    # else:
                    #     track.activate(self.kalman_filter, self.frame_id)
                    #     activated_starcks.append(track)
                    # refined_lost_track.append()

            else:
                track.activate(self.kalman_filter, self.frame_id)
                activated_starcks.append(track)
        lost_stracks = [strack for idx, strack in enumerate(lost_stracks) if idx not in refined_lost_track]

        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)

        return self.get_tracks()

    def get_active_tracks(self):
        return self.tracked_stracks


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < BYTETracker.MAX_PDIST)
    dupa, dupb = [], []
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if i not in dupa]
    resb = [t for i, t in enumerate(stracksb) if i not in dupb]
    return resa, resb
