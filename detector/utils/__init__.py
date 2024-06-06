import numpy as np
from supervision import Detections

TRACK_SHAPE = 7


def to_supervision_detections(detections, *, class_agnostic=False):
    """
    :param detections: np.array([x1, y1, x2, y2, conf, class, track_id])
    :param class_agnostic: bool
    :return:Detections
    """
    if not isinstance(detections, np.ndarray):
        detections.numpy()

    detections[:, 4] *= 100
    detections = detections.astype(int)

    return Detections(
        xyxy=detections[:, :4],
        confidence=detections[:, 4],
        class_id=(
            detections[:, 5]
            if all([not class_agnostic, detections.shape[1]])
            else detections[:, 6]
        ),
        tracker_id=detections[:, 6] if detections.shape[1] == TRACK_SHAPE else None,
    )


def make_labels(detections):
    class_names = {0: "face"}

    if detections.tracker_id is None:
        labels = [
            f"{class_names[class_id]} {conf}%"
            for class_id, conf in zip(detections.class_id, detections.confidence)
        ]
    else:
        labels = [
            f"{tracker_id} {class_names[class_id]} {conf}%"
            for tracker_id, class_id, conf in zip(
                detections.tracker_id, detections.class_id, detections.confidence
            )
        ]
    return labels


__all__ = (
    "to_supervision_detections",
    "make_labels",
)
