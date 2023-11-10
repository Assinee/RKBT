import sys
import numpy as np
from dataclasses import dataclass
from typing import List
from ByteTrack import BYTETracker, Detection, STrack
from utils import generate_frames
from onemetric.cv.utils.iou import box_iou_batch
from tqdm.notebook import tqdm

@dataclass(frozen=True)
class BYTETrackerArgs:
    track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = False

class Tracker:
    def __init__(self):
        sys.path.append("./ByteTrack")
        self.byte_tracker = BYTETracker(BYTETrackerArgs())

    def detections2boxes(self, detections: List[Detection], with_confidence: bool = True) -> np.ndarray:
        return np.array([
            [
                detection.rect.top_left.x,
                detection.rect.top_left.y,
                detection.rect.bottom_right.x,
                detection.rect.bottom_right.y,
                detection.confidence
            ] if with_confidence else [
                detection.rect.top_left.x,
                detection.rect.top_left.y,
                detection.rect.bottom_right.x,
                detection.rect.bottom_right.y
            ]
            for detection
            in detections
        ], dtype=float)

    def tracks2boxes(self, tracks: List[STrack]) -> np.ndarray:
        return np.array([
            track.tlbr
            for track
            in tracks
        ], dtype=float)

    def match_detections_with_tracks(self, detections: List[Detection], tracks: List[STrack]) -> List[Detection]:
        detection_boxes = self.detections2boxes(detections=detections, with_confidence=False)
        tracks_boxes = self.tracks2boxes(tracks=tracks)
        iou = box_iou_batch(tracks_boxes, detection_boxes)
        track2detection = np.argmax(iou, axis=1)

        for tracker_index, detection_index in enumerate(track2detection):
            if iou[tracker_index, detection_index] != 0:
                detections[detection_index].tracker_id = tracks[tracker_index].track_id
        return detections

    def update(self, video_file, detections , total_frames=750):
        frame_iterator = iter(generate_frames(video_file))
        tracked_detections = []
        for idx, frame in enumerate(tqdm(frame_iterator, total= total_frames)):
            tracks = self.byte_tracker.update(
                output_results= self.detections2boxes(detections=detections[idx]),
                img_info=frame.shape,
                img_size=frame.shape
            )
            tracked_detections.append (self.match_detections_with_tracks(detections=detections[idx], tracks=tracks))
        return tracked_detections

