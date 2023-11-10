import math
from typing import List
from utils import Detection
from moviepy.editor import VideoFileClip


class SpeedCalculator:
    def __init__(self):
        pass

    @staticmethod
    def calculate_speed(prev_x, prev_y, curr_x, curr_y, duration):
        dx = curr_x - prev_x
        dy = curr_y - prev_y
        distance = math.hypot(dx, dy)
        speed = distance / duration
        return speed

    @staticmethod
    def add_speed(self,detections: List[List[Detection]], SOURCE_VIDEO_PATH ) -> List[List[Detection]]:
        video = VideoFileClip(SOURCE_VIDEO_PATH)
        duration = video.duration
        all_result = []
        for idx in range(1, len(detections)):  # Corrected the range
            result = []
            for detection in detections[idx]:
                if detection.tracker_id is not None:
                    for previous_detection in detections[idx - 1]:
                        if detection.tracker_id == previous_detection.tracker_id:
                            detection.speed = self.calculate_speed(
                                previous_detection.rect.x,
                                previous_detection.rect.y,
                                detection.rect.x,
                                detection.rect.y,
                                duration
                            )
                result.append(detection)
            all_result.append(result)
        return all_result