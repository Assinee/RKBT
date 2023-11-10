import math
from typing import Optional, List

from bytetrack_inference_demo_yolov5_with_speed import Detection

def calculate_speed(prev_x, prev_y, curr_x, curr_y, duration):
  dx = curr_x - prev_x
  dy = curr_y - prev_y
  distance = math.hypot(dx, dy)
  speed = distance / duration

  return speed

def add_speed(detections: List[Detection],previous_detections: List[Detection], duration: int) -> List[Detection]:
    result = []
    for detection in detections:
            if detection.tracker_id is not None:
                for previous_detection in previous_detections:
                    if(detection.tracker_id==previous_detection.tracker_id):
                        detection.speed=calculate_speed(previous_detection.rect.x,previous_detection.rect.y,detection.rect.x,detection.rect.y,duration)
            result.append(detection)
    return result

# resolves which player is currently in ball possession based on player-ball proximity
def get_player_in_possession(
    player_detections: List[Detection],
    ball_detections: List[Detection],
    proximity: int
) -> Optional[Detection]:
    if len(ball_detections) != 1:
        return None
    ball_detection = ball_detections[0]
    for player_detection in player_detections:
        if player_detection.rect.pad(proximity).contains_point(point=ball_detection.rect.center):
            return player_detection