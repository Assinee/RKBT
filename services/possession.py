from dataclasses import dataclass
from typing import List, Optional

from utils import Detection


@dataclass
class PossessionDetector:

    def __init__(self, proximity: int = 50):
        self.proximity = proximity

    def detect_player_in_possession(
        self,
        all_player_detections: List[List[Detection]],
        all_ball_detections: List[List[Detection]],
    ) -> Optional[Detection]:
        all_player_in_possession=[]
        for idx, ball_detections in enumerate(all_ball_detections):
            if len(ball_detections) != 1:
                all_player_in_possession.append(None)

            ball_detection = ball_detections[0]

            for player_detection in all_player_detections[idx]:
                if player_detection.rect.pad(self.proximity).contains_point(
                    ball_detection.rect.center
                ):
                    all_player_in_possession.append(player_detection)

            all_player_in_possession.append(None)
        return all_player_in_possession
