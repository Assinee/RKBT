from services import Detector, SpeedCalculator,Tracker,PossessionDetector
from services.pipeline import Pipeline
from config import SOURCE_VIDEO_PATH, WEIGHTS_PATH


detector = Detector(WEIGHTS_PATH)
tracker = Tracker()
speed_calculator = SpeedCalculator()
possession_detector = PossessionDetector(proximity=50)

# annotator = Annotator()
# possessionDetector = PossessionDetector()

if __name__ == '__main__':
    pipeline = Pipeline(
        detector=detector,
        speed=speed_calculator,
        tracking=tracker,
        possession=possession_detector
    )

    (
        results_with_speed,
        tracked_detections,
        player_in_possession,
        all_detections,
        referee_detections,
        goalkeeper_detections,
        player_goalkeeper_detections
    ) = pipeline(SOURCE_VIDEO_PATH)
    