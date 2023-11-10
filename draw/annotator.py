from config import BALL_COLOR, BALL_MARKER_FILL_COLOR, PLAYER_COLOR, PLAYER_MARKER_FILL_COLOR, REFEREE_COLOR, THICKNESS
from utils import BaseAnnotator, Color, MarkerAnntator, TextAnnotator, VideoConfig, generate_frames, get_video_writer
from tqdm.notebook import tqdm


class Annotator:
    def __init__(self, video_file, total_frames, target_video_path):
        self.video_file = video_file
        self.total_frames = total_frames
        self.target_video_path = target_video_path
        self.frame_iterator = iter(generate_frames(self.video_file))

        # Initialize video writer and annotators
        self.video_config = VideoConfig(fps=30, width=1920, height=1080)
        self.video_writer = get_video_writer(target_video_path=self.target_video_path, video_config=self.video_config)

        self.base_annotator = BaseAnnotator(
            colors=[BALL_COLOR, PLAYER_COLOR, PLAYER_COLOR, REFEREE_COLOR],
            thickness=THICKNESS)

        self.player_goalkeeper_text_annotator = TextAnnotator(
            PLAYER_COLOR, text_color=Color(255, 255, 255), text_thickness=2)

        self.referee_text_annotator = TextAnnotator(
            REFEREE_COLOR, text_color=Color(0, 0, 0), text_thickness=2)

        self.ball_marker_annotator = MarkerAnntator(color=BALL_MARKER_FILL_COLOR)
        self.player_marker_annotator = MarkerAnntator(color=PLAYER_MARKER_FILL_COLOR)
        self.player_in_possession_marker_annotator = MarkerAnntator(color=PLAYER_MARKER_FILL_COLOR)

    def annotate_video(self, tracked_detections, tracked_goalkeeper_detections, tracked_player_detections, tracked_referee_detections, ball_detections, player_in_possession_detection):
        frame_iterator = iter(generate_frames(self.video_file))

        for idx, frame in enumerate(tqdm(frame_iterator, total=self.total_frames)):
            # Annotate video frame
            annotated_image = frame.copy()
            annotated_image = self.base_annotator.annotate(
                image=annotated_image,
                detections=tracked_detections[idx])

            annotated_image = self.player_goalkeeper_text_annotator.annotate(
                image=annotated_image,
                detections=tracked_goalkeeper_detections[idx] + tracked_player_detections[idx])

            annotated_image = self.referee_text_annotator.annotate(
                image=annotated_image,
                detections=tracked_referee_detections[idx])

            annotated_image = self.ball_marker_annotator.annotate(
                image=annotated_image,
                detections=ball_detections[idx])

            annotated_image = self.player_marker_annotator.annotate(
                image=annotated_image,
                detections=[player_in_possession_detection[idx]] if player_in_possession_detection[idx] else [])

            self.video_writer.write(annotated_image)

        self.video_writer.release()