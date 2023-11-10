class Pipeline:
  def __init__(self, detector, speed, tracking,possession):
    self.detector = detector
    self.tracking = tracking
    self.speed = speed 
    self.possession = possession

  def __call__(self,SOURCE_VIDEO_PATH):
    total_frames = 750 # Optional: Specify the total number of frames to process
    all_detections, ball_detections, referee_detections, goalkeeper_detections, player_detections, player_goalkeeper_detections, detections_for_tracking = self.detector.apply_to_video(SOURCE_VIDEO_PATH, total_frames)
    player_in_possession = self.possession.detect_player_in_possession(player_detections, ball_detections)
    tracked_detections = self.tracking.update(SOURCE_VIDEO_PATH, detections_for_tracking, total_frames)
    results_with_speed = self.speed.add_speed(tracked_detections, SOURCE_VIDEO_PATH)
    return results_with_speed, tracked_detections, player_in_possession,all_detections,referee_detections,goalkeeper_detections,player_goalkeeper_detections