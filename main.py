# main.py

from detect import detect_frame, Detection
from track import BYTETracker, match_detections 
from annotate import annotate_frame
from video import VideoConfig, get_video_writer

import cv2

# Initialize modules
model = load_detection_model() 
byte_tracker = BYTETracker()
annotator = Annotator()  

video_config = VideoConfig(fps=30, width=1920, height=1080)
video_writer = get_video_writer('output.mp4', video_config)

cap = cv2.VideoCapture('input.mp4')

while cap.isOpened():

  # Get next frame
  success, frame = cap.read()
  if not success:
    break
  
  # Detect objects
  detections = detect_frame(frame, model)

  # Track objects
  tracks = byte_tracker.update(frame_shape=frame.shape)
  tracked_detections = match_detections(detections, tracks)

  # Annotate frame  
  annotated_frame = annotate_frame(frame, tracked_detections)

  # Write frame
  video_writer.write(annotated_frame)

cap.release()
video_writer.release()