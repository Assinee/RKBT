from bytetrack_inference_demo_yolov5_with_speed import Detection
from processing import generate_frames
import torch

WEIGHTS_PATH = "../best.pt"
SOURCE_VIDEO_PATH = "../clips/08fd33_4.mp4"
frame_iterator = iter(generate_frames(video_file=SOURCE_VIDEO_PATH))
frame = next(frame_iterator)
model = torch.hub.load('ultralytics/yolov5', 'custom', WEIGHTS_PATH, device=0)


