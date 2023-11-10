from bytetrack_inference_demo_yolov5_with_speed import Detection
from processing import generate_frames
import torch

def loadyolo(WEIGHTS_PATH,SOURCE_VIDEO_PATH):
    frame_iterator = iter(generate_frames(video_file=SOURCE_VIDEO_PATH))
    model = torch.hub.load('ultralytics/yolov5', 'custom', WEIGHTS_PATH, device=0)
    return model, frame_iterator

